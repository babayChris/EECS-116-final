// vio_ekf: a compact MSCKF-style visual-inertial filter for VIOBench, without OpenCV.
//
// State (error state, world = IMU frame at the first frame, so the output needs no alignment):
//   attitude (3, world-frame perturbation), position (3), velocity (3), gyro bias (3), accel bias (3),
//   gravity vector in world (3), then P_CLONES IMU poses (attitude 3 + position 3 each).
// Front end: green channel block mean (P_DS), Shi-Tomasi corners on a grid, translation-only LK on small patches with
// the IMU-predicted motion as the initial guess. Updates: features observed in >= P_MIN_OBS clones are triangulated
// (anchored inverse depth, Gauss-Newton), their Jacobians projected onto the left null space of the point Jacobian
// (the MSCKF update of Mourikis & Roumeliotis 2007), gated, stacked, compressed and applied in one EKF update.
// Ideas from the class's stage 2 presentations: filter with IMU prediction and patch-pixel correction, gravity in
// the state for in-flight starts, green-channel sparse image access, few patches and few saved poses.
#include <viobench_abi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "params.h"

namespace {

constexpr int NI = 18;                        // IMU part of the error state
constexpr int NMAX = NI + 6 * P_CLONES;       // full error state
constexpr int W0 = 320, H0 = 240;
constexpr int GW = W0 / P_DS, GH = H0 / P_DS; // gray image size
constexpr double GRAVITY = 9.81;
constexpr int MAX_OBS = P_CLONES;
constexpr int MAX_ROWS = P_MAX_UPDATE_FEAT * (2 * MAX_OBS - 2);

// ------------------------------------------------------------------ small linear algebra
struct V3 {
    double x, y, z;
    V3 operator+(const V3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    V3 operator-(const V3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    V3 operator*(double s) const { return {x * s, y * s, z * s}; }
    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
inline double dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3& a, const V3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double norm(const V3& a) { return std::sqrt(dot(a, a)); }

struct M3 {
    double m[3][3];
    static M3 eye() { M3 r{}; r.m[0][0] = r.m[1][1] = r.m[2][2] = 1; return r; }
    static M3 zero() { M3 r{}; return r; }
    V3 operator*(const V3& v) const {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }
    M3 operator*(const M3& o) const {
        M3 r;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j];
        return r;
    }
    M3 operator+(const M3& o) const { M3 r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.m[i][j] = m[i][j] + o.m[i][j]; return r; }
    M3 operator*(double s) const { M3 r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.m[i][j] = m[i][j] * s; return r; }
    M3 t() const { M3 r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.m[i][j] = m[j][i]; return r; }
};
inline M3 skew(const V3& v) { M3 r{}; r.m[0][1] = -v.z; r.m[0][2] = v.y; r.m[1][0] = v.z; r.m[1][2] = -v.x; r.m[2][0] = -v.y; r.m[2][1] = v.x; return r; }

M3 expso3(const V3& w) {
    const double th2 = dot(w, w);
    const M3 K = skew(w);
    double a, b;
    if (th2 < 1e-12) { a = 1 - th2 / 6; b = 0.5 - th2 / 24; }
    else { const double th = std::sqrt(th2); a = std::sin(th) / th; b = (1 - std::cos(th)) / th2; }
    return M3::eye() + K * a + (K * K) * b;
}

void orthonormalize(M3& R) {  // one Gram-Schmidt pass on the rows
    V3 r0{R.m[0][0], R.m[0][1], R.m[0][2]}, r1{R.m[1][0], R.m[1][1], R.m[1][2]};
    r0 = r0 * (1.0 / norm(r0));
    r1 = r1 - r0 * dot(r0, r1);
    r1 = r1 * (1.0 / norm(r1));
    const V3 r2 = cross(r0, r1);
    const V3 rows[3] = {r0, r1, r2};
    for (int i = 0; i < 3; i++) { R.m[i][0] = rows[i].x; R.m[i][1] = rows[i].y; R.m[i][2] = rows[i].z; }
}

void quaternion(const M3& R, double q[4]) {  // [qx, qy, qz, qw]
    const double tr = R.m[0][0] + R.m[1][1] + R.m[2][2];
    if (tr > 0) {
        const double s = std::sqrt(tr + 1.0) * 2;
        q[3] = 0.25 * s; q[0] = (R.m[2][1] - R.m[1][2]) / s; q[1] = (R.m[0][2] - R.m[2][0]) / s; q[2] = (R.m[1][0] - R.m[0][1]) / s;
    } else if (R.m[0][0] > R.m[1][1] && R.m[0][0] > R.m[2][2]) {
        const double s = std::sqrt(1.0 + R.m[0][0] - R.m[1][1] - R.m[2][2]) * 2;
        q[3] = (R.m[2][1] - R.m[1][2]) / s; q[0] = 0.25 * s; q[1] = (R.m[0][1] + R.m[1][0]) / s; q[2] = (R.m[0][2] + R.m[2][0]) / s;
    } else if (R.m[1][1] > R.m[2][2]) {
        const double s = std::sqrt(1.0 + R.m[1][1] - R.m[0][0] - R.m[2][2]) * 2;
        q[3] = (R.m[0][2] - R.m[2][0]) / s; q[0] = (R.m[0][1] + R.m[1][0]) / s; q[1] = 0.25 * s; q[2] = (R.m[1][2] + R.m[2][1]) / s;
    } else {
        const double s = std::sqrt(1.0 + R.m[2][2] - R.m[0][0] - R.m[1][1]) * 2;
        q[3] = (R.m[1][0] - R.m[0][1]) / s; q[0] = (R.m[0][2] + R.m[2][0]) / s; q[1] = (R.m[1][2] + R.m[2][1]) / s; q[2] = 0.25 * s;
    }
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int i = 0; i < 4; i++) q[i] /= n;
}

// ------------------------------------------------------------------ image front end
struct Image {
    uint8_t px[GH][GW];
};

void make_gray(const uint8_t* rgb, Image& out) {
    for (int y = 0; y < GH; y++) {
        for (int x = 0; x < GW; x++) {
            int s = 0;
            for (int dy = 0; dy < P_DS; dy++) {
                const uint8_t* row = rgb + ((y * P_DS + dy) * W0 + x * P_DS) * 3 + 1;
                for (int dx = 0; dx < P_DS; dx++) s += row[3 * dx];
            }
            out.px[y][x] = (uint8_t)(s / (P_DS * P_DS));
        }
    }
}

inline float bilinear(const Image& im, float x, float y) {
    const int ix = (int)x, iy = (int)y;
    const float fx = x - ix, fy = y - iy;
    const uint8_t* r0 = &im.px[iy][ix];
    const uint8_t* r1 = &im.px[iy + 1][ix];
    return (1 - fy) * ((1 - fx) * r0[0] + fx * r0[1]) + fy * ((1 - fx) * r1[0] + fx * r1[1]);
}

// translation-only inverse-compositional LK of the patch around (x0, y0) in `a` into `b`, starting at (x, y)
bool track_lk(const Image& a, const Image& b, float x0, float y0, float& x, float& y) {
    constexpr int H = P_HALF, N = (2 * H + 1) * (2 * H + 1);
    if (x0 < H + 1 || y0 < H + 1 || x0 > GW - H - 2 || y0 > GH - H - 2) return false;
    float T[N], gx[N], gy[N];
    float gxx = 0, gxy = 0, gyy = 0;
    int k = 0;
    for (int dy = -H; dy <= H; dy++)
        for (int dx = -H; dx <= H; dx++, k++) {
            const float px = x0 + dx, py = y0 + dy;
            T[k] = bilinear(a, px, py);
            gx[k] = 0.5f * (bilinear(a, px + 1, py) - bilinear(a, px - 1, py));
            gy[k] = 0.5f * (bilinear(a, px, py + 1) - bilinear(a, px, py - 1));
            gxx += gx[k] * gx[k]; gxy += gx[k] * gy[k]; gyy += gy[k] * gy[k];
        }
    const float det = gxx * gyy - gxy * gxy;
    if (det < 1e-3f * N * N) return false;
    const float i00 = gyy / det, i01 = -gxy / det, i11 = gxx / det;
    float err = 0;
    for (int it = 0; it < P_LK_ITERS; it++) {
        if (x < H + 1 || y < H + 1 || x > GW - H - 2 || y > GH - H - 2) return false;
        float bx = 0, by = 0;
        err = 0;
        k = 0;
        for (int dy = -H; dy <= H; dy++)
            for (int dx = -H; dx <= H; dx++, k++) {
                const float e = bilinear(b, x + dx, y + dy) - T[k];
                bx += gx[k] * e; by += gy[k] * e;
                err += std::fabs(e);
            }
        const float ux = i00 * bx + i01 * by, uy = i01 * bx + i11 * by;
        x -= ux; y -= uy;
        if (ux * ux + uy * uy < 0.0009f) break;
    }
    if (x < H + 1 || y < H + 1 || x > GW - H - 2 || y > GH - H - 2) return false;
    return err / N < P_LK_MAX_ERR;
}

float corner_score(const Image& im, int x, int y) {  // Shi-Tomasi min eigenvalue over a 5 x 5 window, per pixel
    float sxx = 0, sxy = 0, syy = 0;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++) {
            const float gx = 0.5f * ((int)im.px[y + dy][x + dx + 1] - (int)im.px[y + dy][x + dx - 1]);
            const float gy = 0.5f * ((int)im.px[y + dy + 1][x + dx] - (int)im.px[y + dy - 1][x + dx]);
            sxx += gx * gx; sxy += gx * gy; syy += gy * gy;
        }
    const float tr = 0.5f * (sxx + syy), d = std::sqrt(0.25f * (sxx - syy) * (sxx - syy) + sxy * sxy);
    return (tr - d) / 25.0f;
}

// ------------------------------------------------------------------ estimator
struct Obs { int clone_id; double u, v; };  // normalized image coordinates

struct Feature {
    int id;
    float x, y;            // pixel position in the gray image, current frame
    int nobs;
    Obs obs[MAX_OBS];
    bool alive;
    bool has_pw;           // a triangulated world point from an earlier track (depth prior)
    V3 pw;
    double sig_rho;        // posterior inverse-depth std of that point
};

struct Clone { int id; M3 R; V3 p; };

double P[NMAX][NMAX];

struct Estimator {
    // calibration
    double fx, fy, cx, cy;      // at the gray resolution
    M3 R_ci; V3 t_ci;           // IMU -> camera
    M3 R_ic; V3 t_ic;           // camera -> IMU
    // state
    M3 R = M3::eye(); V3 p{0, 0, 0}, v{0, 0, 0}, bg{0, 0, 0}, ba{0, 0, 0}, g{0, 0, -GRAVITY};
    Clone clones[P_CLONES]; int nclones = 0, next_clone_id = 0;
    Feature feats[P_MAX_FEAT]; int nfeats = 0, next_feat_id = 0;
    Image img[2]; int cur = 0;
    int frame = 0;
    double t_last = 0;
    V3 w_last{0, 0, 0};
    double depth_est = P_PRED_DEPTH;
    // diagnostics
    int st_tracked = 0, st_lost = 0, st_cand = 0, st_tri_fail = 0, st_chi_fail = 0, st_used = 0, st_rows = 0;

    int dim() const { return NI + 6 * nclones; }

    // ---------------------------------------------------------------- propagation
    void propagate(const double* imu, int count) {
        // Phi blocks (first order, composed per sample) for the rows that change
        M3 th_bg = M3::zero();
        M3 v_th = M3::zero(), v_bg = M3::zero(), v_ba = M3::zero(), v_g = M3::zero();
        M3 p_th = M3::zero(), p_v = M3::zero(), p_bg = M3::zero(), p_ba = M3::zero(), p_g = M3::zero();
        double T = 0;
        for (int i = 0; i < count; i++) {
            const double* r = imu + 7 * i;
            const double dt = r[0] - t_last;
            t_last = r[0];
            if (dt <= 0 || dt > 0.1) continue;
            const V3 w{r[4] - bg.x, r[5] - bg.y, r[6] - bg.z};
            const V3 f{r[1] - ba.x, r[2] - ba.y, r[3] - ba.z};
            w_last = w;
            const V3 a = R * f + g;
            // Phi update: p rows use old v rows, v rows use old theta rows
            const M3 Rdt = R * dt;
            const M3 A = skew(R * f) * (-dt);
            p_th = p_th + v_th * dt; p_v = p_v + M3::eye() * dt; p_bg = p_bg + v_bg * dt; p_ba = p_ba + v_ba * dt; p_g = p_g + v_g * dt;
            v_th = v_th + A; v_bg = v_bg + A * th_bg; v_ba = v_ba + Rdt * (-1.0); v_g = v_g + M3::eye() * dt;
            th_bg = th_bg + Rdt * (-1.0);
            // mean
            p = p + v * dt + a * (0.5 * dt * dt);
            v = v + a * dt;
            R = R * expso3(w * dt);
            T += dt;
        }
        orthonormalize(R);
        // dense Phi for the IMU block: order th(0) p(3) v(6) bg(9) ba(12) g(15)
        static double Phi[NI][NI];
        std::memset(Phi, 0, sizeof(Phi));
        for (int i = 0; i < NI; i++) Phi[i][i] = 1;
        auto put = [&](int r0, int c0, const M3& B) { for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Phi[r0 + i][c0 + j] += B.m[i][j]; };
        put(0, 9, th_bg);
        put(3, 0, p_th); put(3, 6, p_v); put(3, 9, p_bg); put(3, 12, p_ba); put(3, 15, p_g);
        put(6, 0, v_th); put(6, 9, v_bg); put(6, 12, v_ba); put(6, 15, v_g);
        const int n = dim();
        static double tmp[NI][NMAX];
        // tmp = Phi * P[0:NI, :]
        for (int i = 0; i < NI; i++)
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int k = 0; k < NI; k++) s += Phi[i][k] * P[k][j];
                tmp[i][j] = s;
            }
        // P[0:NI,0:NI] = tmp[:,0:NI] Phi^T ; P[0:NI, NI:] = tmp[:, NI:]
        for (int i = 0; i < NI; i++)
            for (int j = 0; j < NI; j++) {
                double s = 0;
                for (int k = 0; k < NI; k++) s += tmp[i][k] * Phi[j][k];
                P[i][j] = s;
            }
        for (int i = 0; i < NI; i++)
            for (int j = NI; j < n; j++) { P[i][j] = tmp[i][j]; P[j][i] = tmp[i][j]; }
        // process noise
        const double qg = P_GYRO_NOISE * P_GYRO_NOISE * T, qa = P_ACC_NOISE * P_ACC_NOISE * T;
        for (int i = 0; i < 3; i++) {
            P[i][i] += qg;
            P[6 + i][6 + i] += qa;
            P[3 + i][3 + i] += qa * T * T / 3;
            P[9 + i][9 + i] += P_GYRO_WALK * P_GYRO_WALK * T;
            P[12 + i][12 + i] += P_ACC_WALK * P_ACC_WALK * T;
        }
    }

    // ---------------------------------------------------------------- clones
    void add_clone() {
        const int n = dim();
        const int c = n;  // new block at the end
        // new clone = (theta, p) of the IMU state: copy rows/cols 0..5
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 6; i++) { P[c + i][j] = P[i][j]; P[j][c + i] = P[j][i]; }
        }
        for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) P[c + i][c + j] = P[i][j];
        clones[nclones] = {next_clone_id++, R, p};
        nclones++;
    }

    void remove_clone(int slot) {
        const int n = dim();
        const int s = NI + 6 * slot;
        // shift rows and columns after the removed block
        for (int i = 0; i < n; i++) {
            if (i >= s && i < s + 6) continue;
            const int ii = i < s ? i : i - 6;
            for (int j = 0; j < n; j++) {
                if (j >= s && j < s + 6) continue;
                const int jj = j < s ? j : j - 6;
                P[ii][jj] = P[i][j];
            }
        }
        for (int k = slot; k < nclones - 1; k++) clones[k] = clones[k + 1];
        nclones--;
    }

    int clone_slot(int id) const {
        for (int k = 0; k < nclones; k++) if (clones[k].id == id) return k;
        return -1;
    }

    // ---------------------------------------------------------------- update
    // Triangulate a feature (anchored inverse depth); fills the world point. Returns false if unusable.
    bool triangulate(const Feature& f, V3& Pw, double* abr = nullptr, M3* Ranchor = nullptr) {  // abr: al, be, rho, posterior std, prior std
        const int n = f.nobs;
        M3 Rwc[MAX_OBS]; V3 pwc[MAX_OBS];
        for (int k = 0; k < n; k++) {
            const Clone& c = clones[clone_slot(f.obs[k].clone_id)];
            Rwc[k] = c.R * R_ic;
            pwc[k] = c.R * t_ic + c.p;
        }
        // relative poses: camera k from anchor (0)
        M3 Rka[MAX_OBS]; V3 tka[MAX_OBS];
        for (int k = 0; k < n; k++) {
            Rka[k] = Rwc[k].t() * Rwc[0];
            tka[k] = Rwc[k].t() * (pwc[0] - pwc[k]);
        }
        double rho_prior = 1.0 / P_DEPTH_PRIOR, sig_prior = P_RHO_SIGMA;
        if (P_CARRY_DEPTH && f.has_pw) {
            const V3 Pa = Rwc[0].t() * (f.pw - pwc[0]);
            if (Pa.z > 0.1) {
                rho_prior = 1.0 / Pa.z;
                sig_prior = std::max((double)P_RHO_SIGMA_MIN, std::min((double)P_RHO_SIGMA, f.sig_rho * P_CARRY_INFLATE));
            }
        }
        double al = f.obs[0].u, be = f.obs[0].v, rho = rho_prior;
        const double sn = P_SIGMA_PX / (fx * P_DS);
        const double wprior = P_RHO_SIGMA > 0 ? std::pow(sn / sig_prior, 2) : 0.0;
        double h22inv = 0;
        double lambda = 1e-3;
        double cost_prev = 1e30;
        for (int it = 0; it < 8; it++) {
            double Hm[3][3] = {}, bv[3] = {}, cost = 0;
            for (int k = 0; k < n; k++) {
                const V3 a{al, be, 1};
                const V3 h = Rka[k] * a + tka[k] * rho;
                if (h.z < 1e-3) return false;
                const double iz = 1.0 / h.z;
                const double ru = h.x * iz - f.obs[k].u, rv = h.y * iz - f.obs[k].v;
                cost += ru * ru + rv * rv;
                // dh/d(al, be, rho)
                const V3 d0{Rka[k].m[0][0], Rka[k].m[1][0], Rka[k].m[2][0]};
                const V3 d1{Rka[k].m[0][1], Rka[k].m[1][1], Rka[k].m[2][1]};
                const V3 d2 = tka[k];
                const V3 d[3] = {d0, d1, d2};
                double J[2][3];
                for (int j = 0; j < 3; j++) {
                    J[0][j] = iz * (d[j].x - h.x * iz * d[j].z);
                    J[1][j] = iz * (d[j].y - h.y * iz * d[j].z);
                }
                for (int a1 = 0; a1 < 3; a1++) {
                    bv[a1] += J[0][a1] * ru + J[1][a1] * rv;
                    for (int a2 = 0; a2 < 3; a2++) Hm[a1][a2] += J[0][a1] * J[0][a2] + J[1][a1] * J[1][a2];
                }
            }
            Hm[2][2] += wprior; bv[2] += wprior * (rho - rho_prior); cost += wprior * (rho - rho_prior) * (rho - rho_prior);
            if (cost > cost_prev) lambda *= 10; else lambda *= 0.3;
            cost_prev = std::min(cost, cost_prev);
            {   // undamped inverse-depth variance (for the carried prior)
                const double a = Hm[0][0], b = Hm[0][1], c = Hm[1][1];
                const double d2 = a * c - b * b;
                const double schur = Hm[2][2] - (d2 > 1e-30 ? (c * Hm[0][2] * Hm[0][2] - 2 * b * Hm[0][2] * Hm[1][2] + a * Hm[1][2] * Hm[1][2]) / d2 : 0);
                h22inv = schur > 1e-30 ? 1.0 / schur : 1e30;
            }
            for (int a1 = 0; a1 < 3; a1++) Hm[a1][a1] *= (1 + lambda);
            // solve 3x3
            const double det = Hm[0][0] * (Hm[1][1] * Hm[2][2] - Hm[1][2] * Hm[2][1]) - Hm[0][1] * (Hm[1][0] * Hm[2][2] - Hm[1][2] * Hm[2][0]) +
                               Hm[0][2] * (Hm[1][0] * Hm[2][1] - Hm[1][1] * Hm[2][0]);
            if (std::fabs(det) < 1e-18) return false;
            double inv[3][3];
            inv[0][0] = (Hm[1][1] * Hm[2][2] - Hm[1][2] * Hm[2][1]) / det;
            inv[0][1] = (Hm[0][2] * Hm[2][1] - Hm[0][1] * Hm[2][2]) / det;
            inv[0][2] = (Hm[0][1] * Hm[1][2] - Hm[0][2] * Hm[1][1]) / det;
            inv[1][0] = (Hm[1][2] * Hm[2][0] - Hm[1][0] * Hm[2][2]) / det;
            inv[1][1] = (Hm[0][0] * Hm[2][2] - Hm[0][2] * Hm[2][0]) / det;
            inv[1][2] = (Hm[0][2] * Hm[1][0] - Hm[0][0] * Hm[1][2]) / det;
            inv[2][0] = (Hm[1][0] * Hm[2][1] - Hm[1][1] * Hm[2][0]) / det;
            inv[2][1] = (Hm[0][1] * Hm[2][0] - Hm[0][0] * Hm[2][1]) / det;
            inv[2][2] = (Hm[0][0] * Hm[1][1] - Hm[0][1] * Hm[1][0]) / det;
            double dx[3];
            for (int a1 = 0; a1 < 3; a1++) dx[a1] = -(inv[a1][0] * bv[0] + inv[a1][1] * bv[1] + inv[a1][2] * bv[2]);
            al += dx[0]; be += dx[1]; rho += dx[2];
            if (dx[0] * dx[0] + dx[1] * dx[1] < 1e-12 && std::fabs(dx[2]) < 1e-6) break;
        }
        if (!(rho > 1.0 / 200.0 && rho < 1.0 / 0.1)) return false;
        // reprojection gate (pixels at 320 x 240)
        double cost = 0;
        for (int k = 0; k < n; k++) {
            const V3 h = Rka[k] * V3{al, be, 1} + tka[k] * rho;
            if (h.z < 1e-3) return false;
            const double ru = (h.x / h.z - f.obs[k].u) * fx * P_DS, rv = (h.y / h.z - f.obs[k].v) * fy * P_DS;
            cost += ru * ru + rv * rv;
        }
        if (std::sqrt(cost / n) > P_MAX_REPROJ_PX) return false;
        Pw = Rwc[0] * (V3{al, be, 1} * (1.0 / rho)) + pwc[0];
        if (abr) { abr[0] = al; abr[1] = be; abr[2] = rho; abr[3] = sn * std::sqrt(h22inv); abr[4] = sig_prior; *Ranchor = Rwc[0]; }
        return true;
    }

    // rows of the stacked, null-space projected update
    double Hs[MAX_ROWS][NMAX];
    double rs[MAX_ROWS];

    // builds a feature's projected rows into Hs/rs starting at `row`; returns the number of rows (0 if rejected)
    int feature_rows(Feature& f, int row) {
        V3 Pw; double abr[5]; M3 Ra;
        st_cand++;
        if (!triangulate(f, Pw, abr, &Ra)) { st_tri_fail++; return 0; }
        // dPw/d(al, be, rho) for the anchored inverse-depth parameters (anchor pose held at its estimate)
        const double id = 1.0 / abr[2];
        const V3 c0{Ra.m[0][0] * id, Ra.m[1][0] * id, Ra.m[2][0] * id};
        const V3 c1{Ra.m[0][1] * id, Ra.m[1][1] * id, Ra.m[2][1] * id};
        const V3 c2 = (Ra * V3{abr[0], abr[1], 1}) * (-id * id);
        const V3 Jf[3] = {c0, c1, c2};
        const int n = f.nobs, m = 2 * n;
        const int nd = dim();
        double Hf[2 * MAX_OBS][3];
        double Hx[2 * MAX_OBS][6 * MAX_OBS];  // compact: columns per observation's clone
        double r[2 * MAX_OBS];
        int slot[MAX_OBS];
        for (int k = 0; k < n; k++) {
            slot[k] = clone_slot(f.obs[k].clone_id);
            const Clone& c = clones[slot[k]];
            const M3 Rt = c.R.t();
            const V3 Pi = Rt * (Pw - c.p);
            const V3 Pc = R_ci * Pi + t_ci;
            if (Pc.z < 0.05) return 0;
            const double iz = 1.0 / Pc.z;
            r[2 * k] = f.obs[k].u - Pc.x * iz;
            r[2 * k + 1] = f.obs[k].v - Pc.y * iz;
            const double Jp[2][3] = {{iz, 0, -Pc.x * iz * iz}, {0, iz, -Pc.y * iz * iz}};
            const M3 A = R_ci * Rt;                       // dPc/dPw
            const M3 Bth = A * skew(Pw - c.p);            // dPc/dtheta_i
            for (int a = 0; a < 2; a++) {
                for (int j = 0; j < 3; j++) {
                    double hf = 0, hth = 0, hp = 0;
                    for (int q = 0; q < 3; q++) { hf += Jp[a][q] * A.m[q][j]; hth += Jp[a][q] * Bth.m[q][j]; hp -= Jp[a][q] * A.m[q][j]; }
                    Hf[2 * k + a][j] = hf;
                    for (int kk = 0; kk < n; kk++) { Hx[2 * k + a][6 * kk + j] = 0; Hx[2 * k + a][6 * kk + 3 + j] = 0; }
                    Hx[2 * k + a][6 * k + j] = hth;
                    Hx[2 * k + a][6 * k + 3 + j] = hp;
                }
            }
        }
        // point Jacobian in (al, be, rho)
        for (int i = 0; i < m; i++) {
            const double h0 = Hf[i][0], h1 = Hf[i][1], h2 = Hf[i][2];
            for (int j = 0; j < 3; j++) Hf[i][j] = h0 * Jf[j].x + h1 * Jf[j].y + h2 * Jf[j].z;
        }
        // left null space of the (al, be) columns (all three without a depth prior) by Householder reflections on [Hf | Hx | r]
        const int nc = 6 * n;
        const int npro = P_RHO_SIGMA > 0 ? 2 : 3;
        for (int j = 0; j < npro; j++) {
            double s = 0;
            for (int i = j; i < m; i++) s += Hf[i][j] * Hf[i][j];
            const double nrm = std::sqrt(s);
            if (nrm < 1e-12) return 0;
            const double alpha = Hf[j][j] > 0 ? -nrm : nrm;
            double u[2 * MAX_OBS];
            for (int i = 0; i < m; i++) u[i] = i < j ? 0 : Hf[i][j];
            u[j] -= alpha;
            double un = 0;
            for (int i = j; i < m; i++) un += u[i] * u[i];
            if (un < 1e-24) continue;
            const double beta = 2.0 / un;
            for (int c = j; c < 3; c++) {
                double d = 0; for (int i = j; i < m; i++) d += u[i] * Hf[i][c];
                d *= beta; for (int i = j; i < m; i++) Hf[i][c] -= d * u[i];
            }
            for (int c = 0; c < nc; c++) {
                double d = 0; for (int i = j; i < m; i++) d += u[i] * Hx[i][c];
                d *= beta; for (int i = j; i < m; i++) Hx[i][c] -= d * u[i];
            }
            double d = 0; for (int i = j; i < m; i++) d += u[i] * r[i];
            d *= beta; for (int i = j; i < m; i++) r[i] -= d * u[i];
        }
        const int mr = m - npro;
        const double sig = P_SIGMA_PX / (fx * P_DS);
        // move the kept rows up to start at 0
        for (int i = 0; i < mr; i++) {
            for (int c = 0; c < nc; c++) Hx[i][c] = Hx[npro + i][c];
            Hf[i][2] = Hf[npro + i][2];
            r[i] = r[npro + i];
        }
        if (npro == 2) {
            // the depth prior's uncertainty along the projected rho column u: R = sig^2 (I + c u u^T); whiten with
            // W = I + (1/sqrt(1 + c |u|^2) - 1) u u^T / |u|^2
            const double cpr = abr[4] * abr[4] / (sig * sig);
            double uu = 0;
            for (int i = 0; i < mr; i++) uu += Hf[i][2] * Hf[i][2];
            if (uu > 1e-20) {
                const double kw = (1.0 / std::sqrt(1 + cpr * uu) - 1) / uu;
                for (int c = 0; c < nc; c++) {
                    double d = 0; for (int i = 0; i < mr; i++) d += Hf[i][2] * Hx[i][c];
                    d *= kw; for (int i = 0; i < mr; i++) Hx[i][c] += d * Hf[i][2];
                }
                double d = 0; for (int i = 0; i < mr; i++) d += Hf[i][2] * r[i];
                d *= kw; for (int i = 0; i < mr; i++) r[i] += d * Hf[i][2];
            }
        }
        if (P_CHI2) {
            // S = Hx Pcc Hx^T + sig^2 I over the involved clones; chi-square test on r
            double HP[2 * MAX_OBS][6 * MAX_OBS];
            for (int i = 0; i < mr; i++)
                for (int c = 0; c < nc; c++) {
                    double s = 0;
                    for (int k = 0; k < nc; k++) {
                        const int gk = NI + 6 * slot[k / 6] + k % 6, gc = NI + 6 * slot[c / 6] + c % 6;
                        s += Hx[i][k] * P[gk][gc];
                    }
                    HP[i][c] = s;
                }
            double S[2 * MAX_OBS][2 * MAX_OBS];
            for (int i = 0; i < mr; i++)
                for (int j = 0; j < mr; j++) {
                    double s = 0;
                    for (int c = 0; c < nc; c++) s += HP[i][c] * Hx[j][c];
                    S[i][j] = s + (i == j ? sig * sig : 0);
                }
            // Cholesky solve
            double L[2 * MAX_OBS][2 * MAX_OBS] = {};
            for (int i = 0; i < mr; i++)
                for (int j = 0; j <= i; j++) {
                    double s = S[i][j];
                    for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
                    if (i == j) { if (s <= 0) return 0; L[i][i] = std::sqrt(s); }
                    else L[i][j] = s / L[j][j];
                }
            double y[2 * MAX_OBS];
            double chi = 0;
            for (int i = 0; i < mr; i++) {
                double s = r[i];
                for (int k = 0; k < i; k++) s -= L[i][k] * y[k];
                y[i] = s / L[i][i];
                chi += y[i] * y[i];
            }
            static const double chi2_95[] = {0, 3.84, 5.99, 7.81, 9.49, 11.07, 12.59, 14.07, 15.51, 16.92, 18.31, 19.68, 21.03, 22.36, 23.68, 25.0, 26.3, 27.59};
            if (chi > P_CHI2_SCALE * chi2_95[std::min(mr, 17)]) { st_chi_fail++; return 0; }
        }
        f.has_pw = true; f.pw = Pw; f.sig_rho = abr[3];
        for (int i = 0; i < mr; i++) {
            for (int c = 0; c < nd; c++) Hs[row + i][c] = 0;
            for (int c = 0; c < nc; c++) Hs[row + i][NI + 6 * slot[c / 6] + c % 6] += Hx[i][c];
            rs[row + i] = r[i];
        }
        return mr;
    }

    void apply_correction(const double* dx) {
        R = expso3(V3{dx[0], dx[1], dx[2]}) * R;
        orthonormalize(R);
        p = p + V3{dx[3], dx[4], dx[5]};
        v = v + V3{dx[6], dx[7], dx[8]};
        bg = bg + V3{dx[9], dx[10], dx[11]};
        ba = ba + V3{dx[12], dx[13], dx[14]};
        g = g + V3{dx[15], dx[16], dx[17]};
        g = g * (GRAVITY / norm(g));
        for (int k = 0; k < nclones; k++) {
            const double* d = dx + NI + 6 * k;
            clones[k].R = expso3(V3{d[0], d[1], d[2]}) * clones[k].R;
            orthonormalize(clones[k].R);
            clones[k].p = clones[k].p + V3{d[3], d[4], d[5]};
        }
        const double sp = norm(v);
        if (sp > P_MAX_SPEED) v = v * (P_MAX_SPEED / sp);
    }

    void ekf_update(int m) {
        const int n = dim();
        // compress when there are more rows than states (Householder QR of [H | r])
        if (m > n) {
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int i = j; i < m; i++) s += Hs[i][j] * Hs[i][j];
                const double nrm = std::sqrt(s);
                if (nrm < 1e-14) continue;
                const double alpha = Hs[j][j] > 0 ? -nrm : nrm;
                static double u[MAX_ROWS];
                for (int i = j; i < m; i++) u[i] = Hs[i][j];
                u[j] -= alpha;
                double un = 0;
                for (int i = j; i < m; i++) un += u[i] * u[i];
                if (un < 1e-28) continue;
                const double beta = 2.0 / un;
                for (int c = j; c < n; c++) {
                    double d = 0; for (int i = j; i < m; i++) d += u[i] * Hs[i][c];
                    d *= beta; for (int i = j; i < m; i++) Hs[i][c] -= d * u[i];
                }
                double d = 0; for (int i = j; i < m; i++) d += u[i] * rs[i];
                d *= beta; for (int i = j; i < m; i++) rs[i] -= d * u[i];
            }
            m = n;
        }
        const double sig = P_SIGMA_PX / (fx * P_DS);
        static double PHt[NMAX][MAX_ROWS > NMAX ? MAX_ROWS : NMAX];
        static double S[NMAX][NMAX];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) s += P[i][k] * Hs[j][k];
                PHt[i][j] = s;
            }
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= i; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) s += Hs[i][k] * PHt[k][j];
                S[i][j] = S[j][i] = s + (i == j ? sig * sig : 0);
            }
        // Cholesky S = L L^T (in place, lower)
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= i; j++) {
                double s = S[i][j];
                for (int k = 0; k < j; k++) s -= S[i][k] * S[j][k];
                if (i == j) { if (s <= 1e-30) return; S[i][i] = std::sqrt(s); }
                else S[i][j] = s / S[j][j];
            }
        // K^T = S^{-1} (P H^T)^T : solve for each state i the m-vector PHt[i][:]
        static double KT[NMAX][MAX_ROWS > NMAX ? MAX_ROWS : NMAX];  // KT[i][:] = row i of K
        for (int i = 0; i < n; i++) {
            double y[MAX_ROWS > NMAX ? MAX_ROWS : NMAX];
            for (int a = 0; a < m; a++) {
                double s = PHt[i][a];
                for (int k = 0; k < a; k++) s -= S[a][k] * y[k];
                y[a] = s / S[a][a];
            }
            for (int a = m - 1; a >= 0; a--) {
                double s = y[a];
                for (int k = a + 1; k < m; k++) s -= S[k][a] * KT[i][k];
                KT[i][a] = s / S[a][a];
            }
        }
        double dx[NMAX];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int a = 0; a < m; a++) s += KT[i][a] * rs[a];
            dx[i] = s;
        }
        // P = P - K (P H^T)^T
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++) {
                double s = 0;
                for (int a = 0; a < m; a++) s += KT[i][a] * PHt[j][a];
                const double val = P[i][j] - s;
                P[i][j] = val; P[j][i] = val;
            }
        for (int i = 0; i < n; i++) if (P[i][i] < 1e-12) P[i][i] = 1e-12;
        apply_correction(dx);
    }

    // features with >= P_MIN_OBS observations in `ids` (feature indices) are used in one update
    void update_with(const int* idx, int count) {
        int rows = 0, used = 0;
        double depth_sum = 0; int depth_n = 0;
        for (int q = 0; q < count && used < P_MAX_UPDATE_FEAT; q++) {
            Feature& f = feats[idx[q]];
            if (f.nobs < P_MIN_OBS) continue;
            const int got = feature_rows(f, rows);
            if (got > 0) { rows += got; used++; st_used++; st_rows += got; (void)depth_sum; (void)depth_n; }
        }
        if (rows > 0) ekf_update(rows);
    }

    // ---------------------------------------------------------------- front end
    void pixel_to_norm(float x, float y, double& u, double& v) const { u = (x - cx) / fx; v = (y - cy) / fy; }

    void detect(const Image& im) {
        if (nfeats >= P_DETECT_BELOW) return;
        constexpr int CW = GW / P_GRID_X, CH = GH / P_GRID_Y;
        int occupied[P_GRID_Y][P_GRID_X] = {};
        for (int k = 0; k < nfeats; k++) {
            const int gx = std::min(P_GRID_X - 1, (int)(feats[k].x / CW)), gy = std::min(P_GRID_Y - 1, (int)(feats[k].y / CH));
            occupied[gy][gx]++;
        }
        const int border = P_HALF + 3;
        for (int gy = 0; gy < P_GRID_Y && nfeats < P_MAX_FEAT; gy++)
            for (int gx = 0; gx < P_GRID_X && nfeats < P_MAX_FEAT; gx++) {
                if (occupied[gy][gx]) continue;
                float best = P_MIN_SCORE; int bx = -1, by = -1;
                const int x0 = std::max(border, gx * CW), x1 = std::min(GW - border, (gx + 1) * CW);
                const int y0 = std::max(border, gy * CH), y1 = std::min(GH - border, (gy + 1) * CH);
                for (int y = y0; y < y1; y += P_DETECT_STEP)
                    for (int x = x0; x < x1; x += P_DETECT_STEP) {
                        const float s = corner_score(im, x, y);
                        if (s > best) { best = s; bx = x; by = y; }
                    }
                if (bx < 0) continue;
                Feature& f = feats[nfeats++];
                f.id = next_feat_id++; f.x = (float)bx; f.y = (float)by; f.nobs = 0; f.alive = true; f.has_pw = false;
            }
    }

    // ---------------------------------------------------------------- per frame
    void step(double t, const uint8_t* rgb, const double* imu, int count, double* out) {
        Image& im = img[cur ^ 1];
        make_gray(rgb, im);
        if (frame == 0) {
            init_state(imu, count);
            t_last = t;
            add_clone();
            detect(im);
            record_obs(im);
        } else {
            const M3 R_prev = R; const V3 p_prev = p;
            propagate(imu, count);
            track(img[cur], im, R_prev, p_prev);
            const bool clone_frame = frame % P_CLONE_EVERY == 0;
            if (clone_frame) {
                if (nclones == P_CLONES) {
                    // features seen in the oldest clone: use them now, then drop the clone
                    const int oldest = clones[0].id;
                    int idx[P_MAX_FEAT]; int ni = 0;
                    for (int k = 0; k < nfeats; k++) if (feats[k].nobs > 0 && feats[k].obs[0].clone_id == oldest) idx[ni++] = k;
                    update_with(idx, ni);
                    for (int k = 0; k < ni; k++) {
                        Feature& f = feats[idx[k]];
                        if (f.nobs >= P_MIN_OBS) f.nobs = 0;  // used: start a fresh track from here
                        else { for (int q = 1; q < f.nobs; q++) f.obs[q - 1] = f.obs[q]; f.nobs--; }
                    }
                    remove_clone(0);
                }
                add_clone();
                record_obs(im);
            }
            // features lost this frame
            int lost[P_MAX_FEAT]; int nl = 0;
            for (int k = 0; k < nfeats; k++) if (!feats[k].alive) lost[nl++] = k;
            if (nl) update_with(lost, nl);
            int w = 0;
            for (int k = 0; k < nfeats; k++) if (feats[k].alive) feats[w++] = feats[k];
            nfeats = w;
            detect(im);
            if (clone_frame) record_obs_new(im);
        }
        cur ^= 1;
        frame++;
        write_output(out);
    }

    void record_obs(const Image&) {
        for (int k = 0; k < nfeats; k++) if (feats[k].alive) push_obs(feats[k]);
    }
    void record_obs_new(const Image&) {  // newly detected features get their first observation at this clone
        const int id = clones[nclones - 1].id;
        for (int k = 0; k < nfeats; k++)
            if (feats[k].nobs == 0 || feats[k].obs[feats[k].nobs - 1].clone_id != id) push_obs(feats[k]);
    }
    void push_obs(Feature& f) {
        const int id = clones[nclones - 1].id;
        if (f.nobs == MAX_OBS) { for (int q = 1; q < f.nobs; q++) f.obs[q - 1] = f.obs[q]; f.nobs--; }
        double u, vv; pixel_to_norm(f.x, f.y, u, vv);
        f.obs[f.nobs++] = {id, u, vv};
    }

    void track(const Image& a, const Image& b, const M3& R_prev, const V3& p_prev) {
        // predicted camera motion: camera at prev -> camera now
        const M3 Rwc0 = R_prev * R_ic, Rwc1 = R * R_ic;
        const V3 pwc0 = R_prev * t_ic + p_prev, pwc1 = R * t_ic + p;
        const M3 R10 = Rwc1.t() * Rwc0;
        const V3 t10 = Rwc1.t() * (pwc0 - pwc1);
        for (int k = 0; k < nfeats; k++) {
            Feature& f = feats[k];
            double u, vv; pixel_to_norm(f.x, f.y, u, vv);
            const V3 X = V3{u, vv, 1} * depth_est;
            const V3 Y = R10 * X + t10;
            float px = f.x, py = f.y;
            if (Y.z > 0.05) { px = (float)(fx * Y.x / Y.z + cx); py = (float)(fy * Y.y / Y.z + cy); }
            float x = px, y = py;
            bool ok = track_lk(a, b, f.x, f.y, x, y);
            if (!ok && (px != f.x || py != f.y)) { x = f.x; y = f.y; ok = track_lk(a, b, f.x, f.y, x, y); }
            if (ok && P_FB_CHECK) {
                float bx = f.x, by = f.y;
                if (!track_lk(b, a, x, y, bx, by) || (bx - f.x) * (bx - f.x) + (by - f.y) * (by - f.y) > 0.5f) ok = false;
            }
            if (ok) { f.x = x; f.y = y; st_tracked++; } else { f.alive = false; st_lost++; }
        }
    }

    void init_state(const double* imu, int count) {
        V3 fm{0, 0, 0};
        for (int i = 0; i < count; i++) fm = fm + V3{imu[7 * i + 1], imu[7 * i + 2], imu[7 * i + 3]};
        fm = fm * (1.0 / std::max(count, 1));
        const double n = norm(fm);
        g = n > 1e-6 ? fm * (-GRAVITY / n) : V3{0, 0, -GRAVITY};
        std::memset(P, 0, sizeof(P));
        const double sg = GRAVITY * P_INIT_TILT;
        const V3 gh = g * (1.0 / GRAVITY);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) P[15 + i][15 + j] = sg * sg * ((i == j) - gh[i] * gh[j]) + 0.01 * gh[i] * gh[j];
            P[6 + i][6 + i] = P_INIT_VEL * P_INIT_VEL;
            P[9 + i][9 + i] = P_INIT_BG * P_INIT_BG;
            P[12 + i][12 + i] = P_INIT_BA * P_INIT_BA;
        }
    }

    void write_output(double* out) const {
        // T_c0_ck = T_ci * T_w_imu_k * T_ic   (world = IMU frame at frame 0)
        const M3 Rc = R_ci * R * R_ic;
        const V3 pc = R_ci * (R * t_ic + p) + t_ci;
        double q[4]; quaternion(Rc, q);
        out[0] = pc.x; out[1] = pc.y; out[2] = pc.z;
        out[3] = q[0]; out[4] = q[1]; out[5] = q[2]; out[6] = q[3];
        // camera centre velocity in world, then in camera-k axes
        const V3 vc_w = v + R * cross(w_last, t_ic);
        const M3 Rwc = R * R_ic;
        const V3 vc = Rwc.t() * vc_w;
        const V3 wc = R_ci * w_last;
        out[7] = vc.x; out[8] = vc.y; out[9] = vc.z;
        out[10] = wc.x; out[11] = wc.y; out[12] = wc.z;
    }
};

Estimator* estimator = nullptr;

// minimal JSON number lookup: the value after "key":
bool json_number(const char* json, uint32_t length, const char* key, double& value) {
    char pattern[32];
    std::snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* end = json + length;
    const size_t pl = std::strlen(pattern);
    for (const char* s = json; s + pl < end; s++) {
        if (std::memcmp(s, pattern, pl) == 0) {
            const char* c = s + pl;
            while (c < end && (*c == ' ' || *c == ':')) c++;
            value = std::strtod(c, nullptr);
            return true;
        }
    }
    return false;
}

bool json_matrix4(const char* json, uint32_t length, const char* key, double M[16]) {
    char pattern[32];
    std::snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* end = json + length;
    const size_t pl = std::strlen(pattern);
    for (const char* s = json; s + pl < end; s++) {
        if (std::memcmp(s, pattern, pl) == 0) {
            const char* c = s + pl;
            int k = 0;
            while (c < end && k < 16) {
                if ((*c >= '0' && *c <= '9') || *c == '-' || *c == '+' || *c == '.') {
                    char* e;
                    M[k++] = std::strtod(c, &e);
                    c = e;
                } else c++;
            }
            return k == 16;
        }
    }
    return false;
}

}  // namespace

extern "C" {

VIOBENCH_EXPORT(abi_version) int32_t abi_version() { return VIOBENCH_ABI_VERSION; }

VIOBENCH_EXPORT(alloc) void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }

VIOBENCH_EXPORT(frame_width) int32_t frame_width() { return 320; }

VIOBENCH_EXPORT(init) int32_t init(const char* json, uint32_t length) {
    double fx, fy, cx, cy, T[16];
    if (!json_number(json, length, "fx", fx) || !json_number(json, length, "fy", fy) || !json_number(json, length, "cx", cx) ||
        !json_number(json, length, "cy", cy) || !json_matrix4(json, length, "T_cam_imu", T)) {
        std::fprintf(stderr, "init: bad config\n");
        return VIOBENCH_STATUS_ERROR;
    }
    Estimator& e = *new Estimator();  // one instance per episode: init runs once
    // intrinsics at the gray resolution (pixel-centre convention)
    e.fx = fx / P_DS; e.fy = fy / P_DS;
    e.cx = (cx + 0.5) / P_DS - 0.5; e.cy = (cy + 0.5) / P_DS - 0.5;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) e.R_ci.m[i][j] = T[4 * i + j];
    e.t_ci = {T[3], T[7], T[11]};
    e.R_ic = e.R_ci.t();
    e.t_ic = (e.R_ic * e.t_ci) * -1.0;
    estimator = &e;
    return VIOBENCH_STATUS_OK;
}

#ifndef __wasm__
// host diagnostics: [frame, nfeats, nclones, tracked, lost, cand, tri_fail, chi_fail, used, rows, v(3), bg(3), ba(3), g(3), sd_v(3), sd_tilt]
int32_t debug_state(double* o) {
    Estimator& e = *estimator;
    double vals[] = {(double)e.frame, (double)e.nfeats, (double)e.nclones, (double)e.st_tracked, (double)e.st_lost, (double)e.st_cand,
                     (double)e.st_tri_fail, (double)e.st_chi_fail, (double)e.st_used, (double)e.st_rows,
                     e.v.x, e.v.y, e.v.z, e.bg.x, e.bg.y, e.bg.z, e.ba.x, e.ba.y, e.ba.z, e.g.x, e.g.y, e.g.z,
                     std::sqrt(P[6][6]), std::sqrt(P[7][7]), std::sqrt(P[8][8]), std::sqrt(P[15][15])};
    for (unsigned i = 0; i < sizeof(vals) / sizeof(double); i++) o[i] = vals[i];
    return 0;
}
#endif

VIOBENCH_EXPORT(step) int32_t step(double t, const uint8_t* frame, const double* imu, int32_t imu_count, double* out) {
    if (!estimator) return VIOBENCH_STATUS_ERROR;
    estimator->step(t, frame, imu, imu_count, out);
    for (int i = 0; i < 13; i++) if (!std::isfinite(out[i])) {
        std::fill(out, out + 13, 0.0); out[6] = 1.0;
        return VIOBENCH_STATUS_OK;
    }
    return VIOBENCH_STATUS_OK;
}

}  // extern "C"
