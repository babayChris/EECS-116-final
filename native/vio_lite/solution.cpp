// vio_lite: low-fuel visual-inertial odometry.
// Rotation: gyro integration with a gyro bias estimate. Translation: a 9-state Kalman filter (world velocity,
// gravity vector, gyro bias) whose only visual input is the coplanarity of a few tracked features' bearings
// P_BASE frames apart:  (p_k - p_j) . (b_j x b_k) = 0, with p_k - p_j = v_k tau - M_jk - g tau^2 / 2 from
// the IMU (linear in v and g). Metric scale comes from the accelerometer (M_jk); no depths, no clones.
// World = IMU frame at frame 0. Front end shared in spirit with native/vio_ekf (green-channel sparse patches).
#include <viobench_abi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "params.h"

namespace {

constexpr int W0 = 320, H0 = 240;
constexpr int GW = W0 / P_DS, GH = H0 / P_DS;
constexpr double GRAVITY = 9.81;
constexpr int NS = P_ATT ? 12 : (P_BIAS ? 9 : 6);   // v(3) g(3) [bg(3)] [theta(3): world-frame attitude error]
static_assert(!P_ATT || P_BIAS, "P_ATT needs P_BIAS");
constexpr int HIST = P_BASE + 1;   // ring of bearings / IMU integrals

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
    V3 tmul(const V3& v) const {  // R^T v
        return {m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z, m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
                m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z};
    }
    M3 t() const { M3 r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.m[i][j] = m[j][i]; return r; }
    M3 operator*(double s) const { M3 r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.m[i][j] = m[i][j] * s; return r; }
};

M3 expso3(const V3& w) {  // second-order Taylor is enough for per-frame angles below ~0.2 rad; exact beyond
    const double th2 = dot(w, w);
    double a, b;
    if (th2 < 0.04) { a = 1 - th2 * (1.0 / 6); b = 0.5 - th2 * (1.0 / 24); }
    else { const double th = std::sqrt(th2); a = std::sin(th) / th; b = (1 - std::cos(th)) / th2; }
    const double x = w.x, y = w.y, z = w.z;
    M3 R;
    R.m[0][0] = 1 - b * (y * y + z * z); R.m[0][1] = -a * z + b * x * y; R.m[0][2] = a * y + b * x * z;
    R.m[1][0] = a * z + b * x * y; R.m[1][1] = 1 - b * (x * x + z * z); R.m[1][2] = -a * x + b * y * z;
    R.m[2][0] = -a * y + b * x * z; R.m[2][1] = a * x + b * y * z; R.m[2][2] = 1 - b * (x * x + y * y);
    return R;
}

void orthonormalize(M3& R) {
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
}

// ------------------------------------------------------------------ front end
constexpr int PH = P_HALF, PN = (2 * PH + 1) * (2 * PH + 1);
constexpr int WR = PH + 1 + P_MARGIN, WS = 2 * WR + 1;
constexpr int OFF = P_POINT ? P_DS / 2 : 0;

inline float green(const uint8_t* rgb, int x, int y) {
    if (P_DS == 1 || P_POINT) return rgb[((y * P_DS + OFF) * W0 + x * P_DS + OFF) * 3 + 1];
    const uint8_t* p = rgb + ((y * P_DS) * W0 + x * P_DS) * 3 + 1;
    int s = 0;
    for (int dy = 0; dy < P_DS; dy++, p += W0 * 3)
        for (int dx = 0; dx < P_DS; dx++) s += p[3 * dx];
    return s * (1.0f / (P_DS * P_DS));
}

inline bool extract_window(const uint8_t* rgb, int cx, int cy, float* win) {
    if (cx - WR < 0 || cy - WR < 0 || cx + WR >= GW || cy + WR >= GH) return false;
    for (int y = 0; y < WS; y++)
        for (int x = 0; x < WS; x++) win[y * WS + x] = green(rgb, cx - WR + x, cy - WR + y);
    return true;
}

struct Template {
    int ix, iy;
    float T[PN], gx[PN], gy[PN];
    float i00, i01, i11, contrast;
    bool ok;
};

inline void make_template(const float* win, int wcx, int wcy, int ix, int iy, Template& t) {
    t.ix = ix; t.iy = iy; t.ok = false;
    const int ox = ix - wcx + WR, oy = iy - wcy + WR;
    if (ox - PH - 1 < 0 || oy - PH - 1 < 0 || ox + PH + 1 >= WS || oy + PH + 1 >= WS) return;
    float gxx = 0, gxy = 0, gyy = 0, mean = 0;
    int k = 0;
    for (int dy = -PH; dy <= PH; dy++) {
        const float* row = win + (oy + dy) * WS + ox;
        for (int dx = -PH; dx <= PH; dx++, k++) {
            t.T[k] = row[dx];
            mean += row[dx];
            const float a = 0.5f * (row[dx + 1] - row[dx - 1]);
            const float b = 0.5f * (row[dx + WS] - row[dx - WS]);
            t.gx[k] = a; t.gy[k] = b;
            gxx += a * a; gxy += a * b; gyy += b * b;
        }
    }
    const float det = gxx * gyy - gxy * gxy;
    if (det < 1e-3f * PN * PN) return;
    mean /= PN;
    float dev = 0;
    for (int q = 0; q < PN; q++) dev += std::fabs(t.T[q] - mean);
    t.contrast = dev / PN;
    t.i00 = gyy / det; t.i01 = -gxy / det; t.i11 = gxx / det;
    t.ok = true;
}

inline bool track_template(const uint8_t* rgb, const Template& t, float& dx, float& dy, float* win, int& wcx, int& wcy) {
    wcx = (int)std::lround(t.ix + dx); wcy = (int)std::lround(t.iy + dy);
    if (!extract_window(rgb, wcx, wcy, win)) return false;
    float err = 0;
    for (int it = 0; it < P_LK_ITERS; it++) {
        const float px = t.ix + dx - (wcx - WR), py = t.iy + dy - (wcy - WR);
        const int bx = (int)std::floor(px), by = (int)std::floor(py);
        if (bx - PH < 0 || by - PH < 0 || bx + PH + 1 >= WS || by + PH + 1 >= WS) return false;
        const float fx = px - bx, fy = py - by;
        const float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
        float sx = 0, sy = 0;
        err = 0;
        int k = 0;
        for (int yy = -PH; yy <= PH; yy++) {
            const float* r0 = win + (by + yy) * WS + bx;
            const float* r1 = r0 + WS;
            for (int xx = -PH; xx <= PH; xx++, k++) {
                const float e = w00 * r0[xx] + w01 * r0[xx + 1] + w10 * r1[xx] + w11 * r1[xx + 1] - t.T[k];
                sx += t.gx[k] * e; sy += t.gy[k] * e;
                err += std::fabs(e);
            }
        }
        const float ux = t.i00 * sx + t.i01 * sy, uy = t.i01 * sx + t.i11 * sy;
        dx -= ux; dy -= uy;
        if (ux * ux + uy * uy < 0.0025f) break;
    }
    err /= PN;
    return err < P_LK_MAX_ERR && err < P_LK_REL_ERR * t.contrast + 1.0f;
}

constexpr int CW = GW / P_GRID_X, CH = GH / P_GRID_Y;
constexpr int CELL_MAX = (CW + 4) * (CH + 4);
bool best_corner(const uint8_t* rgb, int x0, int y0, int x1, int y1, int& bx, int& by) {
    static float buf[CELL_MAX], gxb[CELL_MAX], gyb[CELL_MAX];
    const int w = x1 - x0 + 4, h = y1 - y0 + 4;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) buf[y * w + x] = green(rgb, x0 - 2 + x, y0 - 2 + y);
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            gxb[y * w + x] = buf[y * w + x + 1] - buf[y * w + x - 1];
            gyb[y * w + x] = buf[(y + 1) * w + x] - buf[(y - 1) * w + x];
        }
    float best = P_MIN_SCORE * 4 * 9;
    bx = -1;
    for (int y = 2; y < h - 2; y += P_DETECT_STEP)
        for (int x = 2; x < w - 2; x += P_DETECT_STEP) {
            float sxx = 0, sxy = 0, syy = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    const float a = gxb[(y + dy) * w + x + dx], b = gyb[(y + dy) * w + x + dx];
                    sxx += a * a; sxy += a * b; syy += b * b;
                }
            const float tr = 0.5f * (sxx + syy), d = std::sqrt(0.25f * (sxx - syy) * (sxx - syy) + sxy * sxy);
            if (tr - d > best) { best = tr - d; bx = x0 - 2 + x; by = y0 - 2 + y; }
        }
    return bx >= 0;
}

// Shi-Tomasi score of one point from a 5 x 5 read (3 x 3 tensor of central differences)
inline float point_score(const uint8_t* rgb, int x, int y) {
    float b[25];
    for (int dy = 0; dy < 5; dy++) for (int dx = 0; dx < 5; dx++) b[dy * 5 + dx] = green(rgb, x - 2 + dx, y - 2 + dy);
    float sxx = 0, sxy = 0, syy = 0;
    for (int dy = 1; dy <= 3; dy++)
        for (int dx = 1; dx <= 3; dx++) {
            const float a = b[dy * 5 + dx + 1] - b[dy * 5 + dx - 1], c = b[dy * 5 + dx + 5] - b[dy * 5 + dx - 5];
            sxx += a * a; sxy += a * c; syy += c * c;
        }
    const float tr = 0.5f * (sxx + syy), d = std::sqrt(0.25f * (sxx - syy) * (sxx - syy) + sxy * sxy);
    return tr - d;
}

// sparse corner search: P_DET_CAND points on a jittered grid in the cell scored with a 5 x 5 Shi-Tomasi, then a
// hill climb on the 8 neighbours of the best (P_DET_REFINE steps)
inline float point_score5(const uint8_t* rgb, int x, int y) {
    float b[25];
    for (int dy = 0; dy < 5; dy++) for (int dx = 0; dx < 5; dx++) b[dy * 5 + dx] = green(rgb, x - 2 + dx, y - 2 + dy);
    float sxx = 0, sxy = 0, syy = 0;
    for (int dy = 1; dy <= 3; dy++)
        for (int dx = 1; dx <= 3; dx++) {
            const float a = b[dy * 5 + dx + 1] - b[dy * 5 + dx - 1], c = b[dy * 5 + dx + 5] - b[dy * 5 + dx - 5];
            sxx += a * a; sxy += a * c; syy += c * c;
        }
    const float tr = 0.5f * (sxx + syy), d = std::sqrt(0.25f * (sxx - syy) * (sxx - syy) + sxy * sxy);
    return tr - d;
}
inline bool sparse_corner(const uint8_t* rgb, int x0, int y0, int x1, int y1, unsigned salt, int& bx, int& by) {
    const int w = x1 - x0, h = y1 - y0;
    if (w < 3 || h < 3) return false;
    int nx = 1; while ((nx + 1) * (nx + 1) * h <= P_DET_CAND * w) nx++;   // grid nx x ny ~ P_DET_CAND, aspect of the cell
    int ny = std::max(1, P_DET_CAND / nx);
    const float sx = (float)w / nx, sy = (float)h / ny;
    const float jx = ((salt * 2654435761u) >> 16 & 255) / 256.0f, jy = ((salt * 40503u + 7u) >> 4 & 255) / 256.0f;
    float best = -1;
    bx = -1;
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            const int x = x0 + (int)((i + jx) * sx), y = y0 + (int)((j + jy) * sy);
            const float s = point_score5(rgb, x, y);
            if (s > best) { best = s; bx = x; by = y; }
        }
    for (int it = 0; it < P_DET_REFINE; it++) {
        int nbx = bx, nby = by;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (!dx && !dy) continue;
                const int x = bx + dx * P_DET_REFINE_STEP, y = by + dy * P_DET_REFINE_STEP;
                if (x < x0 || x >= x1 || y < y0 || y >= y1) continue;
                const float s = point_score5(rgb, x, y);
                if (s > best) { best = s; nbx = x; nby = y; }
            }
        if (nbx == bx && nby == by) break;
        bx = nbx; by = nby;
    }
    return best > P_MIN_SCORE * 4 * 9;
}

struct Feature {
    int id;
    float x, y;            // gray pixels, current frame
    int n;                 // frames tracked (history entries valid = min(n, HIST))
    int stale;             // frames since the last tracking (P_TRACK_EVERY)
    V3 bw[HIST];           // world bearing (unit) at the last HIST frames, ring indexed by frame % HIST
    Template tpl;
};

struct FrameRec { double t; V3 A1, A2, lev; };  // cumulative IMU integrals and lever arm R t_ic at a frame

#ifndef __wasm__
static int truth_on = 0; static double truth_R[9], truth_v[3], truth_g[3];
static double nis_sum = 0; static int nis_n = 0; static double nis_hist[8];
#endif
struct Estimator {
    double fx, fy, cx, cy;
    M3 R_ci, R_ic; V3 t_ci, t_ic;
    M3 R = M3::eye(); V3 p{0, 0, 0}, v{0, 0, 0}, g{0, 0, -GRAVITY}, bg{0, 0, 0};
    double P[NS][NS];
    V3 A1{0, 0, 0}, A2{0, 0, 0};
    M3 SR_ = M3::eye(); V3 Sa_{0, 0, 0};
    M3 C = M3::eye();   // attitude corrections since the start: history is stored in C^T (world) axes
    double t0 = 0, t_last = 0, t_prev = 0;
    V3 w_last{0, 0, 0};
    FrameRec rec[HIST];
    Feature feats[P_MAX_FEAT]; int nfeats = 0, next_id = 0;
    int frame = 0, detect_cell = 0;
    uint32_t rng = 12345;
    int st_used = 0, st_rej = 0;
    float win[WS * WS];

    void reset_cov() {
        std::memset(P, 0, sizeof(P));
        const double sg = GRAVITY * P_INIT_TILT;
        const V3 gh = g * (1.0 / norm(g));
        for (int i = 0; i < 3; i++) {
            P[i][i] = P_INIT_VEL * P_INIT_VEL;
            for (int j = 0; j < 3; j++) P[3 + i][3 + j] = sg * sg * ((i == j) - gh[i] * gh[j]) + 0.01 * gh[i] * gh[j];
            if (P_BIAS) P[6 + i][6 + i] = P_INIT_BG * P_INIT_BG;
        }
    }

    // scalar Kalman update: innovation y, row h (NS), variance r; returns false if gated
    bool kf_update(const double* h, double y, double r, double gate, int nupd = NS) {  // states >= nupd are not corrected (Schmidt)
        double ph[NS];
        for (int i = 0; i < NS; i++) { double s = 0; for (int j = 0; j < NS; j++) s += P[i][j] * h[j]; ph[i] = s; }
        double s = r;
        for (int i = 0; i < NS; i++) s += h[i] * ph[i];
#ifndef __wasm__
        if (gate > 0) { const double z = y * y / ((truth_on & 2) ? r : s); nis_sum += z; nis_n++; nis_hist[z < 0.5 ? 0 : z < 1 ? 1 : z < 2 ? 2 : z < 4 ? 3 : z < 9 ? 4 : z < 25 ? 5 : z < 100 ? 6 : 7]++; }
        if (truth_on & 2) return true;
#endif
        if (gate > 0 && y * y > gate * s) return false;
        const double inv = 1.0 / s;
        double dx[NS];
        for (int i = 0; i < NS; i++) dx[i] = i < nupd ? ph[i] * inv * y : 0.0;
        for (int i = 0; i < nupd; i++) {
            const double ki = ph[i] * inv;
            for (int j = i; j < NS; j++) { P[i][j] -= ki * ph[j]; P[j][i] = P[i][j]; }
        }
        v = v + V3{dx[0], dx[1], dx[2]};
        g = g + V3{dx[3], dx[4], dx[5]};
        if (P_BIAS) bg = bg + V3{dx[6], dx[7], dx[8]};
        if (P_ATT) { const M3 E = expso3(V3{dx[9], dx[10], dx[11]}); R = E * R; if (P_HISTCORR) C = E * C; }
        return true;
    }

    void propagate(const double* imu, int count) {
        // interval sums (P_FRAME_IMU) or per sample
        double T = 0;
        if (P_FRAME_IMU) {
            V3 ws{0, 0, 0}, fs{0, 0, 0};
            double tm = 0;
            for (int i = 0; i < count; i++) {
                const double* r = imu + 7 * i;
                const double dt = r[0] - t_last;
                t_last = r[0];
                if (dt <= 0 || dt > 0.1) continue;
                ws = ws + V3{r[4], r[5], r[6]} * dt;
                fs = fs + V3{r[1], r[2], r[3]} * dt;
                tm += (r[0] - 0.5 * dt) * dt;
                T += dt;
            }
            if (T <= 0) return;
            ws = ws - bg * T;
            w_last = ws * (1.0 / T);
            const M3 Rm = R * expso3(ws * 0.5);
            const V3 dv = Rm * fs;
            SR_ = Rm * T; Sa_ = dv;
            const V3 dvr = C.tmul(dv);
            A1 = A1 + dvr;
            A2 = A2 + dvr * (tm / T - t0);
            p = p + v * T + dv * (0.5 * T) + g * (0.5 * T * T);
            v = v + dv + g * T;
            R = R * expso3(ws);
        } else {
            for (int i = 0; i < count; i++) {
                const double* r = imu + 7 * i;
                const double dt = r[0] - t_last;
                t_last = r[0];
                if (dt <= 0 || dt > 0.1) continue;
                const V3 w = V3{r[4], r[5], r[6]} - bg;
                w_last = w;
                const V3 a = R * V3{r[1], r[2], r[3]};
                A1 = A1 + a * dt;
                A2 = A2 + a * ((r[0] - 0.5 * dt - t0) * dt);
                p = p + v * dt + (a + g) * (0.5 * dt * dt);
                v = v + (a + g) * dt;
                R = R * expso3(w * dt);
                T += dt;
            }
            if (T <= 0) return;
        }
        orthonormalize(R);
        if (P_ATT) {
            // F = I + blocks: (v,th) = -[Sa]x, (v,g) = T, (v,bg) = 0.5 [Sa]x SR, (th,bg) = -SR
            M3 Bvt, Bvb, Btb;
            const V3& a = Sa_;
            const M3 Sk = {{{0, -a.z, a.y}, {a.z, 0, -a.x}, {-a.y, a.x, 0}}};
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { Bvt.m[i][j] = -Sk.m[i][j]; Btb.m[i][j] = -SR_.m[i][j]; }
            Bvb = (Sk * SR_) * 0.5;
            double tmp[6][NS];
            for (int j = 0; j < NS; j++)
                for (int i = 0; i < 3; i++) {
                    double sv = P[i][j] + T * P[3 + i][j], st = P[9 + i][j];
                    for (int k = 0; k < 3; k++) { sv += Bvt.m[i][k] * P[9 + k][j] + Bvb.m[i][k] * P[6 + k][j]; st += Btb.m[i][k] * P[6 + k][j]; }
                    tmp[i][j] = sv; tmp[3 + i][j] = st;
                }
            for (int i = 0; i < 3; i++) for (int j = 0; j < NS; j++) { P[i][j] = tmp[i][j]; P[9 + i][j] = tmp[3 + i][j]; }
            for (int i = 0; i < NS; i++) {
                double nv[3], nt[3];
                for (int c = 0; c < 3; c++) {
                    double sv = P[i][c] + T * P[i][3 + c], st = P[i][9 + c];
                    for (int k = 0; k < 3; k++) { sv += P[i][9 + k] * Bvt.m[c][k] + P[i][6 + k] * Bvb.m[c][k]; st += P[i][6 + k] * Btb.m[c][k]; }
                    nv[c] = sv; nt[c] = st;
                }
                for (int c = 0; c < 3; c++) { P[i][c] = nv[c]; P[i][9 + c] = nt[c]; }
            }
            const double qt = P_GYRO_NOISE * P_GYRO_NOISE * T;
            for (int i = 0; i < 3; i++) P[9 + i][9 + i] += qt;
        } else {
        // covariance: v += g T (F = I + T E_vg), noise on v, g, bg
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < NS; j++) P[i][j] += T * P[3 + i][j];
        for (int i = 0; i < NS; i++)
            for (int j = 0; j < 3; j++) P[i][j] += T * P[i][3 + j];
        }
        const double qa = P_ACC_NOISE * P_ACC_NOISE * T, qg = P_G_WALK * P_G_WALK * T, qb = P_GYRO_WALK * P_GYRO_WALK * T;
        for (int i = 0; i < 3; i++) {
            P[i][i] += qa; P[3 + i][3 + i] += qg;
            if (P_BIAS) P[6 + i][6 + i] += qb;
        }
    }

    V3 bearing_cam(float x, float y) const {  // unit bearing in IMU axes of a gray pixel
        const V3 c{(x - cx) / fx, (y - cy) / fy, 1.0};
        const V3 b = R_ic * c;
        return b * (1.0 / norm(b));
    }
    bool project(const V3& bi, float& x, float& y) const {  // IMU-axes direction -> gray pixel
        const V3 c = R_ci * bi;
        if (c.z < 0.1) return false;
        x = (float)(fx * c.x / c.z + cx); y = (float)(fy * c.y / c.z + cy);
        return true;
    }

    void track(const uint8_t* rgb, const M3& Rp, double dtf) {  // Rp: attitude at the previous frame
        const int slot = frame % HIST;
        int k = 0;
        const M3 dR = Rp.t() * R;
        const V3 dc = R.tmul(v * dtf);   // camera displacement since the last frame, current IMU axes (lever arm ignored)
        for (int i = 0; i < nfeats; i++) {
            Feature& f = feats[i];
            if (P_TRACK_EVERY > 1 && (f.id + frame) % P_TRACK_EVERY) {   // not tracked this frame: keep the last bearing
                f.bw[slot] = f.bw[(frame + HIST - 1) % HIST];
                f.stale++;
                feats[k++] = f;
                continue;
            }
            // predicted position: rotation, plus translation at the assumed depth
            const V3 b0 = bearing_cam(f.x, f.y);
            V3 bp = dR.tmul(b0);
            if (P_PRED_DEPTH > 0) bp = bp * P_PRED_DEPTH - dc * (double)(f.stale + 1);
            float px, py;
            if (!project(bp, px, py)) continue;
            float ddx = px - f.tpl.ix, ddy = py - f.tpl.iy;
            int wcx, wcy;
            if (!f.tpl.ok || !track_template(rgb, f.tpl, ddx, ddy, win, wcx, wcy)) continue;
            f.x = f.tpl.ix + ddx; f.y = f.tpl.iy + ddy;
            const int ix = (int)std::lround(f.x), iy = (int)std::lround(f.y);
            make_template(win, wcx, wcy, ix, iy, f.tpl);
            if (!f.tpl.ok) continue;
            f.bw[slot] = C.tmul(R * bearing_cam(f.x, f.y));
            f.n++;
            f.stale = 0;
            feats[k++] = f;
        }
        nfeats = k;
    }

    void detect(const uint8_t* rgb) {
        const int ncell = P_GRID_X * P_GRID_Y;
        for (int c = 0, tried = 0; c < ncell && tried < P_DETECT_CELLS && nfeats < P_MAX_FEAT; c++) {
            const int cell = (detect_cell + c) % ncell;
            const int gx = cell % P_GRID_X, gy = cell / P_GRID_X;
            const int x0 = gx * CW, y0 = gy * CH, x1 = x0 + CW, y1 = y0 + CH;
            bool occupied = false;
            for (int i = 0; i < nfeats; i++)
                if (feats[i].x >= x0 && feats[i].x < x1 && feats[i].y >= y0 && feats[i].y < y1) { occupied = true; break; }
            if (occupied) continue;
            tried++;
            detect_cell = cell + 1;
            int bx = -1, by = -1;
            const int lx0 = std::max(x0, WR + 1), ly0 = std::max(y0, WR + 1), lx1 = std::min(x1, GW - WR - 1), ly1 = std::min(y1, GH - WR - 1);
            if (lx1 <= lx0 || ly1 <= ly0) continue;
            if (P_DET_CAND > 0) {
                if (!sparse_corner(rgb, lx0, ly0, lx1, ly1, (unsigned)(frame * 31 + cell), bx, by)) continue;
            } else if (!best_corner(rgb, lx0, ly0, lx1, ly1, bx, by)) continue;
            int wcx = bx, wcy = by;
            if (!extract_window(rgb, wcx, wcy, win)) continue;
            Feature& f = feats[nfeats];
            make_template(win, wcx, wcy, bx, by, f.tpl);
            if (!f.tpl.ok) continue;
            f.id = next_id++; f.x = (float)bx; f.y = (float)by; f.n = 1; f.stale = 0;
            f.bw[frame % HIST] = C.tmul(R * bearing_cam(f.x, f.y));
            nfeats++;
        }
    }

    void update() {
        const FrameRec& b = rec[frame % HIST];
        int nmeas = 0, ncue = 0;
        double cue_q[P_MAX_FEAT], cue_tau[P_MAX_FEAT]; V3 cue_b2[P_MAX_FEAT], cue_M[P_MAX_FEAT], cue_lev[P_MAX_FEAT];
        const double sb2 = P_SIGMA_PX * P_SIGMA_PX / (fx * fx) * 2 * P_MEAS_INFLATE;
        for (int i = 0; i < nfeats; i++) {
            Feature& f = feats[i];
            const int base = std::min(f.n - 1, P_BASE);
            if (base < P_MIN_BASE || (f.id + frame) % P_MEAS_EVERY || f.stale) continue;
            if (nmeas >= P_MAX_MEAS) break;
            nmeas++;
            const FrameRec& a = rec[(frame - base) % HIST];
            const double tau = b.t - a.t;
            if (tau <= 0) continue;
            const V3 M = C * ((b.A2 - a.A2) - (b.A1 - a.A1) * (a.t - t0));
            const V3 b1 = C * f.bw[(frame - base) % HIST], b2 = C * f.bw[frame % HIST];
            const V3 dlev = C * (b.lev - a.lev);
            const V3 n = cross(b1, b2);
            const V3 dp = v * tau - M - g * (0.5 * tau * tau) + dlev;
            const double y = -dot(n, dp);
            double h[NS];
            h[0] = tau * n.x; h[1] = tau * n.y; h[2] = tau * n.z;
            const double c2 = -0.5 * tau * tau;
            h[3] = c2 * n.x; h[4] = c2 * n.y; h[5] = c2 * n.z;
            if (P_ATT) {
                const V3 hb = R.tmul(cross(b1, cross(b2, dp))) * tau;
                h[6] = hb.x; h[7] = hb.y; h[8] = hb.z;
                const V3 ht = cross(n, v * tau - g * (0.5 * tau * tau));
                h[9] = ht.x; h[10] = ht.y; h[11] = ht.z;
            } else if (P_BIAS) {
                const V3 q = R.tmul(cross(dp, b1));
                const V3 c2i = R.tmul(b2);
                const V3 hb = cross(q, c2i) * tau;
                h[6] = hb.x; h[7] = hb.y; h[8] = hb.z;
            }
            const double r = sb2 * (dot(dp, dp) + 0.01);
            if (kf_update(h, y, r, P_CHI2)) st_used++; else st_rej++;
            if (P_RHO_SIG > 0 && (f.id + frame) % P_RHO_EVERY == 0) {  // flow magnitude |b1 x b2| = rho |dp x b2|, prior rho0 +- sig
                const V3 m = cross(dp, b2);
                const double mn = norm(m);
                if (mn > 1e-4) {
                    if (P_RHO_MEDIAN) {   // keep for the per-frame median row
                        if (ncue < P_MAX_FEAT) { cue_q[ncue] = dot(cross(b1, b2), m) / (mn * mn * P_RHO0); cue_b2[ncue] = b2; cue_tau[ncue] = tau; cue_M[ncue] = M; cue_lev[ncue] = dlev; ncue++; }
                    } else {
                        const V3 hv = cross(b2, m * (1.0 / mn)) * P_RHO0;
                        double h2[NS] = {0};
                        h2[0] = tau * hv.x; h2[1] = tau * hv.y; h2[2] = tau * hv.z;
                        h2[3] = -0.5 * tau * tau * hv.x; h2[4] = -0.5 * tau * tau * hv.y; h2[5] = -0.5 * tau * tau * hv.z;
                        const double y2 = dot(cross(b1, b2), m) / mn - P_RHO0 * mn;  // signed: cheirality
                        kf_update(h2, y2, P_RHO_SIG * P_RHO_SIG * mn * mn + sb2 + 1e-6, P_CHI2, P_RHO_NUPD);
                    }
                }
            }
        }
        if (P_RHO_MEDIAN && ncue >= P_RHO_MEDIAN) {
            // q = (rho_obs / rho0) per feature: 1 if the current velocity matches the prior depth; median feature gives one row
            int idx[P_MAX_FEAT];
            for (int i = 0; i < ncue; i++) idx[i] = i;
            std::nth_element(idx, idx + ncue / 2, idx + ncue, [&](int a2, int b3) { return cue_q[a2] < cue_q[b3]; });
            const int k = idx[ncue / 2];
            const double tau = cue_tau[k];
            const V3 dp = v * tau - cue_M[k] - g * (0.5 * tau * tau) + cue_lev[k];
            const V3 m = cross(dp, cue_b2[k]);
            const double mn = norm(m);
            const V3 hv = cross(cue_b2[k], m * (1.0 / mn)) * P_RHO0;
            double h2[NS] = {0};
            h2[0] = tau * hv.x; h2[1] = tau * hv.y; h2[2] = tau * hv.z;
            h2[3] = -0.5 * tau * tau * hv.x; h2[4] = -0.5 * tau * tau * hv.y; h2[5] = -0.5 * tau * tau * hv.z;
            const double y2 = (cue_q[k] - 1.0) * P_RHO0 * mn;
            kf_update(h2, y2, P_RHO_SIG * P_RHO_SIG * mn * mn + sb2 + 1e-6, 0, P_RHO_NUPD);
        }
    }

    void pseudo() {
        if (P_GNORM_SIGMA > 0) {
            const double gn = norm(g);
            double h[NS] = {0};
            h[3] = g.x / gn; h[4] = g.y / gn; h[5] = g.z / gn;
            kf_update(h, GRAVITY - gn, P_GNORM_SIGMA * P_GNORM_SIGMA, 0);
        }
        if (P_VPRIOR_SIGMA > 0) {
            for (int i = 0; i < 3; i++) {
                double h[NS] = {0}; h[i] = 1;
                kf_update(h, -v[i], P_VPRIOR_SIGMA * P_VPRIOR_SIGMA, 0);
            }
        }
        const double s = norm(v);
        if (!(s < P_MAX_SPEED)) v = std::isfinite(s) ? v * (P_MAX_SPEED / s) : V3{0, 0, 0};
    }

    void step(double t, const uint8_t* rgb, const double* imu, int count, double* out) {
        if (frame == 0) {
            V3 fm{0, 0, 0};
            for (int i = 0; i < count; i++) fm = fm + V3{imu[7 * i + 1], imu[7 * i + 2], imu[7 * i + 3]};
            const double n = norm(fm);
            g = n > 1e-6 ? fm * (-GRAVITY / n) : V3{0, 0, -GRAVITY};
            reset_cov();
            t0 = t; t_last = t;
        } else {
            const M3 Rp = R;
            const double tp = t_prev;
            propagate(imu, count);
#ifndef __wasm__
            if (truth_on & 1) for (int i = 0; i < 9; i++) R.m[i / 3][i % 3] = truth_R[i];
            if (truth_on & 2) v = {truth_v[0], truth_v[1], truth_v[2]};
            if (truth_on & 4) g = {truth_g[0], truth_g[1], truth_g[2]};
#endif
            track(rgb, Rp, t - tp);
        }
        FrameRec& fr = rec[frame % HIST];
        fr.t = t; fr.A1 = A1; fr.A2 = A2; fr.lev = C.tmul(R * t_ic);
        if (frame % 50 == 49) orthonormalize(C);
        if (frame > 0) update();
        pseudo();
        detect(rgb);
        // non-finite watchdog
        if (!std::isfinite(v.x + v.y + v.z + g.x + g.y + g.z + p.x + p.y + p.z)) {
            v = {0, 0, 0}; if (!std::isfinite(g.x + g.y + g.z)) g = {0, 0, -GRAVITY};
            if (!std::isfinite(p.x + p.y + p.z)) p = {0, 0, 0};
            bg = {0, 0, 0}; reset_cov();
        }
        if (P_SHRINK > 0) p = p * (1.0 - P_SHRINK * 0.05);
        frame++;
        t_prev = t;
        write_output(out);
    }

    void write_output(double* out) const {
        const M3 Rc = R_ci * R * R_ic;
        const V3 pc = R_ci * (R * t_ic + p) + t_ci;
        double q[4]; quaternion(Rc, q);
        out[0] = pc.x; out[1] = pc.y; out[2] = pc.z;
        out[3] = q[0]; out[4] = q[1]; out[5] = q[2]; out[6] = q[3];
        const V3 vc_w = v + R * cross(w_last, t_ic);
        const V3 vc = (R * R_ic).tmul(vc_w);
        const V3 wc = R_ci * w_last;
        out[7] = vc.x; out[8] = vc.y; out[9] = vc.z;
        out[10] = wc.x; out[11] = wc.y; out[12] = wc.z;
    }
};

Estimator* estimator = nullptr;

bool json_number(const char* json, uint32_t length, const char* key, double& value) {
    char pattern[32];
    std::snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* end = json + length;
    const size_t pl = std::strlen(pattern);
    for (const char* s = json; s + pl < end; s++)
        if (std::memcmp(s, pattern, pl) == 0) {
            const char* c = s + pl;
            while (c < end && (*c == ' ' || *c == ':')) c++;
            value = std::strtod(c, nullptr);
            return true;
        }
    return false;
}

bool json_matrix4(const char* json, uint32_t length, const char* key, double M[16]) {
    char pattern[32];
    std::snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* end = json + length;
    const size_t pl = std::strlen(pattern);
    for (const char* s = json; s + pl < end; s++)
        if (std::memcmp(s, pattern, pl) == 0) {
            const char* c = s + pl;
            int k = 0;
            while (c < end && k < 16) {
                if ((*c >= '0' && *c <= '9') || *c == '-' || *c == '+' || *c == '.') { char* e; M[k++] = std::strtod(c, &e); c = e; }
                else c++;
            }
            return k == 16;
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
    Estimator& e = *new Estimator();
    const double off = P_POINT ? (double)OFF : 0.5 * (P_DS - 1);
    e.fx = fx / P_DS; e.fy = fy / P_DS;
    e.cx = (cx - off) / P_DS; e.cy = (cy - off) / P_DS;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) e.R_ci.m[i][j] = T[4 * i + j];
    e.t_ci = {T[3], T[7], T[11]};
    e.R_ic = e.R_ci.t();
    e.t_ic = (e.R_ic * e.t_ci) * -1.0;
    estimator = &e;
    return VIOBENCH_STATUS_OK;
}

#ifndef __wasm__
int32_t debug_truth(const double* Rm, const double* vv, const double* gg, int32_t mode) {
    truth_on = mode; for (int i = 0; i < 9; i++) truth_R[i] = Rm[i]; for (int i = 0; i < 3; i++) { truth_v[i] = vv[i]; truth_g[i] = gg[i]; }
    return 0;
}
int32_t debug_nis(double* o) { o[0] = nis_sum; o[1] = nis_n; for (int i = 0; i < 8; i++) o[2 + i] = nis_hist[i]; return 0; }
int32_t debug_state(double* o) {
    Estimator& e = *estimator;
    double vals[] = {(double)e.frame, (double)e.nfeats, (double)e.st_used, (double)e.st_rej, e.v.x, e.v.y, e.v.z, e.g.x, e.g.y, e.g.z,
                     e.bg.x, e.bg.y, e.bg.z, std::sqrt(e.P[0][0]), std::sqrt(e.P[1][1]), std::sqrt(e.P[2][2])};
    for (unsigned i = 0; i < sizeof(vals) / sizeof(double); i++) o[i] = vals[i];
    return 0;
}
#endif

VIOBENCH_EXPORT(step) int32_t step(double t, const uint8_t* frame, const double* imu, int32_t imu_count, double* out) {
    if (!estimator) return VIOBENCH_STATUS_ERROR;
    estimator->step(t, frame, imu, imu_count, out);
    for (int i = 0; i < 13; i++) if (!std::isfinite(out[i])) { std::fill(out, out + 13, 0.0); out[6] = 1.0; break; }
    return VIOBENCH_STATUS_OK;
}

}  // extern "C"
