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
#if defined(__wasm_simd128__) && !defined(P_SCALAR)
#include <wasm_simd128.h>
#endif

// contiguous double kernels (f64x2 SIMD in the wasm build)
inline double dotd(const double* a, const double* b, int n) {
#if defined(__wasm_simd128__) && !defined(P_SCALAR)
    v128_t s0 = wasm_f64x2_splat(0), s1 = wasm_f64x2_splat(0);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
        s1 = wasm_f64x2_add(s1, wasm_f64x2_mul(wasm_v128_load(a + i + 2), wasm_v128_load(b + i + 2)));
    }
    if (i + 2 <= n) { s0 = wasm_f64x2_add(s0, wasm_f64x2_mul(wasm_v128_load(a + i), wasm_v128_load(b + i))); i += 2; }
    s0 = wasm_f64x2_add(s0, s1);
    double r = wasm_f64x2_extract_lane(s0, 0) + wasm_f64x2_extract_lane(s0, 1);
    if (i < n) r += a[i] * b[i];
    return r;
#else
    double r = 0;
    for (int i = 0; i < n; i++) r += a[i] * b[i];
    return r;
#endif
}
inline float dotf(const float* a, const float* b, int n) {
#if defined(__wasm_simd128__) && !defined(P_SCALAR)
    v128_t s0 = wasm_f32x4_splat(0), s1 = wasm_f32x4_splat(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = wasm_f32x4_add(s0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
        s1 = wasm_f32x4_add(s1, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
    }
    if (i + 4 <= n) { s0 = wasm_f32x4_add(s0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i))); i += 4; }
    s0 = wasm_f32x4_add(s0, s1);
    float r = wasm_f32x4_extract_lane(s0, 0) + wasm_f32x4_extract_lane(s0, 1) + wasm_f32x4_extract_lane(s0, 2) + wasm_f32x4_extract_lane(s0, 3);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
#else
    float r = 0;
    for (int i = 0; i < n; i++) r += a[i] * b[i];
    return r;
#endif
}
inline void axpyf(float* y, const float* x, float a, int n) {
#if defined(__wasm_simd128__) && !defined(P_SCALAR)
    const v128_t va = wasm_f32x4_splat(a);
    int i = 0;
    for (; i + 4 <= n; i += 4) wasm_v128_store(y + i, wasm_f32x4_add(wasm_v128_load(y + i), wasm_f32x4_mul(va, wasm_v128_load(x + i))));
    for (; i < n; i++) y[i] += a * x[i];
#else
    for (int i = 0; i < n; i++) y[i] += a * x[i];
#endif
}
inline double dotx(const double* a, const double* b, int n) { return dotd(a, b, n); }
inline float dotx(const float* a, const float* b, int n) { return dotf(a, b, n); }
inline void axpyd(double* y, const double* x, double a, int n);
inline void axpyx(double* y, const double* x, double a, int n) { axpyd(y, x, a, n); }
inline void axpyx(float* y, const float* x, float a, int n) { axpyf(y, x, a, n); }
inline void axpyd(double* y, const double* x, double a, int n) {  // y += a x
#if defined(__wasm_simd128__) && !defined(P_SCALAR)
    const v128_t va = wasm_f64x2_splat(a);
    int i = 0;
    for (; i + 2 <= n; i += 2) wasm_v128_store(y + i, wasm_f64x2_add(wasm_v128_load(y + i), wasm_f64x2_mul(va, wasm_v128_load(x + i))));
    if (i < n) y[i] += a * x[i];
#else
    for (int i = 0; i < n; i++) y[i] += a * x[i];
#endif
}

#if !defined(__wasm__) && defined(P_PROF)
#include <x86intrin.h>
static unsigned long long prof_acc[8], prof_t0;
#define PROF_BEGIN() (prof_t0 = __rdtsc())
#define PROF_END(i) (prof_acc[i] += __rdtsc() - prof_t0)
#else
#define PROF_BEGIN() ((void)0)
#define PROF_END(i) ((void)0)
#endif

namespace {

constexpr int NI = 18;                        // IMU part of the error state
constexpr int NMAX = NI + 6 * (P_CLONES + 1); // full error state (one extra clone while a split update is pending)
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
    if (th2 < 4e-3) { a = 1 - th2 / 6 * (1 - th2 / 20); b = 0.5 - th2 / 24 * (1 - th2 / 30); }  // Taylor: < 1e-10 error below 0.063 rad
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

// ------------------------------------------------------------------ image front end (sparse)
// No full gray image: pixels are read from the RGB frame's green channel, P_DS x P_DS block sums at the gray
// resolution, only in small windows around features and in the grid cells searched for new corners.
constexpr int PH = P_HALF, PN = (2 * PH + 1) * (2 * PH + 1);
constexpr int MARGIN = P_MARGIN;                   // search margin (gray pixels) of a tracking window
constexpr int WR = PH + 1 + MARGIN;                // window radius
constexpr int WS = 2 * WR + 1;                     // window side
constexpr float INV_DS2 = 1.0f / (P_DS * P_DS);

inline float green(const uint8_t* rgb, int x, int y) {  // gray-resolution pixel
    if (P_DS == 1) return rgb[(y * W0 + x) * 3 + 1];
    const uint8_t* p = rgb + ((y * P_DS) * W0 + x * P_DS) * 3 + 1;
    int s = 0;
    for (int dy = 0; dy < P_DS; dy++, p += W0 * 3)
        for (int dx = 0; dx < P_DS; dx++) s += p[3 * dx];
    return s * INV_DS2;
}

// window of side WS centred on integer (cx, cy); false if it does not fit in the image
inline bool extract_window(const uint8_t* rgb, int cx, int cy, float* win) {
    if (cx - WR < 0 || cy - WR < 0 || cx + WR >= GW || cy + WR >= GH) return false;
    for (int y = 0; y < WS; y++)
        for (int x = 0; x < WS; x++) win[y * WS + x] = green(rgb, cx - WR + x, cy - WR + y);
    return true;
}

struct Template {           // patch on the integer grid around (ix, iy) of the previous frame
    int ix, iy;
    float T[PN], gx[PN], gy[PN];
    float i00, i01, i11;    // inverse of the gradient structure matrix
    float contrast;         // mean absolute deviation of the patch
    bool ok;
};

// template from a window centred at (wcx, wcy) for the grid point (ix, iy)
inline void make_template(const float* win, int wcx, int wcy, int ix, int iy, Template& t) {
    t.ix = ix; t.iy = iy; t.ok = false;
    const int ox = ix - wcx + WR, oy = iy - wcy + WR;  // window coords of the centre
    if (ox - PH - 1 < 0 || oy - PH - 1 < 0 || ox + PH + 1 >= WS || oy + PH + 1 >= WS) return;
    float gxx = 0, gxy = 0, gyy = 0;
    int k = 0;
    for (int dy = -PH; dy <= PH; dy++) {
        const float* row = win + (oy + dy) * WS + ox;
        for (int dx = -PH; dx <= PH; dx++, k++) {
            t.T[k] = row[dx];
            const float a = 0.5f * (row[dx + 1] - row[dx - 1]);
            const float b = 0.5f * (row[dx + WS] - row[dx - WS]);
            t.gx[k] = a; t.gy[k] = b;
            gxx += a * a; gxy += a * b; gyy += b * b;
        }
    }
    const float det = gxx * gyy - gxy * gxy;
    if (det < 1e-3f * PN * PN) return;
    float mean = 0;
    for (int q = 0; q < PN; q++) mean += t.T[q];
    mean /= PN;
    float dev = 0;
    for (int q = 0; q < PN; q++) dev += std::fabs(t.T[q] - mean);
    t.contrast = dev / PN;
    t.i00 = gyy / det; t.i01 = -gxy / det; t.i11 = gxx / det;
    t.ok = true;
}

// track the template into the frame: (dx, dy) is the displacement of the grid patch, starting at the prediction.
// win/wcx/wcy receive the window used (for the next template). Returns false if lost.
inline bool track_template(const uint8_t* rgb, const Template& t, float& dx, float& dy, float* win, int& wcx, int& wcy, float& err_out) {
    wcx = (int)std::lround(t.ix + dx); wcy = (int)std::lround(t.iy + dy);
    if (!extract_window(rgb, wcx, wcy, win)) return false;
    float err = 0;
    for (int it = 0; it < P_LK_ITERS; it++) {
        const float px = t.ix + dx - (wcx - WR), py = t.iy + dy - (wcy - WR);  // centre in window coords
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
                const float v = w00 * r0[xx] + w01 * r0[xx + 1] + w10 * r1[xx] + w11 * r1[xx + 1];
                const float e = v - t.T[k];
                sx += t.gx[k] * e; sy += t.gy[k] * e;
                err += std::fabs(e);
            }
        }
        const float ux = t.i00 * sx + t.i01 * sy, uy = t.i01 * sx + t.i11 * sy;
        dx -= ux; dy -= uy;
        if (ux * ux + uy * uy < 0.0009f) break;
    }
    err_out = err / PN;
    return err_out < P_LK_MAX_ERR && err_out < P_LK_REL_ERR * t.contrast + 1.0f;
}

// best Shi-Tomasi corner in a cell (gray coords [x0, x1) x [y0, y1)); score is the min eigenvalue of the 3 x 3
// structure tensor per pixel. Returns false if none above P_MIN_SCORE.
constexpr int CELL_MAX = (GW / P_GRID_X + 4) * (GH / P_GRID_Y + 4);
bool best_corner(const uint8_t* rgb, int x0, int y0, int x1, int y1, int& bx, int& by) {
    static float buf[CELL_MAX], gxb[CELL_MAX], gyb[CELL_MAX];
    const int w = x1 - x0 + 4, h = y1 - y0 + 4;    // 2-pixel border for gradients + window
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) buf[y * w + x] = green(rgb, x0 - 2 + x, y0 - 2 + y);
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            gxb[y * w + x] = buf[y * w + x + 1] - buf[y * w + x - 1];
            gyb[y * w + x] = buf[(y + 1) * w + x] - buf[(y - 1) * w + x];
        }
    float best = P_MIN_SCORE * 4 * 9;  // gradients are 2x central differences, 9 pixels
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
            const float score = tr - d;
            if (score > best) { best = score; bx = x0 - 2 + x; by = y0 - 2 + y; }
        }
    return bx >= 0;
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
    Template tpl;          // patch from the previous frame
};

struct Clone { int id; M3 R; V3 p; };

#if P_FLOATCOV
typedef float preal;
#else
typedef double preal;
#endif
preal P[NMAX][NMAX];

struct Estimator {
    // calibration
    double fx, fy, cx, cy;      // at the gray resolution
    M3 R_ci; V3 t_ci;           // IMU -> camera
    M3 R_ic; V3 t_ic;           // camera -> IMU
    // state
    M3 R = M3::eye(); V3 p{0, 0, 0}, v{0, 0, 0}, bg{0, 0, 0}, ba{0, 0, 0}, g{0, 0, -GRAVITY};
    Clone clones[P_CLONES + 1]; int nclones = 0, next_clone_id = 0;
    Feature feats[P_MAX_FEAT]; int nfeats = 0, next_feat_id = 0;
    int detect_cell = 0;   // round-robin start of the corner search
    int frame = 0;
    double t_last = 0;
    V3 w_last{0, 0, 0};
    double depth_est = P_PRED_DEPTH;
    double log_depth = std::log(P_DEPTH_PRIOR);  // running mean log-depth of well-triangulated points (adaptive prior)
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
        // P <- Phi P Phi^T with the sparse Phi = I + blocks (block order th 0, p 1, v 2, bg 3, ba 4, g 5)
        struct Blk { int r, c; const M3* M; };
        const Blk blks[] = {{0, 3, &th_bg}, {1, 0, &p_th}, {1, 2, &p_v}, {1, 3, &p_bg}, {1, 4, &p_ba}, {1, 5, &p_g},
                            {2, 0, &v_th}, {2, 3, &v_bg}, {2, 4, &v_ba}, {2, 5, &v_g}};
        constexpr int NB = sizeof(blks) / sizeof(blks[0]);
        const int n = dim();
        static double tmp[9][NMAX];  // new rows of the th, p, v blocks: (Phi P)[0:9, :]
        for (int i = 0; i < 9; i++) for (int j = 0; j < n; j++) tmp[i][j] = P[i][j];
        for (int q = 0; q < NB; q++) {
            const M3& M = *blks[q].M;
            const int r0 = 3 * blks[q].r, c0 = 3 * blks[q].c;
            for (int i = 0; i < 3; i++) {
                const double m0 = M.m[i][0], m1 = M.m[i][1], m2 = M.m[i][2];
                double* out = tmp[r0 + i];
                const preal *a0 = P[c0], *a1 = P[c0 + 1], *a2 = P[c0 + 2];
                for (int j = 0; j < n; j++) out[j] += m0 * a0[j] + m1 * a1[j] + m2 * a2[j];
            }
        }
        // rows 0..8 of Phi P are tmp; rows 9..17 are P's. Now (Phi P) Phi^T: columns 0..8 change the same way.
        for (int i = 0; i < 9; i++) for (int j = 0; j < n; j++) P[i][j] = tmp[i][j];
        static double tc[NI][9];  // new columns 0..8 for the IMU rows
        for (int i = 0; i < NI; i++) for (int j = 0; j < 9; j++) tc[i][j] = P[i][j];
        for (int q = 0; q < NB; q++) {
            const M3& M = *blks[q].M;
            const int r0 = 3 * blks[q].r, c0 = 3 * blks[q].c;
            for (int i = 0; i < NI; i++) {
                const double a0 = P[i][c0], a1 = P[i][c0 + 1], a2 = P[i][c0 + 2];
                for (int j = 0; j < 3; j++) tc[i][r0 + j] += M.m[j][0] * a0 + M.m[j][1] * a1 + M.m[j][2] * a2;
            }
        }
        for (int i = 0; i < NI; i++) for (int j = 0; j < 9; j++) P[i][j] = tc[i][j];
        // symmetric fill: IMU-IMU block lower part and the IMU-clone cross terms
        for (int i = 0; i < NI; i++) for (int j = 0; j < i; j++) { const double v2 = 0.5 * (P[i][j] + P[j][i]); P[i][j] = P[j][i] = v2; }
        for (int i = 0; i < 9; i++) for (int j = NI; j < n; j++) P[j][i] = P[i][j];
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
    bool triangulate(const Feature& f, V3& Pw, double* abr, M3* Ranchor, bool use_prior) {  // abr: al, be, rho, posterior std, prior std
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
        if (P_ADAPT_DEPTH) { rho_prior = std::exp(-log_depth); sig_prior = P_RHO_REL * rho_prior; }
        if (P_CARRY_DEPTH && f.has_pw) {
            const V3 Pa = Rwc[0].t() * (f.pw - pwc[0]);
            if (Pa.z > 0.1) {
                rho_prior = 1.0 / Pa.z;
                sig_prior = std::max((double)P_RHO_SIGMA_MIN, std::min((double)P_RHO_SIGMA, f.sig_rho * P_CARRY_INFLATE));
            }
        }
        double al = f.obs[0].u, be = f.obs[0].v, rho = rho_prior;
        const double sn = P_SIGMA_PX / (fx * P_DS);
        const double wprior = use_prior && P_RHO_SIGMA > 0 ? std::pow(sn / sig_prior, 2) : 0.0;
        double h22inv = 0;
        double lambda = 1e-3;
        double cost_prev = 1e30;
        for (int it = 0; it < P_TRI_ITERS; it++) {
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
    double Hs[MAX_ROWS][6 * (P_CLONES + 1)];  // over the clone columns only (MSCKF rows do not involve the IMU state)
    double rs[MAX_ROWS];

    // builds a feature's projected rows into Hs/rs starting at `row`; returns the number of rows (0 if rejected)
    int feature_rows(Feature& f, int row) {
        V3 Pw; double abr[5]; M3 Ra;
        st_cand++;
        // well-observed points need no depth prior: project the whole point out (classic MSCKF); poorly observed ones
        // (little parallax) keep the inverse-depth prior
        PROF_BEGIN();
        bool prior = P_RHO_SIGMA > 0;
        if (prior && P_PRIOR_SWITCH > 0 && triangulate(f, Pw, abr, &Ra, false) && abr[3] < P_PRIOR_SWITCH * abr[2]) {
            prior = false;
            if (P_ADAPT_DEPTH) {
                log_depth += P_DEPTH_RATE * (-std::log(abr[2]) - log_depth);
                log_depth = std::max(std::log(0.5), std::min(std::log(40.0), log_depth));
                depth_est = std::exp(log_depth);
            }
        }
        else if (!triangulate(f, Pw, abr, &Ra, prior)) { st_tri_fail++; PROF_END(5); return 0; }
        PROF_END(5);
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
        const int npro = prior ? 2 : 3;
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
        double r2 = 0;
        for (int i = 0; i < mr; i++) r2 += r[i] * r[i];
        PROF_BEGIN();
        if (P_CHI2 && r2 > P_CHI2_FAST * mr * sig * sig) {
            // S = Hx Pcc Hx^T + sig^2 I over the involved clones; chi-square test on r
            int gidx[6 * MAX_OBS];
            for (int c = 0; c < nc; c++) gidx[c] = NI + 6 * slot[c / 6] + c % 6;
            double Pcc[6 * MAX_OBS][6 * MAX_OBS];
            for (int a2 = 0; a2 < nc; a2++) {
                const preal* Prow = P[gidx[a2]];
                for (int c = 0; c < nc; c++) Pcc[a2][c] = Prow[gidx[c]];
            }
            double HP[2 * MAX_OBS][6 * MAX_OBS];
            for (int i = 0; i < mr; i++) {
                for (int c = 0; c < nc; c++) HP[i][c] = 0;
                for (int k = 0; k < nc; k++) axpyd(HP[i], Pcc[k], Hx[i][k], nc);
            }
            double S[2 * MAX_OBS][2 * MAX_OBS];
            for (int i = 0; i < mr; i++)
                for (int j = 0; j < mr; j++) S[i][j] = dotd(HP[i], Hx[j], nc) + (i == j ? sig * sig : 0);
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
            if (chi > P_CHI2_SCALE * chi2_95[std::min(mr, 17)]) { st_chi_fail++; PROF_END(6); return 0; }
        }
        PROF_END(6);
        f.has_pw = true; f.pw = Pw; f.sig_rho = abr[3];
        for (int i = 0; i < mr; i++) {
            for (int c = 0; c < 6 * (P_CLONES + 1); c++) Hs[row + i][c] = 0;
            for (int c = 0; c < nc; c++) Hs[row + i][6 * slot[c / 6] + c % 6] += Hx[i][c];
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

    // QR-compress the stacked rows to at most nc rows (nc = clone columns they involve); returns the row count
    int compress_rows(int m, int nc) {
        // compress when there are more rows than clone columns (Householder QR of [H | r])
        if (m > nc) {
            static double u[MAX_ROWS];
            double d[6 * (P_CLONES + 1)];
            for (int j = 0; j < nc; j++) {
                double s = 0;
                for (int i = j; i < m; i++) s += Hs[i][j] * Hs[i][j];
                const double nrm = std::sqrt(s);
                if (nrm < 1e-14) continue;
                const double alpha = Hs[j][j] > 0 ? -nrm : nrm;
                for (int i = j; i < m; i++) u[i] = Hs[i][j];
                u[j] -= alpha;
                const double un = s - Hs[j][j] * Hs[j][j] + u[j] * u[j];
                if (un < 1e-28) continue;
                const double beta = 2.0 / un;
                const int w = nc - j;
                for (int c = 0; c < w; c++) d[c] = 0;
                double dr = 0;
                for (int i = j; i < m; i++) { axpyd(d, Hs[i] + j, u[i], w); dr += u[i] * rs[i]; }
                for (int i = j; i < m; i++) { axpyd(Hs[i] + j, d, -beta * u[i], w); rs[i] -= beta * dr * u[i]; }
            }
            m = nc;
        }
        return m;
    }

    void ekf_update(int m, int nc) {
        const int n = dim();
        st_last_m = m;
        const double sig = P_SIGMA_PX / (fx * P_DS);
        constexpr int MR = MAX_ROWS > NMAX ? MAX_ROWS : NMAX;
        static preal PHt[NMAX][MR];
        static preal Hf[MR][6 * (P_CLONES + 1)];
        static double S[MR][MR];
        static preal Lf[MR][MR];
        for (int j = 0; j < m; j++) for (int k = 0; k < nc; k++) Hf[j][k] = (preal)Hs[j][k];
        for (int i = 0; i < n; i++) {
            const preal* Pi = &P[i][NI];
            if (!(P_EKF_SKIP & 1)) for (int j = 0; j < m; j++) PHt[i][j] = dotx(Pi, Hf[j], nc);
        }
        {
            static preal Sf[MR][MR];
            for (int i = 0; i < m; i++) {
                preal* Si = Sf[i];
                for (int j = 0; j < m; j++) Si[j] = 0;
                if (!(P_EKF_SKIP & 2)) for (int k = 0; k < nc; k++) axpyx(Si, PHt[NI + k], Hf[i][k], m);
            }
            for (int i = 0; i < m; i++) for (int j = 0; j <= i; j++) S[i][j] = 0.5 * ((double)Sf[i][j] + (double)Sf[j][i]) + (i == j ? sig * sig : 0);
        }
        // Cholesky S = L L^T (in place, lower), in double
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= i; j++) {
                const double s = S[i][j] - dotd(S[i], S[j], j);
                if (i == j) { if (s <= 1e-30) return; S[i][i] = std::sqrt(s); }
                else S[i][j] = s / S[j][j];
            }
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) Lf[i][j] = j <= i ? (preal)S[i][j] : 0;
        // W = P H^T L^{-T} (one triangular solve per state), then P -= W W^T and dx = W L^{-1} r
        static preal Wm[NMAX][MR];
        double z[MR];
        for (int a2 = 0; a2 < m; a2++) z[a2] = (rs[a2] - dotd(S[a2], z, a2)) / S[a2][a2];
        preal zf[MR];
        for (int a2 = 0; a2 < m; a2++) zf[a2] = (preal)z[a2];
        double dx[NMAX];
        for (int i = 0; i < n; i++) {
            preal* w = Wm[i];
            const preal* ph = PHt[i];
            if (!(P_EKF_SKIP & 4)) for (int a2 = 0; a2 < m; a2++) w[a2] = (ph[a2] - dotx(Lf[a2], w, a2)) / Lf[a2][a2];
            dx[i] = dotx(w, zf, m);
        }
        if (!(P_EKF_SKIP & 8)) for (int i = 0; i < n; i++) {
            const preal* wi = Wm[i];
            for (int j = i; j < n; j++) {
                const preal val = P[i][j] - dotx(wi, Wm[j], m);
                P[i][j] = val; P[j][i] = val;
            }
        }
        for (int i = 0; i < n; i++) if (P[i][i] < 1e-12) P[i][i] = 1e-12;
        if (!(P_EKF_SKIP & 16)) apply_correction(dx);
    }

    // features with >= P_MIN_OBS observations in `ids` (feature indices) are used in one update
    // weak zero-mean velocity pseudo-measurement (bounds IMU-only drift when vision starves)
    void velocity_prior(double sigma) {
        const int n = dim();
        double S[3][3];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) S[i][j] = P[6 + i][6 + j] + (i == j ? sigma * sigma : 0);
        // invert 3x3
        const double det = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0]) + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
        if (std::fabs(det) < 1e-30) return;
        double I[3][3];
        I[0][0] = (S[1][1] * S[2][2] - S[1][2] * S[2][1]) / det; I[0][1] = (S[0][2] * S[2][1] - S[0][1] * S[2][2]) / det; I[0][2] = (S[0][1] * S[1][2] - S[0][2] * S[1][1]) / det;
        I[1][0] = (S[1][2] * S[2][0] - S[1][0] * S[2][2]) / det; I[1][1] = (S[0][0] * S[2][2] - S[0][2] * S[2][0]) / det; I[1][2] = (S[0][2] * S[1][0] - S[0][0] * S[1][2]) / det;
        I[2][0] = (S[1][0] * S[2][1] - S[1][1] * S[2][0]) / det; I[2][1] = (S[0][1] * S[2][0] - S[0][0] * S[2][1]) / det; I[2][2] = (S[0][0] * S[1][1] - S[0][1] * S[1][0]) / det;
        static double K[NMAX][3];
        const double r[3] = {-v.x, -v.y, -v.z};
        double dx[NMAX];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < 3; a++) K[i][a] = P[i][6] * I[0][a] + P[i][7] * I[1][a] + P[i][8] * I[2][a];
            dx[i] = K[i][0] * r[0] + K[i][1] * r[1] + K[i][2] * r[2];
        }
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++) {
                const double val = P[i][j] - (K[i][0] * P[6][j] + K[i][1] * P[7][j] + K[i][2] * P[8][j]);
                P[i][j] = val; P[j][i] = val;
            }
        apply_correction(dx);
    }

    int pending_m = 0, pending_nc = 0;
    int st_last_m = 0, st_last_rows = 0, st_last_ni = 0;  // split update: rows computed at a clone frame, applied at the next frame

    void update_with(const int* idx, int count, bool apply = true) {
        int rows = 0, used = 0;
        double depth_sum = 0; int depth_n = 0;
        for (int q = 0; q < count && used < P_MAX_UPDATE_FEAT; q++) {
            Feature& f = feats[idx[q]];
            if (f.nobs < P_MIN_OBS) continue;
            unsigned long long t_ = 0;
#if !defined(__wasm__) && defined(P_PROF)
            t_ = __rdtsc();
#endif
            const int got = feature_rows(f, rows);
#if !defined(__wasm__) && defined(P_PROF)
            prof_acc[3] += __rdtsc() - t_;
#endif
            (void)t_;
            if (got > 0) { rows += got; used++; st_used++; st_rows += got; (void)depth_sum; (void)depth_n; }
        }
        if (rows == 0) return;
        const int nc = 6 * nclones;
        st_last_rows = rows; st_last_ni = count;
        PROF_BEGIN(); rows = compress_rows(rows, nc); PROF_END(7);
        if (apply) { PROF_BEGIN(); ekf_update(rows, nc); PROF_END(4); }
        else { pending_m = rows; pending_nc = nc; }
    }

    // ---------------------------------------------------------------- front end
    void pixel_to_norm(float x, float y, double& u, double& v) const { u = (x - cx) / fx; v = (y - cy) / fy; }

    void detect(const uint8_t* rgb, int max_cells) {
        int nalive = 0;
        for (int k = 0; k < nfeats; k++) nalive += feats[k].alive;
        if (nalive >= P_DETECT_BELOW) return;
        constexpr int CW = GW / P_GRID_X, CH = GH / P_GRID_Y;
        int occupied[P_GRID_Y * P_GRID_X] = {};
        for (int k = 0; k < nfeats; k++) {
            if (!feats[k].alive) continue;
            const int gx = std::min(P_GRID_X - 1, std::max(0, (int)(feats[k].x / CW))), gy = std::min(P_GRID_Y - 1, std::max(0, (int)(feats[k].y / CH)));
            occupied[gy * P_GRID_X + gx]++;
        }
        const int border = WR + 1;
        int searched = 0;
        for (int q = 0; q < P_GRID_X * P_GRID_Y && nfeats < P_MAX_FEAT && searched < max_cells; q++) {
            const int c = (detect_cell + q) % (P_GRID_X * P_GRID_Y);
            if (occupied[c]) continue;
            const int gx = c % P_GRID_X, gy = c / P_GRID_X;
            const int x0 = std::max(border, gx * CW), x1 = std::min(GW - border, (gx + 1) * CW);
            const int y0 = std::max(border, gy * CH), y1 = std::min(GH - border, (gy + 1) * CH);
            searched++;
            int bx, by;
            if (x1 - x0 < 3 || y1 - y0 < 3 || !best_corner(rgb, x0, y0, x1, y1, bx, by)) continue;
            Feature& f = feats[nfeats];
            static float win[WS * WS];
            if (!extract_window(rgb, bx, by, win)) continue;
            make_template(win, bx, by, bx, by, f.tpl);
            if (!f.tpl.ok) continue;
            nfeats++;
            f.id = next_feat_id++; f.x = (float)bx; f.y = (float)by; f.nobs = 0; f.alive = true; f.has_pw = false;
        }
        detect_cell = (detect_cell + std::max(searched, 1)) % (P_GRID_X * P_GRID_Y);
    }

    // fuel attribution: run `fn` once on a saved copy of the estimator and P, then restore (P_DOUBLE bitmask)
    template <class F> void twice(int bit, F fn) {
        if (!(P_DOUBLE & bit)) return;
        static Estimator* saved = nullptr;
        static preal Psaved[NMAX][NMAX];
        if (!saved) saved = (Estimator*)std::malloc(sizeof(Estimator));
        std::memcpy((void*)saved, (const void*)this, sizeof(Estimator));
        std::memcpy(Psaved, P, sizeof(P));
        fn();
        std::memcpy((void*)this, (const void*)saved, sizeof(Estimator));
        std::memcpy(P, Psaved, sizeof(P));
    }

    // ---------------------------------------------------------------- per frame
    void step(double t, const uint8_t* rgb, const double* imu, int count, double* out) {
        if (frame == 0) {
            init_state(imu, count);
            t_last = t;
            add_clone();
            detect(rgb, P_DETECT_CELLS_FIRST);
            record_obs();
        } else {
            const M3 R_prev = R; const V3 p_prev = p;
            twice(8, [&] { propagate(imu, count); });
            twice(64, [] {});
            PROF_BEGIN(); if (!(P_SKIP & 8)) propagate(imu, count); PROF_END(0);
            // light frames (no EKF update pending) take the velocity prior and the corner search
            const bool light = !P_BALANCE || !P_SPLIT || pending_m == 0;
            const double vp_sigma = P_BALANCE && P_SPLIT ? P_VPRIOR_SIGMA / std::sqrt((double)P_CLONE_EVERY) : P_VPRIOR_SIGMA;
            twice(4, [&] { velocity_prior(vp_sigma); });
            if (P_VPRIOR_SIGMA > 0 && !(P_SKIP & 4) && (light || !P_BALANCE)) velocity_prior(vp_sigma);
            twice(1, [&] { track(rgb, R_prev, p_prev); });
            PROF_BEGIN(); if (!(P_SKIP & 1)) track(rgb, R_prev, p_prev); PROF_END(1);
            const bool clone_frame = frame % P_CLONE_EVERY == 0;
            if (P_SPLIT == 2) {
                // rows on the frame before a clone frame; at the clone frame: EKF update, drop the oldest clone, add the new one
                if (clone_frame) {
                    if (pending_m) {
                        twice(16, [&] { ekf_update(pending_m, pending_nc); });
                        PROF_BEGIN(); ekf_update(pending_m, pending_nc); PROF_END(4);
                        pending_m = 0;
                    }
                    if (nclones == P_CLONES) {
                        const int oldest = clones[0].id;
                        for (int k = 0; k < nfeats; k++) {
                            Feature& f = feats[k];
                            if (f.nobs > 0 && f.obs[0].clone_id == oldest) { for (int q = 1; q < f.nobs; q++) f.obs[q - 1] = f.obs[q]; f.nobs--; }
                        }
                        remove_clone(0);
                    }
                    add_clone();
                    record_obs();
                } else if ((frame + 1) % P_CLONE_EVERY == 0) {
                    int idx[P_MAX_FEAT]; int ni = 0, nforced = 0;
                    if (nclones == P_CLONES) {
                        const int oldest = clones[0].id;
                        for (int k = 0; k < nfeats; k++) if (feats[k].alive && feats[k].nobs > 0 && feats[k].obs[0].clone_id == oldest) idx[ni++] = k;
                        nforced = ni;
                    }
                    for (int k = 0; k < nfeats; k++) if (!feats[k].alive) idx[ni++] = k;
                    twice(32, [&] { if (ni) update_with(idx, ni, false); });
                    if (ni) update_with(idx, ni, false);
                    for (int k = 0; k < nforced; k++) { Feature& f = feats[idx[k]]; if (f.nobs >= P_MIN_OBS) f.nobs = 0; }
                    for (int k = nforced; k < ni; k++) feats[idx[k]].nobs = -1;
                }
                int w = 0;
                for (int k = 0; k < nfeats; k++) if (feats[k].alive || feats[k].nobs >= 0) feats[w++] = feats[k];
                nfeats = w;
            } else if (P_SPLIT) {
                // rows at clone frames, the EKF update at the next frame; lost features wait for the next clone frame
                if (pending_m) {
                    twice(16, [&] { ekf_update(pending_m, pending_nc); });
                    PROF_BEGIN(); ekf_update(pending_m, pending_nc); PROF_END(4);
                    pending_m = 0;
                }
                if (nclones > P_CLONES) remove_clone(0);
                if (clone_frame) {
                    int idx[P_MAX_FEAT]; int ni = 0, nforced = 0;
                    const bool full = nclones == P_CLONES;
                    if (full) {
                        const int oldest = clones[0].id;
                        for (int k = 0; k < nfeats; k++) if (feats[k].alive && feats[k].nobs > 0 && feats[k].obs[0].clone_id == oldest) idx[ni++] = k;
                        nforced = ni;
                    }
                    for (int k = 0; k < nfeats; k++) if (!feats[k].alive) idx[ni++] = k;
                    twice(32, [&] { if (ni) update_with(idx, ni, false); });
                    if (ni) update_with(idx, ni, false);
                    for (int k = 0; k < nforced; k++) {
                        Feature& f = feats[idx[k]];
                        if (f.nobs >= P_MIN_OBS) f.nobs = 0;
                        else { for (int q = 1; q < f.nobs; q++) f.obs[q - 1] = f.obs[q]; f.nobs--; }
                    }
                    for (int k = nforced; k < ni; k++) feats[idx[k]].nobs = -1;  // processed: remove below
                    add_clone();  // the oldest clone goes after the pending update
                    record_obs();
                }
                int w = 0;
                for (int k = 0; k < nfeats; k++) if (feats[k].alive || feats[k].nobs >= 0) feats[w++] = feats[k];
                nfeats = w;
            } else {
            // one update per frame: lost features, and at clone frames (window full) the features seen in the oldest clone
            int idx[P_MAX_FEAT]; int ni = 0, nforced = 0;
            const bool drop_oldest = clone_frame && nclones == P_CLONES;
            if (drop_oldest) {
                const int oldest = clones[0].id;
                for (int k = 0; k < nfeats; k++) if (feats[k].alive && feats[k].nobs > 0 && feats[k].obs[0].clone_id == oldest) idx[ni++] = k;
                nforced = ni;
            }
            for (int k = 0; k < nfeats; k++) if (!feats[k].alive) idx[ni++] = k;
            if (ni) update_with(idx, ni);
            if (drop_oldest) {
                for (int k = 0; k < nforced; k++) {
                    Feature& f = feats[idx[k]];
                    if (f.nobs >= P_MIN_OBS) f.nobs = 0;  // used: start a fresh track from here
                    else { for (int q = 1; q < f.nobs; q++) f.obs[q - 1] = f.obs[q]; f.nobs--; }
                }
                remove_clone(0);
            }
            if (clone_frame) {
                add_clone();
                record_obs();
            }
            int w = 0;
            for (int k = 0; k < nfeats; k++) if (feats[k].alive) feats[w++] = feats[k];
            nfeats = w;
            }
            twice(2, [&] { detect(rgb, P_DETECT_CELLS); });
            PROF_BEGIN(); if (!(P_SKIP & 2) && light) detect(rgb, P_BALANCE && P_SPLIT ? P_DETECT_CELLS * P_CLONE_EVERY : P_DETECT_CELLS); PROF_END(2);
            if (clone_frame) record_obs_new();
        }
        frame++;
        write_output(out);
    }

    void record_obs() {
        for (int k = 0; k < nfeats; k++) if (feats[k].alive) push_obs(feats[k]);
    }
    void record_obs_new() {  // newly detected features get their first observation at this clone
        const int id = clones[nclones - 1].id;
        for (int k = 0; k < nfeats; k++)
            if (feats[k].alive && (feats[k].nobs == 0 || feats[k].obs[feats[k].nobs - 1].clone_id != id)) push_obs(feats[k]);
    }
    void push_obs(Feature& f) {
        const int id = clones[nclones - 1].id;
        if (f.nobs == MAX_OBS) { for (int q = 1; q < f.nobs; q++) f.obs[q - 1] = f.obs[q]; f.nobs--; }
        double u, vv; pixel_to_norm(f.x, f.y, u, vv);
        f.obs[f.nobs++] = {id, u, vv};
    }

    void track(const uint8_t* rgb, const M3& R_prev, const V3& p_prev) {
        // predicted camera motion: camera at prev -> camera now
        const M3 Rwc0 = R_prev * R_ic, Rwc1 = R * R_ic;
        const V3 pwc0 = R_prev * t_ic + p_prev, pwc1 = R * t_ic + p;
        const M3 R10 = Rwc1.t() * Rwc0;
        const V3 t10 = Rwc1.t() * (pwc0 - pwc1);
        static float win[WS * WS];
        for (int k = 0; k < nfeats; k++) {
            Feature& f = feats[k];
            if (!f.alive) continue;
            double u, vv; pixel_to_norm(f.x, f.y, u, vv);
            const V3 X = V3{u, vv, 1} * depth_est;
            const V3 Y = R10 * X + t10;
            float px = f.x, py = f.y;
            if (Y.z > 0.05) { px = (float)(fx * Y.x / Y.z + cx); py = (float)(fy * Y.y / Y.z + cy); }
            float dx = px - f.x, dy = py - f.y, err;
            int wcx, wcy;
            bool ok = track_template(rgb, f.tpl, dx, dy, win, wcx, wcy, err);
            if (!ok && (px != f.x || py != f.y)) { dx = dy = 0; ok = track_template(rgb, f.tpl, dx, dy, win, wcx, wcy, err); }
            if (ok) {
                f.x += dx; f.y += dy;
                const int ix = (int)std::lround(f.x), iy = (int)std::lround(f.y);
                // next template from the same window when it covers the new grid patch
                make_template(win, wcx, wcy, ix, iy, f.tpl);
                if (!f.tpl.ok && extract_window(rgb, ix, iy, win)) make_template(win, ix, iy, ix, iy, f.tpl);
                // the template sits on the integer grid: keep the subpixel position, track the grid patch from it
                if (!f.tpl.ok) ok = false;
            }
            if (ok) st_tracked++; else { f.alive = false; st_lost++; }
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
#if P_BENCH
    {   // fuel micro-benchmark (init fuel): 100 runs of one EKF kernel on dummy data, m = nc = 24, n = 42
        static preal A[NMAX][48], Bm[48][6 * (P_CLONES + 1)], C[48][48];
        for (int i = 0; i < NMAX; i++) for (int j = 0; j < 48; j++) A[i][j] = (preal)(0.001 * (i + j));
        for (int i = 0; i < 48; i++) for (int j = 0; j < 6 * (P_CLONES + 1); j++) Bm[i][j] = (preal)(0.002 * (i - j));
        const int m = 24, nc = 24, n = 42;
        volatile double sink = 0;
        for (int rep = 0; rep < 100; rep++) {
            if (P_BENCH == 1) for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) A[i][j] = dotx(&P[i][NI], Bm[j], nc);
            if (P_BENCH == 2) for (int i = 0; i < m; i++) { for (int j = 0; j < m; j++) C[i][j] = 0; for (int k = 0; k < nc; k++) axpyx(C[i], A[NI + k], Bm[i][k], m); }
            if (P_BENCH == 3) for (int i = 0; i < n; i++) for (int j = i; j < n; j++) P[i][j] -= dotx(A[i], A[j], m);
            if (P_BENCH == 4) for (int i = 0; i < n; i++) for (int a2 = 0; a2 < m; a2++) A[i][a2] = (A[i][a2] - dotx(C[a2], A[i], a2)) / (C[a2][a2] + 1);
            sink = sink + A[3][3] + P[2][2] + C[1][1];
        }
    }
#endif
    return VIOBENCH_STATUS_OK;
}

#ifndef __wasm__
// host diagnostics: [frame, nfeats, nclones, tracked, lost, cand, tri_fail, chi_fail, used, rows, v(3), bg(3), ba(3), g(3), sd_v(3), sd_tilt]
int32_t debug_prof(double* o) {
#ifdef P_PROF
    for (int i = 0; i < 8; i++) { o[i] = (double)prof_acc[i]; prof_acc[i] = 0; }
#endif
    return 0;
}
int32_t debug_state(double* o) {
    Estimator& e = *estimator;
    double vals[] = {(double)e.frame, (double)e.nfeats, (double)e.nclones, (double)e.st_tracked, (double)e.st_lost, (double)e.st_cand,
                     (double)e.st_tri_fail, (double)e.st_chi_fail, (double)e.st_used, (double)e.st_rows,
                     e.v.x, e.v.y, e.v.z, e.bg.x, e.bg.y, e.bg.z, e.ba.x, e.ba.y, e.ba.z, e.g.x, e.g.y, e.g.z,
                     std::sqrt(P[6][6]), std::sqrt(P[7][7]), std::sqrt(P[8][8]), std::sqrt(P[15][15]), (double)e.st_last_m, (double)e.st_last_rows, (double)e.st_last_ni};
    e.st_last_m = e.st_last_rows = e.st_last_ni = 0;
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
