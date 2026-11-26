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
#ifdef P_CHECKPSD
#define CHECKPSD(w) check_psd(w)
#else
#define CHECKPSD(w) ((void)0)
#endif

namespace {

constexpr int NI = 18;                        // IMU part of the error state
constexpr int NMAX = NI + 3 * P_SLAM + 6 * (P_CLONES + 1); // full error state: IMU, SLAM points, clones (+1 while a split update is pending)
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
    int slam;              // SLAM slot (point in the state), -1 if none
};

struct Clone { int id; M3 R; V3 p; };

#ifndef __wasm__
static int ext_n = -1;          // host test hook: >= 0 replaces tracking/detection by these observations
static int ext_id[256];
static float ext_x[256], ext_y[256];
#endif

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
    int st_slam_init = 0, st_slam_upd = 0;
    int st_tracked = 0, st_lost = 0, st_cand = 0, st_tri_fail = 0, st_chi_fail = 0, st_used = 0, st_rows = 0;

    // SLAM points: world xyz in the state between the IMU block and the clones
    int nslam = 0, slam_rr = 0;
    V3 slam_p[P_SLAM > 0 ? P_SLAM : 1];
    int slam_bad[P_SLAM > 0 ? P_SLAM : 1];
    int C0() const { return NI + 3 * nslam; }  // first clone column
    int dim() const { return C0() + 6 * nclones; }

    void insert_block(int pos, int size) {  // zero rows/columns at pos (state grows by size)
        const int n = dim();
        for (int i = n - 1; i >= 0; i--) {
            const int ii = i < pos ? i : i + size;
            for (int j = n - 1; j >= 0; j--) { const int jj = j < pos ? j : j + size; P[ii][jj] = P[i][j]; }
        }
        for (int i = 0; i < n + size; i++) for (int k = 0; k < size; k++) { P[pos + k][i] = 0; P[i][pos + k] = 0; }
    }
    void remove_block(int pos, int size, int n) {  // n = dimension before removal
        for (int i = 0; i < n; i++) {
            if (i >= pos && i < pos + size) continue;
            const int ii = i < pos ? i : i - size;
            for (int j = 0; j < n; j++) {
                if (j >= pos && j < pos + size) continue;
                const int jj = j < pos ? j : j - size;
                P[ii][jj] = P[i][j];
            }
        }
    }
    void remove_slam(int slot) {
        const int n = dim();
        remove_block(NI + 3 * slot, 3, n);
        for (int k = slot; k < nslam - 1; k++) { slam_p[k] = slam_p[k + 1]; slam_bad[k] = slam_bad[k + 1]; }
        nslam--;
        for (int k = 0; k < nfeats; k++) {
            if (feats[k].slam == slot) feats[k].slam = -1;
            else if (feats[k].slam > slot) feats[k].slam--;
        }
    }

    // ---------------------------------------------------------------- propagation
    void propagate(const double* imu, int count) {
        // Phi blocks (first order, composed per sample) for the rows that change
        M3 th_bg = M3::zero();
        M3 v_th = M3::zero(), v_bg = M3::zero(), v_ba = M3::zero(), v_g = M3::zero();
        M3 p_th = M3::zero(), p_v = M3::zero(), p_bg = M3::zero(), p_ba = M3::zero(), p_g = M3::zero();
        double T = 0;
        if (P_PHI_FRAME) {
            // mean per sample; Phi once per frame from the interval sums (first order in T, exact at constant R f)
            M3 SR = M3::zero();      // sum R dt
            V3 Sa{0, 0, 0};          // sum R f dt
            for (int i = 0; i < count; i++) {
                const double* r = imu + 7 * i;
                const double dt = r[0] - t_last;
                t_last = r[0];
                if (dt <= 0 || dt > 0.1) continue;
                const V3 w{r[4] - bg.x, r[5] - bg.y, r[6] - bg.z};
                const V3 f{r[1] - ba.x, r[2] - ba.y, r[3] - ba.z};
                w_last = w;
                const V3 Rf = R * f;
                const V3 a = Rf + g;
                SR = SR + R * dt; Sa = Sa + Rf * dt;
                p = p + v * dt + a * (0.5 * dt * dt);
                v = v + a * dt;
                R = R * expso3(w * dt);
                T += dt;
            }
            if (T > 0) {
                th_bg = SR * (-1.0);
                v_th = skew(Sa) * (-1.0);
                v_ba = SR * (-1.0);
                v_g = M3::eye() * T;
                v_bg = v_th * th_bg * 0.5;
                p_v = M3::eye() * T;
                p_th = v_th * (0.5 * T);
                p_ba = v_ba * (0.5 * T);
                p_g = M3::eye() * (0.5 * T * T);
                p_bg = v_bg * (T / 3.0);
            }
        } else
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
        static preal tmp[9][NMAX];  // new rows of the th, p, v blocks: (Phi P)[0:9, :]
        for (int i = 0; i < 9; i++) for (int j = 0; j < n; j++) tmp[i][j] = P[i][j];
        for (int q = 0; q < NB; q++) {
            const M3& M = *blks[q].M;
            const int r0 = 3 * blks[q].r, c0 = 3 * blks[q].c;
            for (int i = 0; i < 3; i++) {
                const preal m0 = (preal)M.m[i][0], m1 = (preal)M.m[i][1], m2 = (preal)M.m[i][2];
                preal* out = tmp[r0 + i];
                axpyx(out, P[c0], m0, n); axpyx(out, P[c0 + 1], m1, n); axpyx(out, P[c0 + 2], m2, n);
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
        const int s = C0() + 6 * slot;
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
    int feature_rows(Feature& f, int row, bool want_slam = false) {
        V3 Pw; double abr[5]; M3 Ra;
        st_cand++;
        bool slam_mode = false;
        // well-observed points need no depth prior: project the whole point out (classic MSCKF); poorly observed ones
        // (little parallax) keep the inverse-depth prior
        PROF_BEGIN();
        bool prior = P_RHO_SIGMA > 0;
        const bool v_known = P[6][6] + P[7][7] + P[8][8] < 3 * P_SLAM_VSIG * P_SLAM_VSIG;
        if (want_slam && P_SLAM > 0 && nslam < P_SLAM && v_known && triangulate(f, Pw, abr, &Ra, false) && abr[3] < P_SLAM_REL * abr[2]) {
            prior = false; slam_mode = true;
        } else if (prior && P_PRIOR_SWITCH > 0 && triangulate(f, Pw, abr, &Ra, false) && abr[3] < P_PRIOR_SWITCH * abr[2]) {
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
        // point Jacobian in (al, be, rho); a new SLAM point keeps world xyz
        if (!slam_mode) for (int i = 0; i < m; i++) {
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
        double Rf[3][3], Hx1[3][6 * MAX_OBS], r1[3];
        if (slam_mode) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) Rf[i][j] = j >= i ? Hf[i][j] : 0;
                for (int c = 0; c < nc; c++) Hx1[i][c] = Hx[i][c];
                r1[i] = r[i];
            }
        }
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
            for (int c = 0; c < nc; c++) gidx[c] = C0() + 6 * slot[c / 6] + c % 6;
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
        if (slam_mode) {
            // the point joins the state: dPw = Rf^-1 (r1 - Hx1 dx - n1)  =>  J = -Rf^-1 Hx1, cov = J P J^T + sig^2 Rf^-1 Rf^-T
            if (std::fabs(Rf[0][0]) < 1e-9 || std::fabs(Rf[1][1]) < 1e-9 || std::fabs(Rf[2][2]) < 1e-9) return 0;
            double Ri[3][3] = {};  // Rf^-1 (upper triangular)
            for (int c = 0; c < 3; c++) {
                for (int i = 2; i >= 0; i--) {
                    double s2 = (i == c) ? 1.0 : 0.0;
                    for (int k = i + 1; k < 3; k++) s2 -= Rf[i][k] * Ri[k][c];
                    Ri[i][c] = s2 / Rf[i][i];
                }
            }
            double J[3][6 * MAX_OBS], df[3];
            for (int i = 0; i < 3; i++) {
                df[i] = Ri[i][0] * r1[0] + Ri[i][1] * r1[1] + Ri[i][2] * r1[2];
                for (int c = 0; c < nc; c++) J[i][c] = -(Ri[i][0] * Hx1[0][c] + Ri[i][1] * Hx1[1][c] + Ri[i][2] * Hx1[2][c]);
            }
            int gidx[6 * MAX_OBS];
            for (int c = 0; c < nc; c++) gidx[c] = C0() + 6 * slot[c / 6] + c % 6;
            const int nold = dim();
            static double Pfx[3][NMAX];
            for (int k = 0; k < 3; k++)
                for (int j = 0; j < nold; j++) {
                    double s2 = 0;
                    for (int c = 0; c < nc; c++) s2 += J[k][c] * P[gidx[c]][j];
                    Pfx[k][j] = s2;
                }
            double Pff[3][3];
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++) {
                    double s2 = 0;
                    for (int c = 0; c < nc; c++) s2 += Pfx[k][gidx[c]] * J[l][c];
                    double nz = 0;
                    for (int q = 0; q < 3; q++) nz += Ri[k][q] * Ri[l][q];
                    Pff[k][l] = s2 + sig * sig * nz;
                }
            const int pos = NI + 3 * nslam;
            insert_block(pos, 3);
            for (int k = 0; k < 3; k++) {
                for (int j = 0; j < nold; j++) { const int jj = j < pos ? j : j + 3; P[pos + k][jj] = P[jj][pos + k] = (preal)Pfx[k][j]; }
                for (int l = 0; l < 3; l++) P[pos + k][pos + l] = (preal)Pff[k][l];
            }
            slam_p[nslam] = Pw + V3{df[0], df[1], df[2]};
            nslam++; CHECKPSD("after slam init"); nslam--;
            slam_bad[nslam] = 0;
            f.slam = nslam++;
            st_slam_init++;
            if (!P_SLAM_ROWS) { f.has_pw = true; f.pw = Pw; return 0; }
        }
        f.has_pw = true; f.pw = Pw; f.sig_rho = abr[3];
        for (int i = 0; i < mr; i++) {
            for (int c = 0; c < 6 * (P_CLONES + 1); c++) Hs[row + i][c] = 0;
            for (int c = 0; c < nc; c++) Hs[row + i][6 * slot[c / 6] + c % 6] += Hx[i][c];
            rs[row + i] = r[i];
        }
        return mr;
    }

    void apply_correction(const double* dx) {
#ifdef P_CHECKPSD
        { double a = 0, b = 0; for (int i = 0; i < 3; i++) { a += dx[i] * dx[i]; b += dx[6 + i] * dx[6 + i]; }
          if (a > 0.01 || b > 0.25) std::fprintf(stderr, "frame %d big correction: dtheta %.3f dv %.3f nslam %d\n", frame, std::sqrt(a), std::sqrt(b), nslam); }
#endif
        R = expso3(V3{dx[0], dx[1], dx[2]}) * R;
        orthonormalize(R);
        p = p + V3{dx[3], dx[4], dx[5]};
        v = v + V3{dx[6], dx[7], dx[8]};
        bg = bg + V3{dx[9], dx[10], dx[11]};
        ba = ba + V3{dx[12], dx[13], dx[14]};
        g = g + V3{dx[15], dx[16], dx[17]};
        g = g * (GRAVITY / norm(g));
        for (int k = 0; k < nslam; k++) slam_p[k] = slam_p[k] + V3{dx[NI + 3 * k], dx[NI + 3 * k + 1], dx[NI + 3 * k + 2]};
        for (int k = 0; k < nclones; k++) {
            const double* d = dx + C0() + 6 * k;
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
#ifdef P_CHECKPSD
        { double mx = 0; int bi = -1, bj = -1; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (!(std::fabs((double)P[i][j]) < 1e6)) { bi = i; bj = j; }
          for (int i = 0; i < m; i++) for (int k = 0; k < nc; k++) mx = std::max(mx, std::fabs(Hs[i][k]));
          std::fprintf(stderr, "frame %d ekf: n %d m %d nc %d C0 %d nclones %d nslam %d bad P at %d %d, max|H| %.3g\n", frame, n, m, nc, C0(), nclones, nslam, bi, bj, mx); }
#endif
        const double sig = P_SIGMA_PX / (fx * P_DS);
        constexpr int MR = MAX_ROWS > NMAX ? MAX_ROWS : NMAX;
        static preal PHt[NMAX][MR];
        static preal Hf[MR][6 * (P_CLONES + 1)];
        static double S[MR][MR];
        static preal Lf[MR][MR];
        for (int j = 0; j < m; j++) for (int k = 0; k < nc; k++) Hf[j][k] = (preal)Hs[j][k];
        for (int i = 0; i < n; i++) {
            const preal* Pi = &P[i][C0()];
            if (!(P_EKF_SKIP & 1)) for (int j = 0; j < m; j++) PHt[i][j] = dotx(Pi, Hf[j], nc);
        }
        {
            static preal Sf[MR][MR];
            for (int i = 0; i < m; i++) {
                preal* Si = Sf[i];
                for (int j = 0; j < m; j++) Si[j] = 0;
                if (!(P_EKF_SKIP & 2)) for (int k = 0; k < nc; k++) axpyx(Si, PHt[C0() + k], Hf[i][k], m);
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
#ifdef P_CHECKPSD
        { double mn = 1e30, mx = 0; for (int a2 = 0; a2 < m; a2++) { mn = std::min(mn, S[a2][a2]); mx = std::max(mx, S[a2][a2]); }
          double zz = 0; for (int a2 = 0; a2 < m; a2++) zz += z[a2] * z[a2];
          double pm = 0; for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) pm = std::max(pm, std::fabs((double)PHt[i][j]));
          std::fprintf(stderr, "frame %d ekf: L diag %.3g..%.3g  |z|^2 %.3g  max|PHt| %.3g\n", frame, mn, mx, zz, pm); }
#endif
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
        CHECKPSD("after msckf update");
    }

    // features with >= P_MIN_OBS observations in `ids` (feature indices) are used in one update
    // EKF update of the SLAM points against the current pose (rows: 2 per tracked point, 9 nonzero columns each)
    void slam_update() {
        if (nslam == 0) return;
        const int n = dim();
        const double sig = P_SIGMA_PX / (fx * P_DS);
        constexpr int MS = 2 * (P_SLAM > 0 ? P_SLAM : 1);
        int col[MS][9]; double h[MS][9]; double r[MS]; int m = 0;
        const M3 Rt = R.t();
        const M3 A = R_ci * Rt;
        int considered = 0;
        for (int q0 = 0; q0 < nfeats; q0++) {
            const int k = (q0 + slam_rr) % nfeats;  // round-robin over the points when capped
            const Feature& f = feats[k];
            if (!f.alive || f.slam < 0) continue;
            if (considered++ >= P_SLAM_MAXUPD) break;
            const int j = f.slam;
            const V3 Pw = slam_p[j];
            const V3 Pc = A * (Pw - p) + t_ci;
            if (Pc.z < 0.1) { slam_bad[j] += 2; continue; }
            const double iz = 1.0 / Pc.z;
            double u, vv; pixel_to_norm(f.x, f.y, u, vv);
            const double ru = u - Pc.x * iz, rv = vv - Pc.y * iz;
            const double Jp[2][3] = {{iz, 0, -Pc.x * iz * iz}, {0, iz, -Pc.y * iz * iz}};
            const M3 Bth = A * skew(Pw - p);
            double hr[2][9]; int cr[9];
            for (int q = 0; q < 3; q++) { cr[q] = q; cr[3 + q] = 3 + q; cr[6 + q] = NI + 3 * j + q; }
            for (int a = 0; a < 2; a++)
                for (int q = 0; q < 3; q++) {
                    double hth = 0, hf = 0;
                    for (int t = 0; t < 3; t++) { hth += Jp[a][t] * Bth.m[t][q]; hf += Jp[a][t] * A.m[t][q]; }
                    hr[a][q] = hth; hr[a][3 + q] = -hf; hr[a][6 + q] = hf;
                }
            // chi-square gate: S = H P H^T + sig^2 I (2 x 2)
            double HP[2][9], S2[2][2];
            for (int a = 0; a < 2; a++)
                for (int q = 0; q < 9; q++) {
                    double s2 = 0;
                    for (int t = 0; t < 9; t++) s2 += hr[a][t] * P[cr[t]][cr[q]];
                    HP[a][q] = s2;
                }
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
                double s2 = 0; for (int q = 0; q < 9; q++) s2 += HP[a][q] * hr[b][q];
                S2[a][b] = s2 + (a == b ? sig * sig : 0);
            }
            const double det = S2[0][0] * S2[1][1] - S2[0][1] * S2[1][0];
            if (det <= 0) { slam_bad[j]++; continue; }
            const double chi = (S2[1][1] * ru * ru - 2 * S2[0][1] * ru * rv + S2[0][0] * rv * rv) / det;
            if (chi > P_SLAM_CHI2) { slam_bad[j]++; continue; }
#ifdef P_CHECKPSD
            std::fprintf(stderr, "frame %d slam %d: r %.4f %.4f (px %.2f) chi %.2f S %.3g %.3g Pff %.3g depth %.2f\n", frame, j, ru, rv, std::sqrt(ru*ru+rv*rv)*fx*P_DS, chi, S2[0][0], S2[1][1], (double)P[NI+3*j][NI+3*j], Pc.z);
#endif
            slam_bad[j] = 0;
            for (int a = 0; a < 2; a++) { for (int q = 0; q < 9; q++) { col[m][q] = cr[q]; h[m][q] = hr[a][q]; } r[m] = a ? rv : ru; m++; }
        }
        if (m > 0) {
            static preal PHt[NMAX][MS];
            static double S[MS][MS];
            for (int i = 0; i < n; i++)
                for (int a = 0; a < m; a++) {
                    double s2 = 0;
                    for (int q = 0; q < 9; q++) s2 += P[i][col[a][q]] * h[a][q];
                    PHt[i][a] = (preal)s2;
                }
            for (int a = 0; a < m; a++)
                for (int b = 0; b <= a; b++) {
                    double s2 = 0;
                    for (int q = 0; q < 9; q++) s2 += h[a][q] * PHt[col[a][q]][b];
                    S[a][b] = S[b][a] = s2 + (a == b ? sig * sig : 0);
                }
            for (int i = 0; i < m; i++)
                for (int j = 0; j <= i; j++) {
                    const double s2 = S[i][j] - dotd(S[i], S[j], j);
                    if (i == j) { if (s2 <= 1e-30) return; S[i][i] = std::sqrt(s2); }
                    else S[i][j] = s2 / S[j][j];
                }
            double z[MS];
            for (int a = 0; a < m; a++) z[a] = (r[a] - dotd(S[a], z, a)) / S[a][a];
            static preal Wm[NMAX][MS];
            static preal Lf[MS][MS];
            preal zf[MS];
            for (int a = 0; a < m; a++) { zf[a] = (preal)z[a]; for (int b = 0; b < m; b++) Lf[a][b] = b <= a ? (preal)S[a][b] : 0; }
            double dx[NMAX];
            for (int i = 0; i < n; i++) {
                preal* w = Wm[i];
                for (int a = 0; a < m; a++) w[a] = (PHt[i][a] - dotx(Lf[a], w, a)) / Lf[a][a];
                dx[i] = dotx(w, zf, m);
            }
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++) { const preal val = P[i][j] - dotx(Wm[i], Wm[j], m); P[i][j] = val; P[j][i] = val; }
            for (int i = 0; i < n; i++) if (P[i][i] < 1e-12) P[i][i] = 1e-12;
            apply_correction(dx);
            CHECKPSD("after slam update");
            st_slam_upd += m / 2;
        }
        slam_rr += P_SLAM_MAXUPD;
        for (int j = nslam - 1; j >= 0; j--) if (slam_bad[j] >= 2) remove_slam(j);
    }

#ifdef P_CHECKPSD
    void check_psd(const char* where) {
        const int n = dim();
        static double L[NMAX][NMAX];
        double minp = 1e30; int bad = -1;
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++) {
                double s2 = 0.5 * (P[i][j] + P[j][i]);
                for (int k = 0; k < j; k++) s2 -= L[i][k] * L[j][k];
                if (i == j) { if (s2 < minp) { minp = s2; } if (s2 <= 0) { bad = i; s2 = 1e-30; } L[i][i] = std::sqrt(s2); }
                else L[i][j] = s2 / L[j][j];
            }
        double asym = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) asym = std::max(asym, std::fabs((double)P[i][j] - (double)P[j][i]));
        if (bad >= 0 || asym > 1e-9) std::fprintf(stderr, "frame %d %s: not PD at %d (min pivot %.3g), asym %.3g, nslam %d nclones %d\n", frame, where, bad, minp, asym, nslam, nclones);
    }
#endif

    // vertical velocity pseudo-measurement: h = g^T v / |g| = 0 (rows on v and g)
    void vertical_prior(double sigma) {
        const int n = dim();
        const double gn = norm(g);
        const V3 gh = g * (1.0 / gn);
        const double z = -dot(gh, v);  // residual 0 - h
        double hcol[6]; int cidx[6];
        const V3 dg = (v - gh * dot(gh, v)) * (1.0 / gn);  // d(gh.v)/dg
        for (int q = 0; q < 3; q++) { cidx[q] = 6 + q; hcol[q] = gh[q]; cidx[3 + q] = 15 + q; hcol[3 + q] = dg[q]; }
        static preal ph[NMAX];
        double S = sigma * sigma;
        for (int i = 0; i < n; i++) { double s2 = 0; for (int q = 0; q < 6; q++) s2 += P[i][cidx[q]] * hcol[q]; ph[i] = (preal)s2; }
        for (int q = 0; q < 6; q++) S += hcol[q] * ph[cidx[q]];
        double dx[NMAX];
        for (int i = 0; i < n; i++) dx[i] = ph[i] * z / S;
        for (int i = 0; i < n; i++) axpyx(P[i], ph, (preal)(-ph[i] / S), n);
        apply_correction(dx);
    }

    void drop_dead_slam() {
        for (int k = 0; k < nfeats; k++) if (!feats[k].alive && feats[k].slam >= 0) remove_slam(feats[k].slam);
    }

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
        const double r[3] = {-v.x, -v.y, -v.z};
        // K = P[:, v] S^-1 ; P -= K P[v, :] using contiguous copies of rows 6..8 (P is symmetric)
        static preal Rv[3][NMAX];
        for (int a = 0; a < 3; a++) for (int j = 0; j < n; j++) Rv[a][j] = P[6 + a][j];
        double dx[NMAX];
        for (int i = 0; i < n; i++) {
            const double c0 = Rv[0][i], c1 = Rv[1][i], c2 = Rv[2][i];
            const double k0 = c0 * I[0][0] + c1 * I[1][0] + c2 * I[2][0];
            const double k1 = c0 * I[0][1] + c1 * I[1][1] + c2 * I[2][1];
            const double k2 = c0 * I[0][2] + c1 * I[1][2] + c2 * I[2][2];
            dx[i] = k0 * r[0] + k1 * r[1] + k2 * r[2];
            preal* Pi = P[i];
            axpyx(Pi, Rv[0], (preal)-k0, n);
            axpyx(Pi, Rv[1], (preal)-k1, n);
            axpyx(Pi, Rv[2], (preal)-k2, n);
        }
        apply_correction(dx);
    }

    int pending_m = 0, pending_nc = 0;
    int st_last_m = 0, st_last_rows = 0, st_last_ni = 0;  // split update: rows computed at a clone frame, applied at the next frame

    void update_with(const int* idx, int count, bool apply = true) {
        int rows = 0, used = 0;
        double depth_sum = 0; int depth_n = 0;
        int tried = 0, slam_new = 0;
        for (int q = 0; q < count && used < P_MAX_UPDATE_FEAT && tried < P_MAX_CAND; q++) {
            Feature& f = feats[idx[q]];
            if (f.nobs < P_MIN_OBS) continue;
            tried++;
            unsigned long long t_ = 0;
#if !defined(__wasm__) && defined(P_PROF)
            t_ = __rdtsc();
#endif
            const int before_slam = nslam;
            const int got = feature_rows(f, rows, f.alive && f.slam < 0 && slam_new < P_SLAM_INIT_MAX);
            slam_new += nslam - before_slam;
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
            f.id = next_feat_id++; f.x = (float)bx; f.y = (float)by; f.nobs = 0; f.alive = true; f.has_pw = false; f.slam = -1;
        }
        detect_cell = (detect_cell + std::max(searched, 1)) % (P_GRID_X * P_GRID_Y);
    }

    // ---------------------------------------------------------------- visual-inertial initialization
    // Linear closed form (Martinelli-style): over the first frames, camera positions are p_i = v0 t_i + g t_i^2 / 2 +
    // dp_i (dp_i: double integral of the rotated specific force, gyro attitude R_i). Each bearing d_ij of point P_j gives
    // [d_ij]x (P_j - p_i - R_i t_ic) = 0, linear in (P_j, v0, g); points are eliminated (Schur), g is then fixed to 9.81 m/s^2.
    static constexpr int VI_MAXF = P_VI_MAX + 1;
    static constexpr int VI_MAXO = VI_MAXF * P_MAX_FEAT;
    bool vi_active = P_VI_INIT > 0;
    int vi_nf = 0, vi_no = 0;
    double vi_t[VI_MAXF]; M3 vi_R[VI_MAXF]; V3 vi_dp[VI_MAXF], vi_dv[VI_MAXF];
    struct VIObs { int id, fi; float u, v; };
    VIObs vi_obs[VI_MAXO];
    V3 vi_dv_acc{0, 0, 0}, vi_dp_acc{0, 0, 0};
    double vi_t0 = 0;
    int vi_status = 0;  // diagnostics: 1 solved, -1 gave up

    void vi_accumulate(const double* imu, int count, const M3& R_start) {  // integrals of R f over this frame's samples
        M3 Ri = R_start;
        double tl = vi_t0 + vi_t[vi_nf - 1];
        for (int i = 0; i < count; i++) {
            const double* r = imu + 7 * i;
            const double dt = r[0] - tl;
            tl = r[0];
            if (dt <= 0 || dt > 0.1) continue;
            const V3 a = Ri * V3{r[1], r[2], r[3]};
            vi_dp_acc = vi_dp_acc + vi_dv_acc * dt + a * (0.5 * dt * dt);
            vi_dv_acc = vi_dv_acc + a * dt;
            Ri = Ri * expso3(V3{r[4] - bg.x, r[5] - bg.y, r[6] - bg.z} * dt);
        }
    }

    void vi_record(double t) {
        vi_t[vi_nf] = t - vi_t0; vi_R[vi_nf] = R; vi_dp[vi_nf] = vi_dp_acc; vi_dv[vi_nf] = vi_dv_acc;
        for (int k = 0; k < nfeats && vi_no < VI_MAXO; k++)
            if (feats[k].alive) {
                double u, vv; pixel_to_norm(feats[k].x, feats[k].y, u, vv);
                vi_obs[vi_no++] = {feats[k].id, vi_nf, (float)u, (float)vv};
            }
        vi_nf++;
    }

    static void inv3(const double A[3][3], double B[3][3]) {
        const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        const double id = std::fabs(det) > 1e-300 ? 1.0 / det : 0.0;
        B[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * id; B[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * id; B[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * id;
        B[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * id; B[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * id; B[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * id;
        B[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * id; B[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * id; B[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * id;
    }

    // returns true and the initial velocity (IMU-0 frame) and gravity when the solve is usable
    bool vi_solve(V3& v0, V3& g0) {
        double Hxx[6][6] = {}, bx[6] = {};
        int used = 0;
        // features: group observations by id (obs are stored frame by frame)
        static int done_id[VI_MAXO]; int ndone = 0;
        for (int o = 0; o < vi_no; o++) {
            const int id = vi_obs[o].id;
            bool seen = false;
            for (int q = 0; q < ndone; q++) if (done_id[q] == id) { seen = true; break; }
            if (seen) continue;
            done_id[ndone++] = id;
            double Hjj[3][3] = {}, Hjx[3][6] = {}, bj[3] = {};
            double Hxx_j[6][6] = {}, bx_j[6] = {};
            int nobs = 0, first = -1, last = -1;
            for (int q = o; q < vi_no; q++) {
                if (vi_obs[q].id != id) continue;
                const int i = vi_obs[q].fi;
                if (first < 0) first = i;
                last = i;
                V3 d = vi_R[i] * (R_ic * V3{vi_obs[q].u, vi_obs[q].v, 1.0});
                d = d * (1.0 / norm(d));
                const M3 D = skew(d);
                const V3 c = vi_dp[i] + vi_R[i] * t_ic;
                const double ti = vi_t[i];
                // e = D P + B x - y,  B = -D [ti I | ti^2/2 I], y = D c
                double Bm[3][6];
                for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) { Bm[a][b] = -D.m[a][b] * ti; Bm[a][3 + b] = -D.m[a][b] * 0.5 * ti * ti; }
                const V3 y = D * c;
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) { double s2 = 0; for (int k = 0; k < 3; k++) s2 += D.m[k][a] * D.m[k][b]; Hjj[a][b] += s2; }
                    for (int b = 0; b < 6; b++) { double s2 = 0; for (int k = 0; k < 3; k++) s2 += D.m[k][a] * Bm[k][b]; Hjx[a][b] += s2; }
                    double s2 = 0; for (int k = 0; k < 3; k++) s2 += D.m[k][a] * y[k]; bj[a] += s2;
                }
                for (int a = 0; a < 6; a++) {
                    for (int b = 0; b < 6; b++) { double s2 = 0; for (int k = 0; k < 3; k++) s2 += Bm[k][a] * Bm[k][b]; Hxx_j[a][b] += s2; }
                    double s2 = 0; for (int k = 0; k < 3; k++) s2 += Bm[k][a] * y[k]; bx_j[a] += s2;
                }
                nobs++;
            }
            if (nobs < 3 || last - first < P_VI_MINSPAN) continue;
            double Hi[3][3]; inv3(Hjj, Hi);
            // Schur: Hxx -= Hjx^T Hjj^-1 Hjx ; bx -= Hjx^T Hjj^-1 bj   (sign: normal equations of e = D P + B x - y)
            double T1[3][6];
            for (int a = 0; a < 3; a++) for (int b = 0; b < 6; b++) { double s2 = 0; for (int k = 0; k < 3; k++) s2 += Hi[a][k] * Hjx[k][b]; T1[a][b] = s2; }
            double t2[3];
            for (int a = 0; a < 3; a++) { double s2 = 0; for (int k = 0; k < 3; k++) s2 += Hi[a][k] * bj[k]; t2[a] = s2; }
            for (int a = 0; a < 6; a++) {
                for (int b = 0; b < 6; b++) { double s2 = 0; for (int k = 0; k < 3; k++) s2 += Hjx[k][a] * T1[k][b]; Hxx[a][b] += Hxx_j[a][b] - s2; }
                double s2 = 0; for (int k = 0; k < 3; k++) s2 += Hjx[k][a] * t2[k]; bx[a] += bx_j[a] - s2;
            }
            used++;
        }
        if (used < P_VI_MINFEAT) return false;
        // normal equations: Hxx x = bx  (from min |D P + B x - y|^2 after eliminating P)
        // full solve for g direction, then fix |g| and solve v0
        double A[6][7];
        for (int a = 0; a < 6; a++) { for (int b = 0; b < 6; b++) A[a][b] = Hxx[a][b]; A[a][6] = bx[a]; }
        for (int c = 0; c < 6; c++) {
            int piv = c;
            for (int r = c + 1; r < 6; r++) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
            if (std::fabs(A[piv][c]) < 1e-12) return false;
            for (int k = 0; k < 7; k++) std::swap(A[c][k], A[piv][k]);
            for (int r = 0; r < 6; r++) if (r != c) { const double f2 = A[r][c] / A[c][c]; for (int k = c; k < 7; k++) A[r][k] -= f2 * A[c][k]; }
        }
        double x[6];
        for (int a = 0; a < 6; a++) x[a] = A[a][6] / A[a][a];
        V3 gs{x[3], x[4], x[5]};
        const double gn = norm(gs);
        if (!(gn > 0.5 * GRAVITY && gn < 1.5 * GRAVITY)) return false;
        g0 = gs * (GRAVITY / gn);
        // v0 with g fixed: Hvv v = bv - Hvg g
        double Hv[3][3], rhs[3];
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) Hv[a][b] = Hxx[a][b];
            rhs[a] = bx[a] - (Hxx[a][3] * g0.x + Hxx[a][4] * g0.y + Hxx[a][5] * g0.z);
        }
        double Hvi[3][3]; inv3(Hv, Hvi);
        v0 = V3{Hvi[0][0] * rhs[0] + Hvi[0][1] * rhs[1] + Hvi[0][2] * rhs[2], Hvi[1][0] * rhs[0] + Hvi[1][1] * rhs[1] + Hvi[1][2] * rhs[2],
                Hvi[2][0] * rhs[0] + Hvi[2][1] * rhs[1] + Hvi[2][2] * rhs[2]};
        return std::isfinite(v0.x + v0.y + v0.z) && norm(v0) < P_MAX_SPEED;
    }

    void vi_finish(const V3& v0, const V3& g0) {
        const int i = vi_nf - 1;
        const double ti = vi_t[i];
        g = g0;
        v = v0 + g0 * ti + vi_dv[i];
        p = v0 * ti + g0 * (0.5 * ti * ti) + vi_dp[i];
        nclones = 0; nslam = 0; pending_m = 0;
        for (int k = 0; k < nfeats; k++) { feats[k].nobs = 0; feats[k].slam = -1; }
        std::memset(P, 0, sizeof(P));
        const double sg = GRAVITY * P_VI_TILT;
        const V3 gh = g * (1.0 / GRAVITY);
        for (int a = 0; a < 3; a++) {
            P[a][a] = 1e-6;
            P[3 + a][3 + a] = 1e-4;
            for (int b = 0; b < 3; b++) P[15 + a][15 + b] = sg * sg * ((a == b) - gh[a] * gh[b]) + 0.01 * gh[a] * gh[b];
            P[6 + a][6 + a] = P_VI_VSIG * P_VI_VSIG;
            P[9 + a][9 + a] = P_INIT_BG * P_INIT_BG;
            P[12 + a][12 + a] = P_INIT_BA * P_INIT_BA;
        }
        add_clone();
        record_obs();
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
            vi_t0 = t;
            add_clone();
#ifndef __wasm__
            if (ext_n >= 0) external_features(); else
#endif
            detect(rgb, P_DETECT_CELLS_FIRST);
            record_obs();
            if (vi_active) vi_record(t);
        } else if (vi_active) {
            // initialization window: gyro attitude, front end, IMU integrals; no filter updates
            const M3 R_prev = R; const V3 p_prev = p;
            vi_accumulate(imu, count, R);
            propagate(imu, count);
#ifndef __wasm__
            if (ext_n >= 0) external_features(); else
#endif
            track(rgb, R_prev, p_prev);
            int w = 0;
            for (int k = 0; k < nfeats; k++) if (feats[k].alive) feats[w++] = feats[k];
            nfeats = w;
#ifndef __wasm__
            if (ext_n < 0)
#endif
            detect(rgb, P_DETECT_CELLS);
            vi_record(t);
            if (vi_nf > P_VI_INIT && (vi_nf - 1) % P_VI_RETRY == 0) {
                V3 v0, g0;
                if (vi_solve(v0, g0)) {
#ifndef __wasm__
                    if (getenv("VI_DEBUG")) std::fprintf(stderr, "vi solve at %d: v0 %.3f %.3f %.3f  g %.2f %.2f %.2f  (gravity before %.2f %.2f %.2f)\n", vi_nf, v0.x, v0.y, v0.z, g0.x, g0.y, g0.z, g.x, g.y, g.z);
#endif
                    vi_finish(v0, g0); vi_active = false; vi_status = 1;
                }
                else if (vi_nf >= P_VI_MAX) {  // give up: continue from the IMU-only state with a wide velocity prior
                    vi_active = false; vi_status = -1;
                    nclones = 0; for (int k = 0; k < nfeats; k++) feats[k].nobs = 0;
                    v = V3{0, 0, 0};
                    for (int a = 0; a < 3; a++) { for (int b = 0; b < NMAX; b++) { P[6 + a][b] = P[b][6 + a] = 0; } P[6 + a][6 + a] = P_INIT_VEL * P_INIT_VEL; }
                    add_clone(); record_obs();
                }
            }
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
            if (P_VZ_SIGMA > 0 && (light || !P_BALANCE)) vertical_prior(P_BALANCE && P_SPLIT ? P_VZ_SIGMA / std::sqrt((double)P_CLONE_EVERY) : P_VZ_SIGMA);
            twice(1, [&] { track(rgb, R_prev, p_prev); });
#ifndef __wasm__
            if (ext_n >= 0) external_features(); else
#endif
            { PROF_BEGIN(); if (!(P_SKIP & 1)) track(rgb, R_prev, p_prev); PROF_END(1); }
            if (P_SLAM > 0) { drop_dead_slam(); if (light && frame % P_SLAM_EVERY == 1 % P_SLAM_EVERY) slam_update(); }
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
#ifndef __wasm__
            if (ext_n < 0)
#endif
            PROF_BEGIN(); if (!(P_SKIP & 2) && light) detect(rgb, P_BALANCE && P_SPLIT ? P_DETECT_CELLS * P_CLONE_EVERY : P_DETECT_CELLS); PROF_END(2);
            if (clone_frame) record_obs_new();
        }
        frame++;
        if (P_WATCHDOG) watchdog();
        write_output(out);
    }

#ifndef __wasm__
    // host test hook: features = the external list (same id = same point); missing ids die, new ids are born
    void external_features() {
        for (int k = 0; k < nfeats; k++) {
            int q = -1;
            for (int e2 = 0; e2 < ext_n; e2++) if (ext_id[e2] == feats[k].id) { q = e2; break; }
            if (q < 0) { feats[k].alive = false; st_lost++; }
            else if (feats[k].alive) { feats[k].x = ext_x[q]; feats[k].y = ext_y[q]; st_tracked++; }
        }
        for (int e2 = 0; e2 < ext_n && nfeats < P_MAX_FEAT; e2++) {
            bool have = false;
            for (int k = 0; k < nfeats; k++) if (feats[k].id == ext_id[e2]) { have = true; break; }
            if (have) continue;
            Feature& f = feats[nfeats++];
            f.id = ext_id[e2]; f.x = ext_x[e2]; f.y = ext_y[e2]; f.nobs = 0; f.alive = true; f.has_pw = false; f.slam = -1; f.tpl.ok = true;
        }
    }
#endif

    void record_obs() {
        for (int k = 0; k < nfeats; k++) if (feats[k].alive) push_obs(feats[k]);
    }
    void record_obs_new() {  // newly detected features get their first observation at this clone
        const int id = clones[nclones - 1].id;
        for (int k = 0; k < nfeats; k++)
            if (feats[k].alive && (feats[k].nobs == 0 || feats[k].obs[feats[k].nobs - 1].clone_id != id)) push_obs(feats[k]);
    }
    void push_obs(Feature& f) {
        if (f.slam >= 0) return;
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

    // last good pose; a non-finite or absurd state restarts the filter from it (velocity zero, fresh covariance)
    M3 good_R = M3::eye(); V3 good_p{0, 0, 0}; V3 good_g{0, 0, -GRAVITY}; V3 good_bg{0, 0, 0};
    int resets = 0;
    void watchdog() {
        bool ok = std::isfinite(p.x + p.y + p.z + v.x + v.y + v.z) && std::isfinite(R.m[0][0] + R.m[1][1] + R.m[2][2]) && norm(v) < 0.99 * P_MAX_SPEED;
        for (int i = 0; i < NI && ok; i++) ok = std::isfinite((double)P[i][i]) && P[i][i] < 1e4;
        if (ok) { good_R = R; good_p = p; good_g = g; good_bg = bg; return; }
        resets++;
        R = good_R; p = good_p; v = V3{0, 0, 0}; g = good_g; bg = good_bg; ba = V3{0, 0, 0};
        nclones = 0; nslam = 0; pending_m = 0;
        for (int k = 0; k < nfeats; k++) { feats[k].nobs = 0; feats[k].slam = -1; }
        std::memset(P, 0, sizeof(P));
        const double sg = GRAVITY * P_INIT_TILT;
        const V3 gh = g * (1.0 / norm(g));
        for (int i = 0; i < 3; i++) {
            P[i][i] = 0.01;  // attitude: the restart point is uncertain
            P[3 + i][3 + i] = 0.25;
            for (int j = 0; j < 3; j++) P[15 + i][15 + j] = sg * sg * ((i == j) - gh[i] * gh[j]) + 0.01 * gh[i] * gh[j];
            P[6 + i][6 + i] = P_INIT_VEL * P_INIT_VEL;
            P[9 + i][9 + i] = P_INIT_BG * P_INIT_BG;
            P[12 + i][12 + i] = P_INIT_BA * P_INIT_BA;
        }
        add_clone();
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
        const V3 pc = vi_active ? V3{0, 0, 0} : R_ci * (R * t_ic + p) + t_ci;
        double q[4]; quaternion(Rc, q);
        out[0] = pc.x; out[1] = pc.y; out[2] = pc.z;
        out[3] = q[0]; out[4] = q[1]; out[5] = q[2]; out[6] = q[3];
        // camera centre velocity in world, then in camera-k axes
        const V3 vc_w = (vi_active ? V3{0, 0, 0} : v) + R * cross(w_last, t_ic);
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
int32_t debug_set_features(int32_t n, const int32_t* ids, const double* xy) {  // gray-resolution pixels
    ext_n = n;
    for (int i = 0; i < n && i < 256; i++) { ext_id[i] = ids[i]; ext_x[i] = (float)xy[2 * i]; ext_y[i] = (float)xy[2 * i + 1]; }
    return 0;
}
int32_t debug_set_velocity(const double* vw, double sigma) {  // host oracle tests: velocity in the IMU-0 frame
    Estimator& e = *estimator;
    e.v = V3{vw[0], vw[1], vw[2]};
    for (int i = 0; i < 3; i++) { for (int j = 0; j < NMAX; j++) { P[6 + i][j] = 0; P[j][6 + i] = 0; } P[6 + i][6 + i] = sigma * sigma; }
    return 0;
}
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
                     std::sqrt(P[6][6]), std::sqrt(P[7][7]), std::sqrt(P[8][8]), std::sqrt(P[15][15]), (double)e.st_last_m, (double)e.st_last_rows, (double)e.st_last_ni, (double)e.nslam, (double)e.st_slam_init, (double)e.st_slam_upd, (double)e.vi_status, (double)e.vi_nf};
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
