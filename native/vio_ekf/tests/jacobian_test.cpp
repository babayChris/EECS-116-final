// Numerical check of the projected MSCKF rows: g++ -O1 -I../../viobench/viobench/solution/runtime/native -I. tests/jacobian_test.cpp
#include "../solution.cpp"
#include <cstdio>
int main() {
    Estimator& e = *new Estimator();
    e.fx = e.fy = 100; e.cx = 80; e.cy = 60;
    double T[16] = {0, -1, 0, 0.02, 0, 0, -1, 0.01, 1, 0, 0, -0.03, 0, 0, 0, 1};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) e.R_ci.m[i][j] = T[4 * i + j];
    e.t_ci = {T[3], T[7], T[11]}; e.R_ic = e.R_ci.t(); e.t_ic = (e.R_ic * e.t_ci) * -1.0;
    std::memset(P, 0, sizeof(P));
    for (int i = 0; i < NMAX; i++) P[i][i] = 1e-4;
    // 4 clones moving along +y with some rotation; point 3 m ahead (IMU x forward)
    e.nclones = 4;
    for (int k = 0; k < 4; k++) {
        e.clones[k].id = k;
        e.clones[k].R = expso3(V3{0.02 * k, -0.01 * k, 0.05 * k});
        e.clones[k].p = V3{0.1 * k, 0.2 * k, 0.03 * k};
    }
    const V3 Pw{3.0, 0.4, -0.2};
    Feature f{}; f.nobs = 4;
    for (int k = 0; k < 4; k++) {
        const V3 Pc = e.R_ci * (e.clones[k].R.t() * (Pw - e.clones[k].p)) + e.t_ci;
        f.obs[k] = {k, Pc.x / Pc.z, Pc.y / Pc.z};
    }
    int m = e.feature_rows(f, 0);
    std::printf("rows %d, residual", m);
    double H0[8][NMAX], r0[8];
    for (int i = 0; i < m; i++) { r0[i] = e.rs[i]; std::printf(" %.2e", r0[i]); for (int c = 0; c < NMAX; c++) H0[i][c] = e.Hs[i][c]; }
    std::printf("\n");
    const double eps = 1e-6;
    double maxerr = 0, maxval = 0;
    for (int k = 0; k < 4; k++)
        for (int j = 0; j < 6; j++) {
            Estimator saved = e;
            if (j < 3) { V3 d{0, 0, 0}; double* dd = &d.x; dd[j] = eps; e.clones[k].R = expso3(d) * e.clones[k].R; }
            else { double* pp = &e.clones[k].p.x; pp[j - 3] += eps; }
            int m2 = e.feature_rows(f, 0);
            for (int i = 0; i < m2 && i < m; i++) {
                const double num = -(e.rs[i] - r0[i]) / eps;  // r = z - h  =>  dr = -H dx
                const double ana = H0[i][NI + 6 * k + j];
                maxerr = std::max(maxerr, std::fabs(num - ana)); maxval = std::max(maxval, std::fabs(ana));
                if (std::fabs(num - ana) > 1e-3 * (1 + std::fabs(ana))) std::printf("clone %d dof %d row %d: num %.5f ana %.5f\n", k, j, i, num, ana);
            }
            e = saved;
        }
    std::printf("max |num-ana| %.2e (max |H| %.2e)\n", maxerr, maxval);
}
