// Numerical check of the propagation Jacobian (the Phi used for the covariance).
#include "../solution.cpp"
#include <cstdio>
V3 logso3(const M3& R) {
    const double c = std::max(-1.0, std::min(1.0, 0.5 * (R.m[0][0] + R.m[1][1] + R.m[2][2] - 1)));
    const double th = std::acos(c);
    const double s = th < 1e-9 ? 0.5 : th / (2 * std::sin(th));
    return V3{R.m[2][1] - R.m[1][2], R.m[0][2] - R.m[2][0], R.m[1][0] - R.m[0][1]} * s;
}
void state_vec(const Estimator& a, const Estimator& b, double* d) {  // b - a in error coordinates
    V3 th = logso3(b.R * a.R.t());
    V3 vals[6] = {th, b.p - a.p, b.v - a.v, b.bg - a.bg, b.ba - a.ba, b.g - a.g};
    for (int i = 0; i < 6; i++) { d[3 * i] = vals[i].x; d[3 * i + 1] = vals[i].y; d[3 * i + 2] = vals[i].z; }
}
int main() {
    double imu[70];
    for (int i = 0; i < 10; i++) {
        double* r = imu + 7 * i;
        r[0] = 0.005 * (i + 1); r[1] = 1.0 + 0.1 * i; r[2] = -0.5; r[3] = 9.5; r[4] = 0.3; r[5] = -0.2 + 0.05 * i; r[6] = 0.8;
    }
    Estimator base{};
    base.R = expso3(V3{0.1, -0.2, 0.3}); base.v = V3{0.5, -0.3, 0.1}; base.bg = V3{0.01, 0.02, -0.01}; base.ba = V3{0.1, -0.1, 0.05};
    base.g = V3{0.3, -0.2, -9.8}; base.t_last = 0;
    std::memset(P, 0, sizeof(P));
    // analytic Phi: propagate P = I (only IMU block), no noise -> P' = Phi Phi^T; instead recover Phi columns by P = e_j e_j^T? use finite differences on the mean and compare with P' = Phi I Phi^T diag
    Estimator a = base; a.propagate(imu, 10);
    const double eps = 1e-6;
    double num[NI][NI];
    for (int j = 0; j < NI; j++) {
        Estimator b = base;
        double d[NI] = {}; d[j] = eps;
        b.R = expso3(V3{d[0], d[1], d[2]}) * b.R; b.p = b.p + V3{d[3], d[4], d[5]}; b.v = b.v + V3{d[6], d[7], d[8]};
        b.bg = b.bg + V3{d[9], d[10], d[11]}; b.ba = b.ba + V3{d[12], d[13], d[14]}; b.g = b.g + V3{d[15], d[16], d[17]};
        b.propagate(imu, 10);
        double out[NI]; state_vec(a, b, out);
        for (int i = 0; i < NI; i++) num[i][j] = out[i] / eps;
    }
    // analytic: run propagate with P = I and zero noise is not separable; recompute Phi by calling with P = E_jj and reading P*?  simpler: P = I => P' = Phi Phi^T + Q. compare with num num^T + Q
    std::memset(P, 0, sizeof(P));
    for (int i = 0; i < NI; i++) P[i][i] = 1;
    Estimator c = base; c.propagate(imu, 10);
    double maxe = 0;
    for (int i = 0; i < NI; i++) for (int j = 0; j < NI; j++) {
        double s = 0; for (int k = 0; k < NI; k++) s += num[i][k] * num[j][k];
        double q = 0;
        if (i == j) { const double T = 0.05; if (i < 3) q = P_GYRO_NOISE * P_GYRO_NOISE * T; else if (i < 6) q = P_ACC_NOISE * P_ACC_NOISE * T * T * T / 3; else if (i < 9) q = P_ACC_NOISE * P_ACC_NOISE * T; else if (i < 12) q = P_GYRO_WALK * P_GYRO_WALK * T; else if (i < 15) q = P_ACC_WALK * P_ACC_WALK * T; }
        const double e = std::fabs(P[i][j] - s - q);
        if (e > 1e-4) std::printf("P[%d][%d] analytic %.6f numeric %.6f\n", i, j, P[i][j] - q, s);
        maxe = std::max(maxe, e);
    }
    std::printf("max err %.2e\n", maxe);
    for (int i = 0; i < NI; i++) { for (int j = 0; j < NI; j++) std::printf("%7.4f ", num[i][j]); std::printf("\n"); }
}
