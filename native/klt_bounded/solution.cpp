// C++ port of solutions/klt_bounded/solution.py (persistent KLT tracks, gyro-aided known-rotation translation
// direction, constant-speed fusion, multi-frame baseline). Derived from the VIOBench klt_imu example (MIT).
// The Python version's per-frame floor (about 4.5 M fuel for the interpreter, IMU loop and output) goes away here.
// Settings: params.h. Build: viobench bundle <this directory> -o klt_bounded.wasm
#include "viobench_abi.h"
#include "params.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double GRAVITY = 9.81;
constexpr double CALIBRATION_SECONDS = 0.5;
constexpr double REST_ACCEL_STD = 1.0;
constexpr double REST_GYRO_STD = 0.2;

using Vec3 = cv::Vec3d;
using Mat3 = cv::Matx33d;
using Mat4 = cv::Matx44d;

Mat3 rodrigues(const Vec3& w) {  // exp of the skew matrix of w
    const double angle = cv::norm(w);
    const Mat3 S(0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0);
    double a, b;
    if (angle < 1e-12) {
        a = 1.0;
        b = 0.5;
    } else {
        a = std::sin(angle) / angle;
        b = (1.0 - std::cos(angle)) / (angle * angle);
    }
    return Mat3::eye() + a * S + b * (S * S);
}

Mat3 rotation(const Mat4& T) {
    return Mat3(T(0, 0), T(0, 1), T(0, 2), T(1, 0), T(1, 1), T(1, 2), T(2, 0), T(2, 1), T(2, 2));
}

Vec3 translation(const Mat4& T) { return Vec3(T(0, 3), T(1, 3), T(2, 3)); }

Mat4 compose(const Mat3& R, const Vec3& t) {
    Mat4 T = Mat4::eye();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) T(i, j) = R(i, j);
        T(i, 3) = t[i];
    }
    return T;
}

Mat4 rigid_inverse(const Mat4& T) {
    const Mat3 Rt = rotation(T).t();
    return compose(Rt, -(Rt * translation(T)));
}

std::array<double, 4> quaternion(const Mat3& R) {  // [qx, qy, qz, qw]
    std::array<double, 4> q{};
    const double trace = R(0, 0) + R(1, 1) + R(2, 2);
    if (trace > 0) {
        const double s = std::sqrt(trace + 1.0) * 2;
        q = {(R(2, 1) - R(1, 2)) / s, (R(0, 2) - R(2, 0)) / s, (R(1, 0) - R(0, 1)) / s, 0.25 * s};
    } else {
        int i = 0;
        if (R(1, 1) > R(i, i)) i = 1;
        if (R(2, 2) > R(i, i)) i = 2;
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        const double s = std::sqrt(1.0 + R(i, i) - R(j, j) - R(k, k)) * 2;
        q[i] = 0.25 * s;
        q[j] = (R(j, i) + R(i, j)) / s;
        q[k] = (R(k, i) + R(i, k)) / s;
        q[3] = (R(k, j) - R(j, k)) / s;
    }
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (auto& v : q) v /= n;
    return q;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t n = v.size(), m = n / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    if (n % 2) return v[m];
    const double hi = v[m];
    return 0.5 * (hi + *std::max_element(v.begin(), v.begin() + m));
}

struct Frame {
    std::vector<long> ids;
    std::vector<cv::Point2f> points;
    Mat3 R_wc;
    double t = 0;
};

class Estimator {
public:
    Estimator(const cv::Matx33d& K, const Mat4& T_cam_imu) : K_(K), T_cam_imu_(T_cam_imu), rng_(0) {
        T_imu_cam_ = rigid_inverse(T_cam_imu);
        R_bc_ = rotation(T_imu_cam_);
    }

    bool initialized() const { return initialized_; }

    // One frame: gray image at the working scale and the IMU rows since the last frame. Writes the 13 output values.
    void step(double t, const cv::Mat& frame, const double* imu, int n, double* out) {
        if (!initialized_) {
            for (int i = 0; i < n; i++) calibration_.emplace_back(imu + 7 * i, imu + 7 * i + 7);
            if (calibration_.size() < 2 || calibration_.back()[0] - calibration_.front()[0] < CALIBRATION_SECONDS) {
                std::fill(out, out + 13, 0.0);
                out[6] = 1.0;
                return;
            }
            initialize(n ? imu[7 * (n - 1)] : t);
            frame.copyTo(previous_frame_);
            detect(frame, previous_points_);
            write_state(out);
            return;
        }
        const Mat3 R_wb_before = R_wb_;
        const Vec3 displacement_imu = integrate(imu, n);
        const Mat3 R_c1_c0 = (R_wb_ * R_bc_).t() * (R_wb_before * R_bc_);
        Vec3 direction_camera;
        const bool have_direction = track(frame, R_c1_c0, direction_camera);
        double dt = 0.05;
        if (n) dt = std::max(imu[7 * (n - 1)] - imu[0] + (n > 1 ? imu[7] - imu[0] : 0.0), 1e-3);
        if (have_direction) {
            Vec3 direction_world = (R_wb_before * R_bc_) * direction_camera;
            direction_world *= 1.0 / std::max(cv::norm(direction_world), 1e-12);
            if (P_BASELINE_FRAMES > 1) {
                Vec3 longer;
                if (direction_multiframe(longer)) direction_world = longer;
            }
            last_direction_ = direction_world;
            has_last_direction_ = true;
            const Vec3 displacement = P_SPEED_PRIOR * dt * direction_world;
            p_ += displacement - displacement_imu;
            v_ = displacement * (1.0 / dt);
        } else {
            p_ -= displacement_imu;
            if (has_last_direction_) {  // coast: keep the last velocity
                p_ += P_SPEED_PRIOR * dt * last_direction_;
                v_ = P_SPEED_PRIOR * last_direction_;
            } else {
                v_ = Vec3(0, 0, 0);
            }
        }
        if (P_BIAS_BASELINE > 1) update_gyro_bias();
        frame.copyTo(previous_frame_);
        replenish(frame);
        if (P_TRACE) std::fprintf(stderr, "F %d dir %d kept %d prev %d\n", frame_index_++, (int)have_direction, trace_kept_, (int)previous_points_.size());
        const int keep = std::max(P_BASELINE_FRAMES, P_BIAS_BASELINE);
        if (keep > 1) {
            history_.push_back(Frame{ids_, previous_points_, R_wb_ * R_bc_, last_t_});
            while ((int)history_.size() > keep) history_.pop_front();
        }
        write_state(out);
    }

    void reseed() { rng_.seed(0); }

private:
    void initialize(double t) {
        const size_t m = calibration_.size();
        Vec3 mean_a(0, 0, 0), mean_g(0, 0, 0), var_a(0, 0, 0), var_g(0, 0, 0);
        for (const auto& r : calibration_) {
            mean_a += Vec3(r[1], r[2], r[3]);
            mean_g += Vec3(r[4], r[5], r[6]);
        }
        mean_a *= 1.0 / m;
        mean_g *= 1.0 / m;
        for (const auto& r : calibration_) {
            for (int k = 0; k < 3; k++) {
                var_a[k] += (r[1 + k] - mean_a[k]) * (r[1 + k] - mean_a[k]);
                var_g[k] += (r[4 + k] - mean_g[k]) * (r[4 + k] - mean_g[k]);
            }
        }
        double sa = 0, sg = 0;
        for (int k = 0; k < 3; k++) {
            sa += var_a[k] / m;
            sg += var_g[k] / m;
        }
        const bool at_rest = std::sqrt(sa) < REST_ACCEL_STD && std::sqrt(sg) < REST_GYRO_STD;
        gyro_bias_ = at_rest ? mean_g : Vec3(0, 0, 0);
        const Vec3 z = mean_a * (1.0 / cv::norm(mean_a));
        Vec3 x = Vec3(1, 0, 0) - z[0] * z;
        x *= 1.0 / cv::norm(x);
        const Vec3 y = z.cross(x);
        R_wb_ = Mat3(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]);
        last_t_ = t;
        initialized_ = true;
        T_w_c0_ = pose_camera();
    }

    Vec3 integrate(const double* imu, int n) {
        const Vec3 start = p_;
        const Vec3 g(0, 0, -GRAVITY);
        for (int i = 0; i < n; i++) {
            const double* r = imu + 7 * i;
            const Vec3 accel(r[1], r[2], r[3]), gyro = Vec3(r[4], r[5], r[6]) - gyro_bias_;
            omega_ = gyro;
            const double dt = r[0] - last_t_;
            if (dt > 0) {
                R_wb_ = R_wb_ * rodrigues(gyro * dt);
                const Vec3 a = R_wb_ * accel + g;
                p_ += v_ * dt + 0.5 * a * dt * dt;
                v_ += a * dt;
            }
            last_t_ = r[0];
        }
        return p_ - start;
    }

    Mat4 pose_camera() const { return compose(R_wb_, p_) * T_imu_cam_; }

    void write_state(double* out) const {
        const Mat4 T = rigid_inverse(T_w_c0_) * pose_camera();
        const Mat3 R_cam_imu = rotation(T_cam_imu_);
        const Vec3 r = translation(T_imu_cam_);
        const Vec3 velocity = R_cam_imu * (R_wb_.t() * v_ + omega_.cross(r));
        const Vec3 angular = R_cam_imu * omega_;
        const auto q = quaternion(rotation(T));
        for (int i = 0; i < 3; i++) out[i] = T(i, 3);
        for (int i = 0; i < 4; i++) out[3 + i] = q[i];
        for (int i = 0; i < 3; i++) {
            out[7 + i] = velocity[i];
            out[10 + i] = angular[i];
        }
    }

    // Unit position of camera 1 in camera 0 coordinates given R (camera 0 -> camera 1 axes); false if none.
    bool direction_known_rotation(const std::vector<cv::Point2f>& p0, const std::vector<cv::Point2f>& p1, const Mat3& R, Vec3& result) {
        const double fx = K_(0, 0), fy = K_(1, 1), cx = K_(0, 2), cy = K_(1, 2);
        const int n = (int)p0.size();
        std::vector<Vec3> x1(n), r0(n), A(n);
        std::vector<double> flow(n);
        for (int k = 0; k < n; k++) {
            const Vec3 a((p0[k].x - cx) / fx, (p0[k].y - cy) / fy, 1.0);
            x1[k] = Vec3((p1[k].x - cx) / fx, (p1[k].y - cy) / fy, 1.0);
            r0[k] = R * a;
            const double dx = r0[k][0] / r0[k][2] - x1[k][0], dy = r0[k][1] / r0[k][2] - x1[k][1];
            flow[k] = std::sqrt(dx * dx + dy * dy);
            A[k] = r0[k].cross(x1[k]);
        }
        if (median(flow) * fx < P_MIN_DEROTATED_PARALLAX * P_SCALE) return false;
        const double threshold = P_KR_THRESHOLD_PX * P_SCALE / fx;
        auto distance = [&](const Vec3& t, int k) {
            const Vec3 line = t.cross(r0[k]);
            return std::abs(A[k].dot(t)) / std::max(std::sqrt(line[0] * line[0] + line[1] * line[1]), 1e-12);
        };
        Vec3 best;
        int best_count = -1;
        for (int h = 0; h < P_KR_HYPOTHESES; h++) {
            const int i = (int)(rng_() % (uint32_t)n);
            const int j = (i + 1 + (int)(rng_() % (uint32_t)(n - 1))) % n;
            Vec3 t = A[i].cross(A[j]);
            const double norm = cv::norm(t);
            if (norm <= 1e-12) continue;
            t *= 1.0 / norm;
            int count = 0;
            for (int k = 0; k < n; k++) count += distance(t, k) < threshold;
            if (count > best_count) {
                best_count = count;
                best = t;
            }
        }
        if (best_count < 0) return false;
        Vec3 t = best;
        for (int iteration = 0; iteration < 2; iteration++) {  // weighted least squares on the inliers
            cv::Matx33d M = cv::Matx33d::zeros();
            int count = 0;
            for (int k = 0; k < n; k++) {
                const Vec3 line = t.cross(r0[k]);
                const double w = 1.0 / std::max(std::sqrt(line[0] * line[0] + line[1] * line[1]), 1e-12);
                if (std::abs(A[k].dot(t)) * w >= threshold) continue;
                const Vec3 a = A[k] * w;
                M += a * a.t();
                count++;
            }
            if (count < P_MIN_INLIERS) return false;
            cv::Mat values, vectors;
            cv::eigen(cv::Mat(M), values, vectors);  // descending eigenvalues, eigenvectors as rows
            t = Vec3(vectors.at<double>(2, 0), vectors.at<double>(2, 1), vectors.at<double>(2, 2));
        }
        std::vector<double> depth;
        for (int k = 0; k < n; k++) {
            if (distance(t, k) >= threshold) continue;
            const Vec3 c = x1[k].cross(r0[k]);
            depth.push_back(-(x1[k].cross(t)).dot(c) / std::max(c.dot(c), 1e-18));
        }
        if ((int)depth.size() < P_MIN_INLIERS) return false;
        if (median(depth) < 0) t = -t;
        result = -(R.t() * t);
        return true;
    }

    static void subsample(std::vector<cv::Point2f>& p0, std::vector<cv::Point2f>& p1) {
        const int n = (int)p0.size();
        if (P_E_MAX_POINTS <= 0 || n <= P_E_MAX_POINTS) return;
        std::vector<cv::Point2f> a, b;
        for (int k = 0; k < P_E_MAX_POINTS; k++) {
            const int i = (int)std::lround((double)k * (n - 1) / (P_E_MAX_POINTS - 1));
            a.push_back(p0[i]);
            b.push_back(p1[i]);
        }
        p0.swap(a);
        p1.swap(b);
    }

    bool track(const cv::Mat& frame, const Mat3& R_c1_c0, Vec3& direction) {
        // strips: track any surviving points so a partial refill accumulates; otherwise the upstream early exit
        const int needed = P_DETECT_STRIPS > 0 ? 1 : P_MIN_TRACKS;
        if (previous_frame_.empty() || (int)previous_points_.size() < needed) return false;
        std::vector<cv::Point2f> tracked;
        std::vector<uchar> status;
        std::vector<float> error;
        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, P_LK_ITERS, P_LK_EPS);
        cv::calcOpticalFlowPyrLK(previous_frame_, frame, previous_points_, tracked, status, error, cv::Size(P_LK_WIN, P_LK_WIN),
                                 P_LK_LEVELS, criteria);
        const int n = (int)previous_points_.size();
        std::vector<bool> good(n);
        for (int k = 0; k < n; k++) good[k] = status[k] == 1;
        if (P_LK_FB_PX > 0) {
            std::vector<cv::Point2f> back;
            std::vector<uchar> status_back;
            cv::calcOpticalFlowPyrLK(frame, previous_frame_, tracked, back, status_back, error, cv::Size(P_LK_WIN, P_LK_WIN),
                                     P_LK_LEVELS, criteria);
            for (int k = 0; k < n; k++) {
                const double miss = cv::norm(back[k] - previous_points_[k]);
                good[k] = good[k] && status_back[k] == 1 && miss < P_LK_FB_PX * P_SCALE;
            }
        }
        next_points_.clear();
        next_ids_.clear();
        has_next_ = true;
        const bool ids_valid = ids_.size() == (size_t)n;
        std::vector<cv::Point2f> p0, p1;
        for (int k = 0; k < n; k++) {
            if (!good[k]) continue;
            p0.push_back(previous_points_[k]);
            p1.push_back(tracked[k]);
            const auto& q = tracked[k];
            if (q.x >= 0 && q.x < frame.cols && q.y >= 0 && q.y < frame.rows) {
                next_points_.push_back(q);
                if (ids_valid) next_ids_.push_back(ids_[k]);
            }
        }
        next_ids_valid_ = ids_valid;
        if ((int)p0.size() < P_MIN_TRACKS) return false;
        subsample(p0, p1);
        return direction_known_rotation(p0, p1, R_c1_c0, direction);
    }

    bool direction_multiframe(Vec3& result) {
        if ((int)history_.size() < P_BASELINE_FRAMES || !has_next_ || !next_ids_valid_) return false;
        const Frame& then = history_[history_.size() - P_BASELINE_FRAMES];
        std::vector<cv::Point2f> p0, p1;
        if (!common_tracks(then, p0, p1)) return false;
        subsample(p0, p1);
        const Mat3 R_wc_now = R_wb_ * R_bc_;
        Vec3 d;
        if (!direction_known_rotation(p0, p1, R_wc_now.t() * then.R_wc, d)) return false;
        d = then.R_wc * d;
        result = d * (1.0 / std::max(cv::norm(d), 1e-12));
        return true;
    }

    // Tracks seen both in `then` and in the current frame (matched by id), in the order of increasing id.
    bool common_tracks(const Frame& then, std::vector<cv::Point2f>& p0, std::vector<cv::Point2f>& p1) const {
        if (then.ids.size() != then.points.size() || !has_next_ || !next_ids_valid_) return false;
        // ids are increasing within each list's new part but not overall: match by sorting (the Python version uses intersect1d)
        std::vector<std::pair<long, int>> a, b;
        for (int k = 0; k < (int)then.ids.size(); k++) a.emplace_back(then.ids[k], k);
        for (int k = 0; k < (int)next_ids_.size(); k++) b.emplace_back(next_ids_[k], k);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        p0.clear();
        p1.clear();
        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i].first < b[j].first) {
                i++;
            } else if (b[j].first < a[i].first) {
                j++;
            } else {
                p0.push_back(then.points[a[i].second]);
                p1.push_back(next_points_[b[j].second]);
                i++;
                j++;
            }
        }
        return (int)p0.size() >= P_MIN_TRACKS;
    }

    // Gyro bias from vision: over P_BIAS_BASELINE frames, the small rotation delta (camera-1 axes, R_true = exp(delta)
    // R_gyro) that best explains the common tracks together with the translation direction; the bias error is
    // R_bc delta / elapsed. Same as _update_gyro_bias in the Python version.
    void update_gyro_bias() {
        if ((int)history_.size() < P_BIAS_BASELINE) return;
        const Frame& then = history_[history_.size() - P_BIAS_BASELINE];
        std::vector<cv::Point2f> p0, p1;
        if (!common_tracks(then, p0, p1)) return;
        const double elapsed = last_t_ - then.t;
        if (elapsed <= 0) return;
        const Mat3 R = (R_wb_ * R_bc_).t() * then.R_wc;
        const double fx = K_(0, 0), fy = K_(1, 1), cx = K_(0, 2), cy = K_(1, 2);
        const int n = (int)p0.size();
        std::vector<Vec3> x0(n), x1(n);
        for (int k = 0; k < n; k++) {
            x0[k] = Vec3((p0[k].x - cx) / fx, (p0[k].y - cy) / fy, 1.0);
            x1[k] = Vec3((p1[k].x - cx) / fx, (p1[k].y - cy) / fy, 1.0);
        }
        const double threshold = P_KR_THRESHOLD_PX * P_SCALE / fx;
        Vec3 delta(0, 0, 0);
        for (int iteration = 0; iteration < 2; iteration++) {
            const Mat3 Rc = rodrigues(delta) * R;
            Vec3 d;
            if (!direction_known_rotation(p0, p1, Rc, d)) return;
            const Vec3 t = -(Rc * d);
            cv::Matx33d GtG = cv::Matx33d::eye() * P_BIAS_RIDGE;
            Vec3 GtE(0, 0, 0);
            int count = 0;
            for (int k = 0; k < n; k++) {
                const Vec3 r0 = Rc * x0[k];
                const Vec3 A = r0.cross(x1[k]);
                const Vec3 line = t.cross(r0);
                const double w = 1.0 / std::max(std::sqrt(line[0] * line[0] + line[1] * line[1]), 1e-12);
                const double e = A.dot(t) * w;
                if (std::abs(e) >= 3 * threshold) continue;
                const Vec3 g = r0.cross(x1[k].cross(t)) * w;
                GtG += g * g.t();
                GtE += g * e;
                count++;
            }
            if (count < P_MIN_INLIERS) return;
            delta += -(GtG.inv() * GtE);
        }
        const Vec3 error = (R_bc_ * delta) * (1.0 / elapsed);
        const double cap = P_BIAS_MAX_STEP_DPS * M_PI / 180.0;
        for (int k = 0; k < 3; k++) gyro_bias_[k] += std::clamp(P_BIAS_GAIN * error[k], -cap, cap);
    }

    void new_ids(size_t n) {
        for (size_t k = 0; k < n; k++) ids_.push_back(id_counter_++);
    }

    void detect(const cv::Mat& frame, std::vector<cv::Point2f>& points) {
        points.clear();
        cv::goodFeaturesToTrack(frame, points, P_MAX_CORNERS, P_QUALITY, P_MIN_DISTANCE, cv::noArray(), P_BLOCK_SIZE);
        ids_.clear();
        new_ids(points.size());
    }

    void replenish(const cv::Mat& frame) {
        const bool usable = has_next_ && next_ids_valid_ && !next_points_.empty();
        std::vector<cv::Point2f> kept;
        std::vector<long> kept_ids;
        kept.swap(next_points_);
        kept_ids.swap(next_ids_);
        has_next_ = false;
        trace_kept_ = usable ? (int)kept.size() : -1;
        if (P_DETECT_STRIPS > 0) {  // bounded detection: only one horizontal strip of the image per frame, cycling
            if (!usable) {
                kept.clear();
                kept_ids.clear();
            }
            ids_ = kept_ids;
            if (usable && kept.size() >= P_REPLENISH_FRACTION * P_MAX_CORNERS) {
                previous_points_ = kept;
                return;
            }
            const int strips = P_DETECT_STRIPS, k = strip_++ % strips;
            const int y0 = frame.rows * k / strips, y1 = frame.rows * (k + 1) / strips;
            const cv::Rect roi(0, y0, frame.cols, y1 - y0);
            cv::Mat mask(roi.size(), CV_8U, cv::Scalar(255));
            for (const auto& q : kept) cv::circle(mask, cv::Point((int)q.x, (int)q.y - y0), (int)P_MIN_DISTANCE, cv::Scalar(0), -1);
            std::vector<cv::Point2f> fresh;
            const int want = std::max(1, std::min(P_MAX_CORNERS - (int)kept.size(), (P_MAX_CORNERS + strips - 1) / strips + 2));
            cv::goodFeaturesToTrack(frame(roi), fresh, want, P_QUALITY, P_MIN_DISTANCE, mask, P_BLOCK_SIZE);
            for (auto& q : fresh) q.y += (float)y0;
            new_ids(fresh.size());
            kept.insert(kept.end(), fresh.begin(), fresh.end());
            previous_points_ = kept;
            return;
        }
        if (!usable) {
            detect(frame, previous_points_);
            return;
        }
        ids_ = kept_ids;
        if (kept.size() >= P_REPLENISH_FRACTION * P_MAX_CORNERS) {
            previous_points_ = kept;
            return;
        }
        cv::Mat mask(frame.size(), CV_8U, cv::Scalar(255));
        for (const auto& q : kept) cv::circle(mask, cv::Point((int)q.x, (int)q.y), (int)P_MIN_DISTANCE, cv::Scalar(0), -1);
        std::vector<cv::Point2f> fresh;
        cv::goodFeaturesToTrack(frame, fresh, P_MAX_CORNERS - (int)kept.size(), P_QUALITY, P_MIN_DISTANCE, mask, P_BLOCK_SIZE);
        new_ids(fresh.size());
        kept.insert(kept.end(), fresh.begin(), fresh.end());
        previous_points_ = kept;
    }

    cv::Matx33d K_;
    Mat4 T_cam_imu_, T_imu_cam_, T_w_c0_ = Mat4::eye();
    Mat3 R_bc_, R_wb_ = Mat3::eye();
    Vec3 v_{0, 0, 0}, p_{0, 0, 0}, omega_{0, 0, 0}, gyro_bias_{0, 0, 0}, last_direction_{0, 0, 0};
    bool has_last_direction_ = false, initialized_ = false;
    double last_t_ = 0;
    std::vector<std::vector<double>> calibration_;
    cv::Mat previous_frame_;
    std::vector<cv::Point2f> previous_points_, next_points_;
    std::vector<long> ids_, next_ids_;
    bool has_next_ = false, next_ids_valid_ = false;
    long id_counter_ = 0;
    int frame_index_ = 0, trace_kept_ = 0, strip_ = 0;
    std::deque<Frame> history_;
    std::mt19937 rng_;
};

std::unique_ptr<Estimator> estimator;
int width = 320, height = 240;
cv::Mat gray_full, gray;

const cv::Mat& to_gray(const uint8_t* rgb) {
    if (P_FAST_GRAY == -1) {  // full-block mean of the green channel only (separable row sums, integer adds)
        const int block = (int)std::lround(1.0 / P_SCALE);
        const int w = width / block, h = height / block;
        gray.create(h, w, CV_8U);
        std::vector<int> acc(w);
        const int shift_div = block * block;
        for (int i = 0; i < h; i++) {
            std::fill(acc.begin(), acc.end(), 0);
            for (int a = 0; a < block; a++) {
                const uint8_t* q = rgb + 3 * (size_t)(i * block + a) * width + 1;
                int* out = acc.data();
                if (block == 4) {
                    for (int j = 0; j < w; j++, q += 12) out[j] += q[0] + q[3] + q[6] + q[9];
                } else if (block == 8 && P_GRAY_ROWS_8 > 0) {  // block 8: P_GRAY_ROWS_8 of the 8 rows, every other column
                    if (a % (8 / P_GRAY_ROWS_8) != 8 / P_GRAY_ROWS_8 / 2) continue;
                    for (int j = 0; j < w; j++, q += 24) out[j] += 2 * (q[0] + q[6] + q[12] + q[18]);
                } else if (block == 8) {
                    for (int j = 0; j < w; j++, q += 24) out[j] += q[0] + q[3] + q[6] + q[9] + q[12] + q[15] + q[18] + q[21];
                } else {
                    for (int j = 0; j < w; j++) {
                        int s = 0;
                        for (int b = 0; b < block; b++, q += 3) s += q[0];
                        out[j] += s;
                    }
                }
            }
            uint8_t* row = gray.ptr<uint8_t>(i);
            const int used = (block == 8 && P_GRAY_ROWS_8 > 0) ? 8 * P_GRAY_ROWS_8 : shift_div;
            for (int j = 0; j < w; j++) row[j] = (uint8_t)((acc[j] + used / 2) / used);
        }
        return gray;
    }
    if (P_FAST_GRAY == -3 && (int)std::lround(1.0 / P_SCALE) == 4) {  // (R+G+B) block mean, 12 bytes per block row via SWAR
        const int w = width / 4, h = height / 4;
        gray.create(h, w, CV_8U);
        std::vector<uint32_t> acc(w);
        const uint64_t lo = 0x00FF00FF00FF00FFull;
        for (int i = 0; i < h; i++) {
            std::fill(acc.begin(), acc.end(), 0u);
            for (int a = 0; a < 4; a++) {
                const uint8_t* q = rgb + 3 * (size_t)(i * 4 + a) * width;
                for (int j = 0; j < w; j++, q += 12) {
                    uint64_t x;
                    uint32_t y;
                    std::memcpy(&x, q, 8);
                    std::memcpy(&y, q + 8, 4);
                    uint64_t pairs = (x & lo) + ((x >> 8) & lo);
                    const uint32_t yl = (y & 0x00FF00FFu) + ((y >> 8) & 0x00FF00FFu);
                    pairs += (uint64_t)(yl & 0xFFFFu) + ((uint64_t)(yl >> 16));
                    acc[j] += (uint32_t)((pairs * 0x0001000100010001ull) >> 48);
                }
            }
            uint8_t* row = gray.ptr<uint8_t>(i);
            for (int j = 0; j < w; j++) row[j] = (uint8_t)((acc[j] + 24) / 48);
        }
        return gray;
    }
    if (P_FAST_GRAY == -2) {  // OpenCV area resize of the RGB image, then gray at the small size
        const cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(rgb));
        cv::resize(image, gray_full, cv::Size(), P_SCALE, P_SCALE, cv::INTER_AREA);
        cv::cvtColor(gray_full, gray, cv::COLOR_RGB2GRAY);
        return gray;
    }
    if (P_FAST_GRAY > 0) {  // block mean of luma over a P_FAST_GRAY-strided subset of each 1/scale block, straight from RGB
        const int block = (int)std::lround(1.0 / P_SCALE), stride = P_FAST_GRAY;
        const int w = width / block, h = height / block;
        gray.create(h, w, CV_8U);
        const int per_block = ((block + stride - 1) / stride) * ((block + stride - 1) / stride);
        const int offset = (block - 1 - (block - 1) / stride * stride) / 2;
        for (int i = 0; i < h; i++) {
            uint8_t* row = gray.ptr<uint8_t>(i);
            for (int j = 0; j < w; j++) {
                int sum = 0;
                for (int a = offset; a < block; a += stride) {
                    const uint8_t* q = rgb + 3 * ((size_t)(i * block + a) * width + j * block);
                    for (int b = offset; b < block; b += stride) {
                        const uint8_t* px = q + 3 * b;
                        sum += 77 * px[0] + 150 * px[1] + 29 * px[2];
                    }
                }
                row[j] = (uint8_t)((sum / per_block + 128) >> 8);
            }
        }
        return gray;
    }
    const cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(rgb));
    cv::cvtColor(image, gray_full, cv::COLOR_RGB2GRAY);
    if (P_SCALE == 1.0) return gray_full;
    cv::resize(gray_full, gray, cv::Size(), P_SCALE, P_SCALE, cv::INTER_AREA);
    return gray;
}

double number(const cv::FileNode& node, const char* name) {
    if (!node.isReal() && !node.isInt()) throw std::runtime_error(std::string("missing number: ") + name);
    return (double)node;
}

// Run a throwaway estimator over synthetic frames so first-use costs land in init (like the Python warmup).
void warmup(const cv::Matx33d& K, const Mat4& T_cam_imu) {
    Estimator dummy(K, T_cam_imu);
    cv::Mat noise(height, width + 64, CV_8UC3);
    cv::RNG generator(0);
    generator.fill(noise, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(noise, noise, cv::Size(5, 5), 0);
    std::vector<double> imu(70);
    double out[13];
    for (int k = 0; k < 16; k++) {
        const double t = (k + 1) * 10 / 200.0;
        for (int i = 0; i < 10; i++) {
            double* r = &imu[7 * i];
            r[0] = t - (9 - i) / 200.0;
            r[1] = k >= 11 ? 0.5 : 0.0;
            r[2] = 0.0;
            r[3] = GRAVITY;
            r[4] = r[5] = r[6] = k >= 11 ? 0.01 : 0.0;
        }
        const cv::Mat crop = noise(cv::Rect(3 * k, 0, width, height)).clone();
        dummy.step(t, to_gray(crop.data), imu.data(), 10, out);
    }
}

}  // namespace

extern "C" {

VIOBENCH_EXPORT(abi_version) int32_t abi_version() { return VIOBENCH_ABI_VERSION; }

VIOBENCH_EXPORT(alloc) void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }

VIOBENCH_EXPORT(frame_width) int32_t frame_width() { return 320; }

VIOBENCH_EXPORT(init) int32_t init(const char* json, uint32_t length) {
    try {
        setvbuf(stdout, nullptr, _IONBF, 0);
        setvbuf(stderr, nullptr, _IONBF, 0);
        cv::setRNGSeed(0);
        cv::FileStorage config(std::string(json, length), cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        const double fx = number(config["fx"], "fx"), fy = number(config["fy"], "fy");
        const double cx = number(config["cx"], "cx"), cy = number(config["cy"], "cy");
        if (!config["width"].empty()) width = (int)number(config["width"], "width");
        if (!config["height"].empty()) height = (int)number(config["height"], "height");
        Mat4 T = Mat4::eye();
        const cv::FileNode rows = config["T_cam_imu"];
        if (!rows.isSeq() || rows.size() != 4) throw std::runtime_error("T_cam_imu must be 4 x 4");
        for (int i = 0; i < 4; i++) {
            const cv::FileNode row = rows[i];
            if (!row.isSeq() || row.size() != 4) throw std::runtime_error("T_cam_imu must be 4 x 4");
            for (int j = 0; j < 4; j++) T(i, j) = number(row[j], "T_cam_imu");
        }
        const cv::Matx33d K(fx * P_SCALE, 0, (cx + 0.5) * P_SCALE - 0.5, 0, fy * P_SCALE, (cy + 0.5) * P_SCALE - 0.5, 0, 0, 1);
        if (P_WARMUP) warmup(K, T);
        estimator = std::make_unique<Estimator>(K, T);
        cv::setRNGSeed(0);
        return VIOBENCH_STATUS_OK;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return VIOBENCH_STATUS_ERROR;
    }
}

VIOBENCH_EXPORT(step) int32_t step(double t, const uint8_t* frame, const double* imu, int32_t imu_count, double* out) {
    try {
        if (!estimator) return VIOBENCH_STATUS_ERROR;
        if (P_PROBE == 1 || P_PROBE == 2) {  // measurement only: 1 = gray conversion alone, 2 = nothing
            if (P_PROBE == 1) to_gray(frame);
            std::fill(out, out + 13, 0.0);
            out[6] = 1.0;
            return VIOBENCH_STATUS_OK;
        }
        estimator->step(t, to_gray(frame), imu, imu_count, out);
        return VIOBENCH_STATUS_OK;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "step failed: %s\n", e.what());
        return VIOBENCH_STATUS_ERROR;
    }
}

}  // extern "C"
