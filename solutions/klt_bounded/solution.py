"""Bounded-compute KLT + IMU.

Derived from the VIOBench `klt_imu` example (https://github.com/jonas-eschmann/viobench, MIT, pinned commit
17303aa14ba14e16cb54d9291d684371e1d3b548, viobench/solution/examples/python/klt_imu/solution.py). The estimator is
unchanged; the per-frame visual work is bounded by parameters read from `params.json` in the bundle:

  scale            process the gray image at this fraction of its size (camera matrix scaled to match)
  max_corners      goodFeaturesToTrack budget; min_distance, block_size, quality go with it
  lk_win, lk_levels  pyramidal LK window and pyramid depth
  e_max_points     at most this many tracks (evenly spread over the track list) enter RANSAC and recoverPose
  e_max_iters      RANSAC iteration cap for findEssentialMat
  e_threshold_px   RANSAC inlier threshold in pixels at full resolution (scaled with the image)
  e_prob           RANSAC confidence; with the inlier ratio it sets the adaptive iteration count
  lk_iters, lk_eps LK termination criteria per pyramid level (OpenCV default 30, 0.01)
  warmup           run every OpenCV call once in __init__ on synthetic data, so first-use costs land in the
                   separately budgeted init instead of the first tracked frame (outputs are unaffected)

Profiling the upstream baseline showed its peak-fuel frames are those where findEssentialMat + recoverPose run on
all ~300 tracks with many RANSAC iterations; on frames without parallax both are skipped. Bounding the points and
iterations bounds the worst frame; the remaining parameters lower the floor set by detection and tracking.
Stage 2 options (off by default):

  direction        "essential" (upstream: 5-point essential matrix + recoverPose) or "known_rotation": the rotation
                   between frames comes from the integrated gyroscope, so only the translation direction t remains.
                   Each track gives one linear constraint t . (R x0 x x1) = 0; a fixed number of 2-point hypotheses
                   (kr_hypotheses) is scored on all tracks at once (kr_threshold_px), refined on the inliers by the
                   smallest eigenvector, and its sign chosen by cheirality. Frames whose parallax after removing the
                   rotation is below min_derotated_parallax_px give no direction (pure rotation: trust the IMU).
  fusion           how the visual direction and the IMU combine (upstream: IMU displacement length along the visual
                   direction, velocity averaged with it). "project": IMU displacement and velocity are projected onto
                   the visual direction (drift across it is removed). "constant_speed": a fixed step length
                   (speed_prior m/s) along the visual direction; without a direction the camera is held still.
                   Sim(3) alignment fits one global scale, so only consistent step lengths matter for the headline score.
  no_vision        constant_speed only: "hold" (stand still) or "coast" (keep the last velocity) without a direction
  direction_smoothing  constant_speed only: weight in [0, 1) of the previous world direction in an exponential
                   average with the new one (0: none)
  speed_model      constant_speed only: "constant" (default) or "flow": step length scaled by
                   (derotated image motion / its running average) ** speed_flow_alpha, clipped to speed_flow_clip;
                   or "metric": speed from a sliding-window least-squares fit of the IMU velocity changes to the
                   visual directions. Over the last metric_window_s seconds, s_k d_k = c + V_k + b (t_k - t_now)
                   for every frame k with a direction d_k, where V_k is the integrated IMU velocity change and the
                   unknowns are the speeds s_k, the start velocity c and a constant world-frame bias b (accelerometer
                   bias and tilt leakage of gravity). metric_smooth ties neighbouring speeds, metric_prior pulls them
                   weakly toward the current estimate (straight, steady flight carries no scale information). The
                   newest speed, clipped to metric_clip, updates the estimate with weight metric_alpha.
  tilt_tau         if set, roll and pitch are pulled toward the long-run average of the specific force (world frame)
                   with this time constant (s); on a multirotor the accelerometer shows gravity only on average
  track_mode       "redetect" (upstream: new corners every frame) or "persist": tracks are kept and topped up with
                   new corners away from existing ones when fewer than replenish_fraction * max_corners remain
  lk_fb_px         if set, a forward-backward LK check drops tracks whose return position misses by more than this
                   many pixels (full resolution); costs a second LK pass
  baseline_frames  track_mode persist, constant_speed: direction from tracks seen baseline_frames frames ago (larger
                   parallax, lag of about half the baseline); falls back to the one-frame direction
  rng_seed         seed of the hypothesis sampling (known_rotation)
  speed_tau        if set, velocity decays with this time constant (s) every frame, bounding IMU drift
  imu_vectorized   the same IMU integration as upstream with array operations instead of per-sample Python work.

Missing parameters take the upstream values, so an empty params.json reproduces the baseline's behaviour.
"""
import json

import cv2
import numpy as np

try:
    import solution_bundle as _bundle
    PARAMS = json.loads(_bundle.read_text("params.json"))
except Exception:  # no params.json: upstream defaults
    PARAMS = {}

GRAVITY = 9.81
CALIBRATION_SECONDS = 0.5
REST_ACCEL_STD = 1.0
REST_GYRO_STD = 0.2
MIN_TRACKS = PARAMS.get("min_tracks", 30)
MIN_INLIERS = PARAMS.get("min_inliers", 15)
MIN_PARALLAX_PIXELS = 1.0  # at full resolution

SCALE = float(PARAMS.get("scale", 1.0))
MAX_CORNERS = int(PARAMS.get("max_corners", 300))
QUALITY = float(PARAMS.get("quality", 0.01))
MIN_DISTANCE = float(PARAMS.get("min_distance", 7))
BLOCK_SIZE = int(PARAMS.get("block_size", 7))
LK_WIN = int(PARAMS.get("lk_win", 21))
LK_LEVELS = int(PARAMS.get("lk_levels", 3))
E_MAX_POINTS = PARAMS.get("e_max_points")  # None: all tracks
E_MAX_ITERS = int(PARAMS.get("e_max_iters", 1000))  # OpenCV's default
E_THRESHOLD = float(PARAMS.get("e_threshold_px", 1.0)) * SCALE
E_PROB = float(PARAMS.get("e_prob", 0.999))
LK_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, int(PARAMS.get("lk_iters", 30)), float(PARAMS.get("lk_eps", 0.01)))
WARMUP = bool(PARAMS.get("warmup", False))
DIRECTION = PARAMS.get("direction", "essential")
KR_HYPOTHESES = int(PARAMS.get("kr_hypotheses", 32))
KR_THRESHOLD_PX = float(PARAMS.get("kr_threshold_px", 1.0))  # at full resolution
MIN_DEROTATED_PARALLAX = float(PARAMS.get("min_derotated_parallax_px", 1.0))  # at full resolution
IMU_VECTORIZED = bool(PARAMS.get("imu_vectorized", False))
FUSION = PARAMS.get("fusion", "upstream")
SPEED_PRIOR = float(PARAMS.get("speed_prior", 0.5))
SPEED_TAU = PARAMS.get("speed_tau")
NO_VISION = PARAMS.get("no_vision", "hold")
DIRECTION_SMOOTHING = float(PARAMS.get("direction_smoothing", 0.0))
SPEED_MODEL = PARAMS.get("speed_model", "constant")
SPEED_FLOW_ALPHA = float(PARAMS.get("speed_flow_alpha", 1.0))
SPEED_FLOW_CLIP = tuple(PARAMS.get("speed_flow_clip", [0.25, 4.0]))
SPEED_FLOW_TAU = float(PARAMS.get("speed_flow_tau", 5.0))
TILT_TAU = PARAMS.get("tilt_tau")
TRACK_MODE = PARAMS.get("track_mode", "redetect")
REPLENISH_FRACTION = float(PARAMS.get("replenish_fraction", 0.7))
RNG_SEED = int(PARAMS.get("rng_seed", 0))
LK_FB_PX = PARAMS.get("lk_fb_px")
BASELINE_FRAMES = int(PARAMS.get("baseline_frames", 1))
METRIC_WINDOW_S = float(PARAMS.get("metric_window_s", 2.0))
METRIC_SMOOTH = float(PARAMS.get("metric_smooth", 1.0))
METRIC_PRIOR = float(PARAMS.get("metric_prior", 0.1))
METRIC_BIAS = bool(PARAMS.get("metric_bias", True))
METRIC_MIN_DIRECTIONS = int(PARAMS.get("metric_min_directions", 8))
METRIC_MAX_DIRECTIONS = int(PARAMS.get("metric_max_directions", 48))
METRIC_ALPHA = float(PARAMS.get("metric_alpha", 0.3))
METRIC_CLIP = tuple(PARAMS.get("metric_clip", [0.05, 4.0]))
METRIC_ANCHOR = PARAMS.get("metric_anchor", "estimate")  # prior target: "estimate" (current) or "prior" (speed_prior)
METRIC_BIAS_WEIGHT = float(PARAMS.get("metric_bias_weight", 0.0))  # ridge on the bias (per s^2 of window)
BIAS_BASELINE = int(PARAMS.get("bias_baseline_frames", 0))  # 0: off; else frames between the views compared
BIAS_GAIN = float(PARAMS.get("bias_gain", 0.05))
BIAS_ATTITUDE_GAIN = float(PARAMS.get("bias_attitude_gain", 0.0))
BIAS_RIDGE = float(PARAMS.get("bias_ridge", 1e-6))
BIAS_MAX_STEP = float(PARAMS.get("bias_max_step_dps", 0.2))  # cap per update (deg/s)
LM_MIN_PARALLAX_DEG = float(PARAMS.get("lm_min_parallax_deg", 3.0))  # speed_model landmark: triangulation gate
LM_MIN_POINTS = int(PARAMS.get("lm_min_points", 8))
LM_ALPHA = float(PARAMS.get("lm_alpha", 0.3))  # weight of the new speed measurement in the running estimate
LM_CLIP = tuple(PARAMS.get("lm_clip", [0.25, 4.0]))  # measured / running speed ratio bounds
LM_RESIDUAL = float(PARAMS.get("lm_residual", 2.0))
LM_NORMALIZE_TAU = PARAMS.get("lm_normalize_tau")
LM_SHADOW = bool(PARAMS.get("lm_shadow", False))  # landmarks live on a constant-speed shadow trajectory (open loop)
LM_BETA = float(PARAMS.get("lm_beta", 0.5))  # shadow: speed = speed_prior * (smoothed measurement / long-run mean) ** beta
LM_TAU = float(PARAMS.get("lm_tau", 10.0))  # s: output speed_prior * (speed / its long-run geometric mean)  # outlier gate on per-point step estimates, in median abs deviations
PROBE = PARAMS.get("probe")  # measurement only: "imu_only" skips all vision work (fuel floor of the rest)


def matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.empty(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q / np.linalg.norm(q)


class Solution:
    def __init__(self, config, warm=WARMUP):
        cv2.setRNGSeed(0)  # reproducible fuel (docs/RUNTIME.md)
        self.K = np.array([[config["fx"] * SCALE, 0, (config["cx"] + 0.5) * SCALE - 0.5],
                           [0, config["fy"] * SCALE, (config["cy"] + 0.5) * SCALE - 0.5], [0, 0, 1]])
        self.T_cam_imu = np.asarray(config["T_cam_imu"], dtype=float)
        self.T_imu_cam = np.linalg.inv(self.T_cam_imu)
        self.R_bc = self.T_imu_cam[:3, :3]
        self.calibration_samples = []
        self.initialized = False
        self.R_wb = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.omega = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.gravity_world = np.array([0.0, 0.0, -GRAVITY])
        self.last_t = None
        self.T_w_c0 = None
        self.previous_frame = None
        self.previous_points = None
        self.frame_count = 0
        self.rng = np.random.default_rng(RNG_SEED)
        self.last_direction = None
        self.last_flow = None  # median derotated image motion of the last direction estimate (normalised units)
        self.flow_average = None  # running geometric mean of last_flow
        self.next_points = None  # tracks surviving into the current frame (track_mode persist)
        self.next_ids = None
        self.ids = None  # identity of each entry of previous_points
        self.id_counter = 0
        self.history = []  # (ids, points, R_wc) of recent frames, for baseline_frames > 1
        self.metric_velocity = np.zeros(3)  # integrated IMU velocity change since start (speed_model metric)
        self.metric_window = []  # (t_mid, velocity change at mid-interval, world direction or None)
        self.metric_speed = SPEED_PRIOR
        self.lm_birth = {}  # track id -> (world bearing, camera centre) at the first observation
        self.lm_points = {}  # track id -> triangulated world point (trajectory scale)
        self.lm_speed = SPEED_PRIOR
        self.lm_used = 0
        if warm:
            self._warmup(config)

    def _warmup(self, config):
        """Run a throwaway copy of the estimator through synthetic frames, then each OpenCV path once more, so that
        one-time costs (first calls, lazy initialisation) are paid in init. Touches no state of this instance."""
        width, height = int(config.get("width", 320)), int(config.get("height", 240))
        rng = np.random.default_rng(0)
        texture = cv2.GaussianBlur(rng.integers(0, 256, (height, width + 64, 3), dtype=np.uint8), (5, 5), 0)
        dummy = Solution(config, warm=False)
        rate, per_frame = 200.0, 10
        for k in range(16):
            t = (k + 1) * per_frame / rate
            times = t - (per_frame - 1 - np.arange(per_frame)) / rate
            imu = np.zeros((per_frame, 7))
            imu[:, 0] = times
            imu[:, 3] = GRAVITY
            imu[:, 4:7] = 0.01 * (k >= 11)
            imu[:, 1] = 0.5 * (k >= 11)
            frame = np.ascontiguousarray(texture[:, 3 * k:3 * k + width])
            dummy.step(t, frame, imu)
        rgb = cv2.GaussianBlur(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), (5, 5), 0)
        a = self._gray(rgb)
        b = np.roll(a, 2, axis=1)
        points = self._detect(a)
        if points is not None and len(points) >= 8:
            cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(LK_WIN, LK_WIN), maxLevel=LK_LEVELS, criteria=LK_CRITERIA)
        h, w = a.shape
        X = np.column_stack([rng.uniform(-1, 1, 60), rng.uniform(-1, 1, 60), rng.uniform(3, 6, 60)])
        p0 = (X[:, :2] / X[:, 2:]) * self.K[0, 0] + self.K[:2, 2]
        p1 = ((X[:, :2] + [0.1, 0.0]) / X[:, 2:]) * self.K[0, 0] + self.K[:2, 2]
        p0, p1 = p0.astype(np.float32), p1.astype(np.float32)
        E, mask = cv2.findEssentialMat(p0, p1, self.K, method=cv2.RANSAC, prob=E_PROB, threshold=E_THRESHOLD, maxIters=E_MAX_ITERS)
        if E is not None and E.shape == (3, 3):
            cv2.recoverPose(E, p0, p1, self.K, mask=mask)
        cv2.Rodrigues(np.zeros(3))
        cv2.setRNGSeed(0)

    # --- IMU (unchanged from upstream) ---------------------------------------------------------------------------

    def _initialize(self, t):
        samples = np.asarray(self.calibration_samples)
        mean_accel = samples[:, 1:4].mean(axis=0)
        at_rest = np.linalg.norm(samples[:, 1:4].std(axis=0)) < REST_ACCEL_STD and np.linalg.norm(samples[:, 4:7].std(axis=0)) < REST_GYRO_STD
        self.gyro_bias = samples[:, 4:7].mean(axis=0) if at_rest else np.zeros(3)
        z_world = mean_accel / np.linalg.norm(mean_accel)
        x_world = np.array([1.0, 0.0, 0.0]) - z_world[0] * z_world
        x_world /= np.linalg.norm(x_world)
        self.R_wb = np.stack([x_world, np.cross(z_world, x_world), z_world])
        self.last_t = t
        self.initialized = True
        self.T_w_c0 = self._pose_camera()

    def _integrate(self, imu):
        start = self.p.copy()
        for row in imu:
            t, accel, gyro = row[0], row[1:4], row[4:7]
            self.omega = gyro - self.gyro_bias
            dt = t - self.last_t
            if dt > 0:
                self.R_wb = self.R_wb @ cv2.Rodrigues((gyro - self.gyro_bias) * dt)[0]
                acceleration = self.R_wb @ accel + self.gravity_world
                self.p += self.v * dt + 0.5 * acceleration * dt * dt
                self.v += acceleration * dt
            self.last_t = t
        return self.p - start

    def _integrate_vectorized(self, imu):
        """Same discretisation as _integrate (rotate, then accelerate with the new rotation), with array operations."""
        if len(imu) == 0:
            return np.zeros(3)
        start = self.p.copy()
        times = imu[:, 0]
        gyro = imu[:, 4:7] - self.gyro_bias
        dts = np.diff(np.concatenate([[self.last_t], times]))
        dts = np.where(dts > 0, dts, 0.0)
        theta = gyro * dts[:, None]
        angle = np.sqrt((theta * theta).sum(axis=1))
        small = angle < 1e-12
        safe = np.where(small, 1.0, angle)
        a = np.where(small, 1.0, np.sin(safe) / safe)
        b = np.where(small, 0.5, (1.0 - np.cos(safe)) / (safe * safe))
        x, y, z = theta[:, 0], theta[:, 1], theta[:, 2]
        zero = np.zeros_like(x)
        S = np.stack([np.stack([zero, -z, y], 1), np.stack([z, zero, -x], 1), np.stack([-y, x, zero], 1)], 1)
        dR = np.eye(3) + a[:, None, None] * S + b[:, None, None] * (S @ S)
        rotations = np.empty_like(dR)
        R = self.R_wb
        for i in range(len(dR)):
            R = R @ dR[i]
            rotations[i] = R
        acceleration = (rotations @ imu[:, 1:4, None])[:, :, 0] + self.gravity_world
        dv = acceleration * dts[:, None]
        v_before = self.v + np.concatenate([np.zeros((1, 3)), np.cumsum(dv, axis=0)[:-1]])
        self.p = self.p + (v_before * dts[:, None] + 0.5 * dv * dts[:, None]).sum(axis=0)
        self.v = self.v + dv.sum(axis=0)
        self.R_wb = R
        self.omega = gyro[-1]
        self.last_t = times[-1]
        return self.p - start

    def _pose_camera(self):
        T_wb = np.eye(4)
        T_wb[:3, :3] = self.R_wb
        T_wb[:3, 3] = self.p
        return T_wb @ self.T_imu_cam

    def _state(self):
        T_c0_c = np.linalg.inv(self.T_w_c0) @ self._pose_camera()
        R_cam_imu, r = self.T_cam_imu[:3, :3], self.T_imu_cam[:3, 3]
        velocity = R_cam_imu @ (self.R_wb.T @ self.v + np.cross(self.omega, r))
        return np.concatenate([T_c0_c[:3, 3], matrix_to_quaternion(T_c0_c[:3, :3]), velocity, R_cam_imu @ self.omega])

    # --- vision (bounded) --------------------------------------------------------------------------------------

    def _direction_known_rotation(self, p0, p1, R):
        """Unit position of camera 1 in camera 0 coordinates given R (camera 0 -> camera 1 axes), or None."""
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        n = len(p0)
        x0 = np.column_stack([(p0[:, 0] - cx) / fx, (p0[:, 1] - cy) / fy, np.ones(n)])
        x1 = np.column_stack([(p1[:, 0] - cx) / fx, (p1[:, 1] - cy) / fy, np.ones(n)])
        r0 = x0 @ R.T  # R x0: bearings of camera 0 expressed in camera 1 axes
        derotated = r0[:, :2] / r0[:, 2:3] - x1[:, :2]
        flow = float(np.median(np.sqrt((derotated * derotated).sum(axis=1))))
        self.last_flow = flow
        if flow * fx < MIN_DEROTATED_PARALLAX * SCALE:
            return None  # no translation visible: pure rotation or a distant scene
        A = np.cross(r0, x1)  # A t = x1 . (t x R x0): zero for the true direction
        i = self.rng.integers(0, n, KR_HYPOTHESES)
        j = (i + 1 + self.rng.integers(0, n - 1, KR_HYPOTHESES)) % n
        T = np.cross(A[i], A[j])
        norms = np.sqrt((T * T).sum(axis=1))
        T = T[norms > 1e-12] / norms[norms > 1e-12][:, None]
        if len(T) == 0:
            return None
        threshold = KR_THRESHOLD_PX * SCALE / fx  # normalised image units

        def epipolar_distance(t):  # distance of x1 from the epipolar line t x R x0, per track (and hypothesis)
            lines = np.cross(t[None, ...], r0.reshape((n,) + (1,) * (t.ndim - 1) + (3,)))
            return np.abs(A @ t.T) / np.maximum(np.sqrt(lines[..., 0] ** 2 + lines[..., 1] ** 2), 1e-12)

        inliers = epipolar_distance(T) < threshold
        t = T[int(np.argmax(inliers.sum(axis=0)))]
        for _ in range(2):  # refine: weighted least squares on the inliers, weights turn A t into distances
            lines = np.cross(t, r0)
            weight = 1.0 / np.maximum(np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2), 1e-12)
            mask = np.abs(A @ t) * weight < threshold
            if int(mask.sum()) < MIN_INLIERS:
                return None
            Ai = A[mask] * weight[mask, None]
            t = np.linalg.eigh(Ai.T @ Ai)[1][:, 0]
        mask = epipolar_distance(t) < threshold
        if int(mask.sum()) < MIN_INLIERS:
            return None
        # cheirality: depth of each inlier in camera 0, from lambda1 x1 = lambda0 R x0 + t
        c = np.cross(x1[mask], r0[mask])
        depth = -(np.cross(x1[mask], t) * c).sum(axis=1) / np.maximum((c * c).sum(axis=1), 1e-18)
        if np.median(depth) < 0:
            t = -t
        return -R.T @ t

    def _track(self, frame, R_c1_c0=None):
        if self.previous_frame is None or self.previous_points is None or len(self.previous_points) < MIN_TRACKS:
            return None
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame, frame, self.previous_points, None,
                                                      winSize=(LK_WIN, LK_WIN), maxLevel=LK_LEVELS, criteria=LK_CRITERIA)
        good = status.ravel() == 1
        if LK_FB_PX is not None:
            back, status_back, _ = cv2.calcOpticalFlowPyrLK(frame, self.previous_frame, tracked, None, winSize=(LK_WIN, LK_WIN),
                                                            maxLevel=LK_LEVELS, criteria=LK_CRITERIA)
            miss = np.sqrt(((back - self.previous_points) ** 2).sum(axis=2)).ravel()
            good &= (status_back.ravel() == 1) & (miss < float(LK_FB_PX) * SCALE)
        if TRACK_MODE == "persist":
            h, w = frame.shape
            inside = good & (tracked[:, 0, 0] >= 0) & (tracked[:, 0, 0] < w) & (tracked[:, 0, 1] >= 0) & (tracked[:, 0, 1] < h)
            self.next_points = tracked[inside].reshape(-1, 1, 2).astype(np.float32)
            self.next_ids = self.ids[inside] if self.ids is not None and len(self.ids) == len(good) else None
        p0 = self.previous_points[good].reshape(-1, 2)
        p1 = tracked[good].reshape(-1, 2)
        if len(p0) < MIN_TRACKS:
            return None
        if DIRECTION == "known_rotation":
            if E_MAX_POINTS is not None and len(p0) > E_MAX_POINTS:
                keep = np.linspace(0, len(p0) - 1, int(E_MAX_POINTS)).round().astype(int)
                p0, p1 = p0[keep], p1[keep]
            return self._direction_known_rotation(p0.astype(float), p1.astype(float), R_c1_c0)
        if np.median(np.linalg.norm(p1 - p0, axis=1)) < MIN_PARALLAX_PIXELS * SCALE:
            return None
        if E_MAX_POINTS is not None and len(p0) > E_MAX_POINTS:
            keep = np.linspace(0, len(p0) - 1, int(E_MAX_POINTS)).round().astype(int)  # goodFeaturesToTrack orders by corner strength
            p0, p1 = p0[keep], p1[keep]
        E, inliers = cv2.findEssentialMat(p0, p1, self.K, method=cv2.RANSAC, prob=E_PROB, threshold=E_THRESHOLD, maxIters=E_MAX_ITERS)
        if E is None or E.shape != (3, 3) or inliers is None or int(inliers.sum()) < MIN_INLIERS:
            return None
        count, R, t, _ = cv2.recoverPose(E, p0, p1, self.K, mask=inliers)
        if count < MIN_INLIERS:
            return None
        return (-R.T @ t).ravel()

    def _replenish(self, frame):
        """track_mode persist: surviving tracks, topped up with new corners away from them when too few remain."""
        kept, kept_ids = self.next_points, self.next_ids
        self.next_points = self.next_ids = None
        if kept is None or len(kept) == 0 or kept_ids is None:
            return self._detect(frame)
        if len(kept) >= REPLENISH_FRACTION * MAX_CORNERS:
            self.ids = kept_ids
            return kept
        mask = np.full(frame.shape, 255, np.uint8)
        for x, y in kept.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y)), int(MIN_DISTANCE), 0, -1)
        new = cv2.goodFeaturesToTrack(frame, maxCorners=int(MAX_CORNERS - len(kept)), qualityLevel=QUALITY,
                                      minDistance=MIN_DISTANCE, blockSize=BLOCK_SIZE, mask=mask)
        if new is None:
            self.ids = kept_ids
            return kept
        self.ids = np.concatenate([kept_ids, self._new_ids(len(new))])
        return np.concatenate([kept, new.astype(np.float32)])

    def _new_ids(self, n):
        self.id_counter += n
        return np.arange(self.id_counter - n, self.id_counter)

    def _detect(self, frame):
        points = cv2.goodFeaturesToTrack(frame, maxCorners=MAX_CORNERS, qualityLevel=QUALITY, minDistance=MIN_DISTANCE, blockSize=BLOCK_SIZE)
        self.ids = None if points is None else self._new_ids(len(points))
        return None if points is None else points.astype(np.float32)

    def _gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if SCALE != 1.0:
            gray = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        return gray

    # --- protocol (unchanged from upstream) --------------------------------------------------------------------

    def step(self, t, frame, imu):
        frame = self._gray(frame)
        self.frame_count += 1
        if not self.initialized:
            self.calibration_samples.extend(imu.tolist())
            samples = self.calibration_samples
            if len(samples) < 2 or samples[-1][0] - samples[0][0] < CALIBRATION_SECONDS:
                return [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]
            self._initialize(imu[-1, 0] if len(imu) else t)
            self.previous_frame = frame.copy()
            self.previous_points = self._detect(frame)
            return self._state()

        R_wb_before = self.R_wb.copy()
        v_before = self.v.copy()
        displacement_imu = self._integrate_vectorized(imu) if IMU_VECTORIZED else self._integrate(imu)
        R_c1_c0 = (self.R_wb @ self.R_bc).T @ (R_wb_before @ self.R_bc)
        self.last_flow = None
        direction_camera = None if PROBE == "imu_only" else self._track(frame, R_c1_c0)
        dt = max(imu[-1, 0] - imu[0, 0] + (imu[1, 0] - imu[0, 0] if len(imu) > 1 else 0.0), 1e-3) if len(imu) else 0.05
        if SPEED_MODEL == "metric":
            dv = self.v - v_before
            t_mid = (imu[-1, 0] if len(imu) else t) - 0.5 * dt
            velocity_mid = self.metric_velocity + 0.5 * dv
            self.metric_velocity = self.metric_velocity + dv
        if direction_camera is not None:
            R_wc_previous = R_wb_before @ self.R_bc
            direction_world = R_wc_previous @ direction_camera
            direction_world = direction_world / max(np.linalg.norm(direction_world), 1e-12)
            if FUSION == "project":
                displacement = max(float(displacement_imu @ direction_world), 0.0) * direction_world
                self.p += displacement - displacement_imu
                self.v = max(float(self.v @ direction_world), 0.0) * direction_world
            elif FUSION == "constant_speed":
                if BASELINE_FRAMES > 1:
                    longer = self._direction_multiframe()
                    if longer is not None:
                        direction_world = longer
                if DIRECTION_SMOOTHING > 0 and self.last_direction is not None:
                    blended = DIRECTION_SMOOTHING * self.last_direction + (1 - DIRECTION_SMOOTHING) * direction_world
                    norm = np.linalg.norm(blended)
                    direction_world = blended / norm if norm > 1e-9 else direction_world
                self.last_direction = direction_world
                speed = SPEED_PRIOR
                if SPEED_MODEL == "flow" and self.last_flow is not None and self.last_flow > 0:
                    rate = self.last_flow / dt
                    if self.flow_average is None:
                        self.flow_average = rate
                    else:
                        a = min(dt / SPEED_FLOW_TAU, 1.0)
                        self.flow_average = float(np.exp((1 - a) * np.log(self.flow_average) + a * np.log(rate)))
                    speed = SPEED_PRIOR * float(np.clip((rate / self.flow_average) ** SPEED_FLOW_ALPHA, *SPEED_FLOW_CLIP))
                    self.last_speed = speed
                elif SPEED_MODEL == "landmark":
                    speed = self.last_speed = self._landmark_speed(direction_world, displacement_imu, dt)
                elif SPEED_MODEL == "metric":
                    self._metric_push(t_mid, velocity_mid, direction_world)
                    speed = self.last_speed = self._metric_fit()
                displacement = speed * dt * direction_world
                self.p += displacement - displacement_imu
                self.v = displacement / dt
            else:
                scale = np.linalg.norm(displacement_imu)
                displacement = scale * direction_world
                correction = displacement - displacement_imu
                self.p += correction
                self.v = 0.5 * self.v + 0.5 * displacement / dt
        elif FUSION == "constant_speed":
            if SPEED_MODEL == "landmark" and LM_SHADOW and self.last_direction is not None and NO_VISION == "coast":
                self.lm_shadow = self._camera_centre() + SPEED_PRIOR * dt * self.last_direction
            if SPEED_MODEL == "metric":
                self._metric_push(t_mid, velocity_mid, None)
            self.p -= displacement_imu
            if NO_VISION == "coast" and self.last_direction is not None:
                speed = getattr(self, "last_speed", SPEED_PRIOR)
                self.p += speed * dt * self.last_direction
                self.v = speed * self.last_direction
            else:  # no visible translation: hold still
                self.v = np.zeros(3)
        if SPEED_TAU is not None:
            self.v *= np.exp(-dt / float(SPEED_TAU))
        if TILT_TAU is not None and len(imu):
            self._correct_tilt(imu, dt)
        if BIAS_BASELINE > 1 and PROBE != "imu_only":
            self._update_gyro_bias()
        if SPEED_MODEL == "landmark":
            self._landmark_update()
        current_ids, current_points = self.next_ids, self.next_points
        self.previous_frame = frame.copy()
        if PROBE != "imu_only":
            self.previous_points = self._replenish(frame) if TRACK_MODE == "persist" else self._detect(frame)
        keep = max(BASELINE_FRAMES, BIAS_BASELINE)
        if keep > 1:
            self.history.append((self.ids, None if self.previous_points is None else self.previous_points.reshape(-1, 2).copy(),
                                 self.R_wb @ self.R_bc, self.last_t))
            del self.history[:-keep]
        return self._state()

    def _world_bearings(self, points):
        """Unit bearings in world axes of pixel points in the current frame."""
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x = np.column_stack([(points[:, 0] - cx) / fx, (points[:, 1] - cy) / fy, np.ones(len(points))])
        b = x @ (self.R_wb @ self.R_bc).T
        return b / np.linalg.norm(b, axis=1, keepdims=True)

    def _camera_centre(self):
        if LM_SHADOW:
            return getattr(self, "lm_shadow", np.zeros(3))
        return (self._pose_camera())[:3, 3]

    def _landmark_speed(self, direction, displacement_imu, dt):
        """Step length along `direction` that best re-projects the triangulated landmarks seen in this frame (rotation
        known, centre c = c_prev + s d: cross(X - c_prev, b) = s cross(d, b) per landmark, least squares with a robust gate)."""
        if LM_SHADOW:
            shadow_prev = self._camera_centre()
            self.lm_shadow = shadow_prev + SPEED_PRIOR * dt * direction
        if self.next_ids is None or self.next_points is None or not self.lm_points:
            return self.lm_speed
        ids = self.next_ids
        keep = [k for k, i in enumerate(ids) if int(i) in self.lm_points]
        if len(keep) < LM_MIN_POINTS:
            return self.lm_speed
        b = self._world_bearings(self.next_points.reshape(-1, 2)[keep].astype(float))
        X = np.array([self.lm_points[int(ids[k])] for k in keep])
        # centre before this frame's motion: the IMU displacement was already added to p, remove it
        c_prev = shadow_prev if LM_SHADOW else self._camera_centre() - displacement_imu
        u = np.cross(direction, b)
        r = np.cross(X - c_prev, b)
        den = (u * u).sum(axis=1)
        ok = den > 1e-8
        if ok.sum() < LM_MIN_POINTS:
            return self.lm_speed
        s_each = (u[ok] * r[ok]).sum(axis=1) / den[ok]
        w = den[ok]
        med = np.median(s_each)
        mad = np.median(np.abs(s_each - med)) + 1e-9
        inl = np.abs(s_each - med) < LM_RESIDUAL * 1.4826 * mad
        if inl.sum() < LM_MIN_POINTS:
            return self.lm_speed
        step = float((w[inl] * s_each[inl]).sum() / w[inl].sum())
        self.lm_used += 1
        if LM_SHADOW:
            m = max(step / dt, 1e-3)
            ema = getattr(self, "lm_ema", m)
            ema = (1 - LM_ALPHA) * ema + LM_ALPHA * float(np.clip(m, LM_CLIP[0] * ema, LM_CLIP[1] * ema))
            a = min(dt / LM_TAU, 1.0)
            mean = float(np.exp((1 - a) * np.log(getattr(self, "lm_mean", ema)) + a * np.log(ema)))
            self.lm_ema, self.lm_mean = ema, mean
            self.lm_speed = SPEED_PRIOR * (ema / mean) ** LM_BETA
            return self.lm_speed
        if PARAMS.get("lm_open_loop"):
            self.lm_measured = step / dt
            return SPEED_PRIOR
        if LM_NORMALIZE_TAU is None:
            measured = float(np.clip(step / dt, LM_CLIP[0] * self.lm_speed, LM_CLIP[1] * self.lm_speed))
            self.lm_speed = (1 - LM_ALPHA) * self.lm_speed + LM_ALPHA * measured
            return self.lm_speed
        # relative speed only: raw estimate over its long-run geometric mean, times speed_prior (anchors the scale)
        raw = getattr(self, "lm_raw", step / dt)
        measured = float(np.clip(step / dt, LM_CLIP[0] * raw, LM_CLIP[1] * raw))
        raw = max((1 - LM_ALPHA) * raw + LM_ALPHA * measured, 1e-6)
        a = min(dt / float(LM_NORMALIZE_TAU), 1.0)
        mean = float(np.exp((1 - a) * np.log(getattr(self, "lm_mean", raw)) + a * np.log(raw)))
        self.lm_raw, self.lm_mean = raw, mean
        self.lm_speed = SPEED_PRIOR * raw / mean
        return self.lm_speed

    def _landmark_update(self):
        """After this frame's pose: record first observations of new tracks and (re)triangulate tracks whose bearing
        changed by at least lm_min_parallax_deg since their first observation (midpoint of the two rays)."""
        if self.next_ids is None or self.next_points is None or len(self.next_ids) == 0:
            return
        c = self._camera_centre()
        b = self._world_bearings(self.next_points.reshape(-1, 2).astype(float))
        alive = set()
        cos_gate = np.cos(np.radians(LM_MIN_PARALLAX_DEG))
        for k, i in enumerate(self.next_ids):
            i = int(i)
            alive.add(i)
            birth = self.lm_birth.get(i)
            if birth is None:
                self.lm_birth[i] = (b[k], c.copy())
                continue
            b0, c0 = birth
            if float(b0 @ b[k]) > cos_gate:
                continue
            # closest points of the rays c0 + a b0 and c + e b
            w0 = c0 - c
            a11, a12, a22 = 1.0, float(b0 @ b[k]), 1.0
            d1, d2 = float(b0 @ w0), float(b[k] @ w0)
            den = a11 * a22 - a12 * a12
            if den < 1e-9:
                continue
            a = (a12 * d2 - a22 * d1) / den
            e = (a11 * d2 - a12 * d1) / den
            if a <= 0 or e <= 0:
                continue
            self.lm_points[i] = 0.5 * ((c0 + a * b0) + (c + e * b[k]))
        for store in (self.lm_birth, self.lm_points):
            for i in list(store):
                if i not in alive:
                    del store[i]

    def _metric_push(self, t_mid, velocity, direction):
        self.metric_window.append((t_mid, velocity, direction))
        while self.metric_window and self.metric_window[0][0] < t_mid - METRIC_WINDOW_S:
            self.metric_window.pop(0)

    def _metric_fit(self):
        """Speed (m/s) at the newest frame from the sliding window; see speed_model "metric"."""
        rows = [(t, v, d) for t, v, d in self.metric_window if d is not None][-METRIC_MAX_DIRECTIONS:]
        m = len(rows)
        if m < METRIC_MIN_DIRECTIONS:
            return self.metric_speed
        t_now, v_now = rows[-1][0], rows[-1][1]
        extra = 6 if METRIC_BIAS else 3
        ridge = 3 if METRIC_BIAS and METRIC_BIAS_WEIGHT > 0 else 0
        A = np.zeros((3 * m + (m - 1) + m + ridge, m + extra))
        y = np.zeros(A.shape[0])
        for k, (t, v, d) in enumerate(rows):  # s_k d_k - c - b (t_k - t_now) = V_k - V_now
            r = slice(3 * k, 3 * k + 3)
            A[r, k] = d
            A[r, m:m + 3] = -np.eye(3)
            if METRIC_BIAS:
                A[r, m + 3:m + 6] = -(t - t_now) * np.eye(3)
            y[r] = v - v_now
        o = 3 * m
        idx = np.arange(m - 1)
        A[o + idx, idx] = -METRIC_SMOOTH
        A[o + idx, idx + 1] = METRIC_SMOOTH
        o += m - 1
        A[o + np.arange(m), np.arange(m)] = METRIC_PRIOR
        y[o:o + m] = METRIC_PRIOR * (SPEED_PRIOR if METRIC_ANCHOR == "prior" else self.metric_speed)
        if ridge:
            A[o + m + np.arange(3), m + 3 + np.arange(3)] = METRIC_BIAS_WEIGHT
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        s = float(np.clip(x[m - 1], *METRIC_CLIP))
        self.metric_speed = (1 - METRIC_ALPHA) * self.metric_speed + METRIC_ALPHA * s
        return self.metric_speed

    def _direction_multiframe(self):
        """World direction of motion from the frame baseline_frames ago to now, using tracks seen in both."""
        if len(self.history) < BASELINE_FRAMES or self.next_ids is None or self.next_points is None:
            return None
        ids_then, points_then, R_wc_then = self.history[-BASELINE_FRAMES][:3]
        if ids_then is None or points_then is None:
            return None
        common, i_then, i_now = np.intersect1d(ids_then, self.next_ids, return_indices=True)
        if len(common) < MIN_TRACKS:
            return None
        p0 = points_then[i_then].astype(float)
        p1 = self.next_points.reshape(-1, 2)[i_now].astype(float)
        if E_MAX_POINTS is not None and len(p0) > E_MAX_POINTS:
            keep = np.linspace(0, len(p0) - 1, int(E_MAX_POINTS)).round().astype(int)
            p0, p1 = p0[keep], p1[keep]
        R_wc_now = self.R_wb @ self.R_bc
        d = self._direction_known_rotation(p0, p1, R_wc_now.T @ R_wc_then)
        if d is None:
            return None
        d = R_wc_then @ d
        return d / max(np.linalg.norm(d), 1e-12)

    def _common_tracks(self, entry):
        ids_then, points_then = entry[0], entry[1]
        if ids_then is None or points_then is None or self.next_ids is None or self.next_points is None:
            return None
        common, i_then, i_now = np.intersect1d(ids_then, self.next_ids, return_indices=True)
        if len(common) < MIN_TRACKS:
            return None
        return points_then[i_then].astype(float), self.next_points.reshape(-1, 2)[i_now].astype(float)

    def _update_gyro_bias(self):
        """Compare the gyro rotation over bias_baseline_frames with the rotation that best explains the tracks seen in
        both views (translation direction and a small rotation correction delta, alternately), and move the gyro bias
        toward delta / elapsed time. delta is in camera-1 axes: R_true = exp(delta) R_gyro, so the bias error is R_bc delta / T."""
        if len(self.history) < BIAS_BASELINE:
            return
        entry = self.history[-BIAS_BASELINE]
        pair = self._common_tracks(entry)
        if pair is None:
            return
        p0, p1 = pair
        R_wc_then, t_then = entry[2], entry[3]
        elapsed = self.last_t - t_then
        if elapsed <= 0:
            return
        R = (self.R_wb @ self.R_bc).T @ R_wc_then
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        n = len(p0)
        x0 = np.column_stack([(p0[:, 0] - cx) / fx, (p0[:, 1] - cy) / fy, np.ones(n)])
        x1 = np.column_stack([(p1[:, 0] - cx) / fx, (p1[:, 1] - cy) / fy, np.ones(n)])
        threshold = KR_THRESHOLD_PX * SCALE / fx
        delta = np.zeros(3)
        for _ in range(2):
            Rc = cv2.Rodrigues(delta)[0] @ R
            d = self._direction_known_rotation(p0, p1, Rc)
            if d is None:
                return
            t = -Rc @ d
            r0 = x0 @ Rc.T
            A = np.cross(r0, x1)
            lines = np.cross(t, r0)
            w = 1.0 / np.maximum(np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2), 1e-12)
            mask = np.abs(A @ t) * w < 3 * threshold
            if int(mask.sum()) < MIN_INLIERS:
                return
            g = np.cross(r0, np.cross(x1, t)) * w[:, None]
            e = (A @ t) * w
            G, E = g[mask], e[mask]
            step = -np.linalg.solve(G.T @ G + BIAS_RIDGE * np.eye(3), G.T @ E)
            delta = delta + step
        error = self.R_bc @ delta / elapsed
        cap = np.radians(BIAS_MAX_STEP)
        update = np.clip(BIAS_GAIN * error, -cap, cap)
        self.gyro_bias = self.gyro_bias + update
        if BIAS_ATTITUDE_GAIN > 0:
            self.R_wb = self.R_wb @ cv2.Rodrigues(-BIAS_ATTITUDE_GAIN * (self.R_bc @ delta) / BIAS_BASELINE)[0]

    def _correct_tilt(self, imu, dt):
        """Rotate R_wb by a small world-frame rotation moving the specific force of this frame toward +z."""
        f = self.R_wb @ imu[:, 1:4].mean(axis=0)
        norm = np.linalg.norm(f)
        if norm < 1e-6:
            return
        axis = np.cross(f / norm, np.array([0.0, 0.0, 1.0]))
        angle = min(dt / float(TILT_TAU), 1.0) * float(np.arcsin(min(np.linalg.norm(axis), 1.0)))
        if np.linalg.norm(axis) > 1e-12:
            self.R_wb = cv2.Rodrigues(axis / np.linalg.norm(axis) * angle)[0] @ self.R_wb
