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
                   (derotated image motion / its running average) ** speed_flow_alpha, clipped to speed_flow_clip
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
PROBE = PARAMS.get("probe")  # measurement only: "imu_only" skips all vision work (fuel floor of the rest)
SPEED_FEATURES = bool(PARAMS.get("speed_features", False)) or SPEED_MODEL == "learned"
SPEED_MODEL_SPEC = None
if SPEED_MODEL == "learned":  # weights written by scripts/speed_fit.py --export, bundled next to this file
    SPEED_MODEL_SPEC = json.loads(_bundle.read_text("speed_model.json"))
    for _key in ("mean", "std", "weights"):
        SPEED_MODEL_SPEC[_key] = np.asarray(SPEED_MODEL_SPEC[_key], dtype=float)
SPEED_EMA_TAUS = (0.5, 2.0)  # seconds; running averages of the speed features
SPEED_EMA_KEYS = ("flow_med", "flow_p75", "gyro_norm", "acc_x", "acc_y", "acc_z", "acc_std")
SPEED_FEATURE_NAMES = (("has_flow", "flow_med", "flow_p25", "flow_p75", "track_fraction", "has_direction", "gyro_norm",
                        "acc_x", "acc_y", "acc_z", "acc_norm_minus_g", "acc_std", "gyro_std")
                       + tuple(f"ema{tau}_{key}" for tau in SPEED_EMA_TAUS for key in SPEED_EMA_KEYS))


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
        self.flow_magnitudes = None  # derotated image motion per track of this frame's one-frame solve (normalised units)
        self.flow_tracks = 0
        self.speed_ema = {}  # (tau, key) -> running average
        self.speed_features = None  # this frame's feature vector (SPEED_FEATURE_NAMES), when SPEED_FEATURES
        if SPEED_MODEL == "learned":
            self.last_speed = SPEED_MODEL_SPEC["train_mean_speed"]
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
        magnitudes = np.sqrt((derotated * derotated).sum(axis=1))
        flow = float(np.median(magnitudes))
        self.last_flow = flow
        if self.flow_magnitudes is None:  # keep the one-frame solve; the multi-frame solve runs after it
            self.flow_magnitudes = magnitudes
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
        self.flow_tracks = len(p0)
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
        displacement_imu = self._integrate_vectorized(imu) if IMU_VECTORIZED else self._integrate(imu)
        R_c1_c0 = (self.R_wb @ self.R_bc).T @ (R_wb_before @ self.R_bc)
        self.last_flow = None
        self.flow_magnitudes, self.flow_tracks = None, 0
        direction_camera = None if PROBE == "imu_only" else self._track(frame, R_c1_c0)
        dt = max(imu[-1, 0] - imu[0, 0] + (imu[1, 0] - imu[0, 0] if len(imu) > 1 else 0.0), 1e-3) if len(imu) else 0.05
        if SPEED_FEATURES:
            self.speed_features = self._speed_features(imu, dt, direction_camera is not None)
            if SPEED_MODEL == "learned":
                self.last_speed = self._learned_speed(self.speed_features)  # also used while coasting
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
                speed = self.last_speed if SPEED_MODEL == "learned" else SPEED_PRIOR
                if SPEED_MODEL == "flow" and self.last_flow is not None and self.last_flow > 0:
                    rate = self.last_flow / dt
                    if self.flow_average is None:
                        self.flow_average = rate
                    else:
                        a = min(dt / SPEED_FLOW_TAU, 1.0)
                        self.flow_average = float(np.exp((1 - a) * np.log(self.flow_average) + a * np.log(rate)))
                    speed = SPEED_PRIOR * float(np.clip((rate / self.flow_average) ** SPEED_FLOW_ALPHA, *SPEED_FLOW_CLIP))
                    self.last_speed = speed
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
        current_ids, current_points = self.next_ids, self.next_points
        self.previous_frame = frame.copy()
        if PROBE != "imu_only":
            self.previous_points = self._replenish(frame) if TRACK_MODE == "persist" else self._detect(frame)
        if BASELINE_FRAMES > 1:
            self.history.append((self.ids, None if self.previous_points is None else self.previous_points.reshape(-1, 2).copy(),
                                 self.R_wb @ self.R_bc))
            del self.history[:-BASELINE_FRAMES]
        return self._state()

    def _speed_features(self, imu, dt, has_direction):
        """Causal per-frame inputs of the learned speed model, in SPEED_FEATURE_NAMES order.

        Vision: derotated image motion of this frame's tracks as an angular rate (normalised image units per second;
        roughly speed / depth). IMU: specific force in IMU axes (on a multirotor its x/y part is mostly rotor drag,
        which grows with body-frame velocity), its fluctuation, and the gyro rate. Running averages smooth both
        over SPEED_EMA_TAUS; the flow averages only update on frames that have a flow measurement."""
        m = self.flow_magnitudes
        has_flow = m is not None and len(m) > 0
        if has_flow:
            flow_med, flow_p25, flow_p75 = (float(v) / dt for v in np.percentile(m, (50, 25, 75)))
        else:
            flow_med = flow_p25 = flow_p75 = 0.0
        if len(imu):
            acc, gyro = imu[:, 1:4], imu[:, 4:7] - self.gyro_bias
            acc_norm = np.sqrt((acc * acc).sum(axis=1))
            gyro_norm_samples = np.sqrt((gyro * gyro).sum(axis=1))
            acc_mean = acc.mean(axis=0)
            acc_std, acc_norm_mean = float(acc_norm.std()), float(acc_norm.mean())
            gyro_norm, gyro_std = float(gyro_norm_samples.mean()), float(gyro_norm_samples.std())
        else:
            acc_mean, acc_std, acc_norm_mean, gyro_norm, gyro_std = np.array([0.0, 0.0, GRAVITY]), 0.0, GRAVITY, 0.0, 0.0
        current = {"flow_med": flow_med, "flow_p75": flow_p75, "gyro_norm": gyro_norm, "acc_x": float(acc_mean[0]),
                   "acc_y": float(acc_mean[1]), "acc_z": float(acc_mean[2]), "acc_std": acc_std}
        averages = []
        for tau in SPEED_EMA_TAUS:
            a = min(dt / tau, 1.0)
            for key in SPEED_EMA_KEYS:
                if key.startswith("flow") and not has_flow:
                    value = self.speed_ema.get((tau, key), 0.0)
                else:
                    previous = self.speed_ema.get((tau, key))
                    value = current[key] if previous is None else previous + a * (current[key] - previous)
                self.speed_ema[(tau, key)] = value
                averages.append(value)
        return np.array([float(has_flow), flow_med, flow_p25, flow_p75, self.flow_tracks / MAX_CORNERS, float(has_direction),
                         gyro_norm, current["acc_x"], current["acc_y"], current["acc_z"], acc_norm_mean - GRAVITY, acc_std,
                         gyro_std] + averages)

    def _learned_speed(self, features):
        """speed_model "learned": speed (m/s) from a linear model on the speed features, as exported to speed_model.json
        by scripts/speed_fit.py --export. Each model input is a logged feature, the log of one (plus log_eps) or the
        hypotenuse of two; the prediction is clipped at zero and shrunk toward the training mean speed."""
        m = SPEED_MODEL_SPEC
        x = np.empty(len(m["columns"]))
        for c, (op, i, j) in enumerate(m["columns"]):
            if op == "raw":
                x[c] = features[i]
            elif op == "log":
                x[c] = np.log(features[i] + m["log_eps"])
            else:  # "hypot"
                x[c] = np.hypot(features[i], features[j])
        q = max(float(((x - m["mean"]) / m["std"]) @ m["weights"] + m["intercept"]), 0.0)
        return m["shrink"] * q + (1.0 - m["shrink"]) * m["train_mean_speed"]

    def _direction_multiframe(self):
        """World direction of motion from the frame baseline_frames ago to now, using tracks seen in both."""
        if len(self.history) < BASELINE_FRAMES or self.next_ids is None or self.next_points is None:
            return None
        ids_then, points_then, R_wc_then = self.history[0]
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
