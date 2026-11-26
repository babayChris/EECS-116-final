"""Simulate stage-3-like episodes: agile flights that start in motion, FoV U(75, 135) deg, motion blur U(0, 50) ms.

A local guess at the stage 3 setting (the TA announcement), not the real generator:
  * flights: scripts/stage3_autopilot.patch (faster autopilot) must be applied to the pinned viobench first;
    make_stage3_episodes.sh does that and reverts it afterwards
  * start in motion: `--crop` seconds are flown first and then removed from the episode (frames, IMU, ground truth)
  * FoV: nominal 105 deg +- 30 deg, fixed per episode
  * motion blur: exposure U(0, 50) ms per episode; a frame is the mean, in linear light, of the 200 Hz renders inside
    the exposure (the simulator renders every IMU step for this purpose; frame times move to mid-exposure upstream)
Warehouse and forest scenes are not in the public scene catalogs, so only ProcTHOR / HSSD scenes are available.

    python scripts/make_stage3_episodes.py -o work/data/s3/a --scene ProcTHOR-Train-7 --episodes 4 --seed 3100 --duration 45
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from viobench.tools.datasets.episode import Episode, FrameSequence
from viobench.tools.datasets.generate.simulation import SimConfig, scene_references, simulate
from viobench.tools.datasets.storage import write_episode

MAX_EXPOSURE = 0.050


def _to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _to_srgb(linear):
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(np.maximum(linear, 0), 1 / 2.4) - 0.055)


def blur_camera(rng, imu_rate_hz):
    exposure = float(rng.uniform(0.0, MAX_EXPOSURE))
    k = max(1, int(round(exposure * imu_rate_hz)))

    def develop(renders, _rng):
        if len(renders) == 1:
            image = renders[-1]
        else:
            image = _to_srgb(_to_linear(renders.astype(np.float32)).mean(axis=0))
        return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)

    return k, develop, {"exposure_drawn_seconds": exposure}


def crop(episode, seconds):
    """Drops the first `seconds` of flight: the remaining episode starts in motion."""
    if seconds <= 0:
        return episode
    start = int(np.searchsorted(episode.frame_times, episode.frame_times[0] + seconds))
    interval = float(np.median(np.diff(episode.frame_times)))
    t0 = episode.frame_times[start]
    frames = episode.frames
    count = len(frames) - start
    sequence = FrameSequence(lambda i, _f=frames: _f[start + i], count, frames.frame_shape,
                             encoded=None if frames.encoded is None else (lambda i, _f=frames: _f.encoded(start + i)))
    imu = episode.imu[episode.imu[:, 0] > t0 - interval + 1e-9]
    gt = episode.ground_truth
    keep = gt[:, 0] >= t0 - interval - 1e-9
    config = {**episode.config, "episode": {**episode.config["episode"], "num_frames": count, "hover_seconds": 0.0,
                                             "cropped_seconds": float(t0 - episode.frame_times[0])}}
    return Episode(name=episode.name, config=config, frame_times=episode.frame_times[start:], frames=sequence, imu=imu,
                   ground_truth=gt[keep], T_body_camera=episode.T_body_camera,
                   meta={**episode.meta, "cropped_seconds": float(t0 - episode.frame_times[0])})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--scene", nargs="+", default=["ProcTHOR-Train-1"])
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--duration", type=float, default=45.0, help="seconds kept per episode")
    p.add_argument("--crop", type=float, default=5.0, help="seconds flown before the episode starts")
    p.add_argument("--fov", type=float, default=105.0)
    p.add_argument("--fov-range", type=float, default=30.0)
    p.add_argument("--spool", default=None)
    a = p.parse_args(argv)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    total = a.duration + a.crop
    sim = SimConfig(instances=min(4, a.episodes), duration_seconds=total, waypoints=0, fov_range=a.fov_range,
                    hold_seconds=0.0, spool=a.spool, fov=a.fov)
    written = 0
    for episode in simulate(sim, scene_references(a.scene), a.episodes, seed=a.seed, camera=blur_camera, name="s3"):
        episode = crop(episode, a.crop)
        path = write_episode(episode, out / f"{episode.name}-{a.seed}.zip")
        gt = episode.ground_truth
        dist = float(np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1).sum())
        dur = gt[-1, 0] - gt[0, 0]
        print(f"wrote {path}: {episode.num_frames} frames, fov {episode.config['camera']['horizontal_fov_degrees']:.1f}, "
              f"exposure {episode.config['episode'].get('exposure_seconds', 0) * 1000:.0f} ms, path {dist:.1f} m in {dur:.1f} s "
              f"({dist / max(dur, 1e-9):.2f} m/s)", flush=True)
        written += 1
    print(f"{written} episodes in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
