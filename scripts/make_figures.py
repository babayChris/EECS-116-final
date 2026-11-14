"""Figures for the write-up, from the recorded results (results/klt-bounded/*.json) and local per-frame fuel logs.

    .venv-vio/bin/python scripts/make_figures.py       # writes results/figures/*.png

Static PNGs for a document. Colours: categorical slots of the dataviz reference palette (blue = our estimator,
orange = upstream examples, gray = context), text in neutral ink, recessive grid.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "klt-bounded"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8d8b85"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False,
})

summary = json.loads((RES / "summary.json").read_text())
held = json.loads((RES / "heldout.json").read_text())
host = json.loads((RES / "host_study.json").read_text())
V, B, N = summary["variants"], summary["baselines"], summary["native_baselines"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


# 1. Accuracy vs peak fuel (public set) ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.9))
upstream = [("Constant pose", B["minimal"]), ("Dead reckoning", B["dead_reckoning"]), ("KLT + IMU", B["klt_imu"]),
            ("VINS-Mono", N["vins_mono"]), ("OpenVINS", N["openvins"])]
ours = [("Stage 1 (v23)", V["v23_quarter_c40"]), ("Stage 2 (v56)", V["v56_cs_par025_coast"]),
        ("Persistent tracks (e3)", V["e3_persist05"]), ("+ 3-frame baseline (e5)", V["e5_p05_base3"])]
pts = sorted((d["fuel_per_frame_max"], d["sim3_ate_mean_m"]) for _, d in upstream)
stair_x, stair_y, best = [], [], None
for x, y in pts:
    if best is None or y < best:
        if best is not None:
            stair_x.append(x); stair_y.append(best)
        stair_x.append(x); stair_y.append(y); best = y
stair_x.append(3e10); stair_y.append(best)
ax.plot(stair_x, stair_y, color=GRAY, lw=1.5, ls="--", zorder=1, label="Upstream frontier")
ax.scatter([d["fuel_per_frame_max"] for _, d in upstream], [d["sim3_ate_mean_m"] for _, d in upstream], s=70,
           color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3, label="Upstream examples")
ax.scatter([d["fuel_per_frame_max"] for _, d in ours], [d["sim3_ate_mean_m"] for _, d in ours], s=70, marker="D",
           color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4, label="Our estimator")
offsets = {"Constant pose": (8, -14), "Dead reckoning": (-8, 8), "KLT + IMU": (8, 8), "VINS-Mono": (-6, 10), "OpenVINS": (-8, -16),
           "Stage 1 (v23)": (-10, -16), "Stage 2 (v56)": (-10, 8), "Persistent tracks (e3)": (-10, -14), "+ 3-frame baseline (e5)": (10, -6)}
for name, d in upstream + ours:
    dx, dy = offsets[name]
    ax.annotate(f"{name}  {d['sim3_ate_mean_m']:.2f} m", (d["fuel_per_frame_max"], d["sim3_ate_mean_m"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=9, color=INK, ha="right" if dx < 0 else "left")
ax.set_xscale("log"); ax.set_xlim(1e5, 2e10); ax.set_ylim(0, 3.5)
ax.set_xlabel("Peak fuel per frame (log scale, lower is better)"); ax.set_ylabel("Mean Sim(3) ATE, m (lower is better)")
ax.set_title("Accuracy against peak compute, 10 public episodes")
ax.legend(loc="upper right", fontsize=9)
save(fig, "fig1_pareto.png")

# 2. Step-by-step attribution ----------------------------------------------------------------------------------------
steps = [("Upstream KLT + IMU", B["klt_imu"]), ("Stage 1: bounded compute (v23)", V["v23_quarter_c40"]),
         ("+ known-rotation direction (v32)", V["v32_s2ab"]), ("+ constant-speed fusion (v43)", V["v43_s2_constspeed"]),
         ("+ coast without vision (v46)", V["v46_cs_coast"]), ("+ 0.25 px parallax gate (v56)", V["v56_cs_par025_coast"]),
         ("+ persistent tracks (e3)", V["e3_persist05"]), ("+ 3-frame baseline (e5)", V["e5_p05_base3"])]
fig, ax = plt.subplots(figsize=(8.6, 4.2))
y = np.arange(len(steps))[::-1]
vals = [d["sim3_ate_mean_m"] for _, d in steps]
ax.barh(y, vals, height=0.62, color=[ORANGE] + [BLUE] * (len(steps) - 1), edgecolor=SURFACE, linewidth=2)
for yi, (name, d) in zip(y, steps):
    ax.text(d["sim3_ate_mean_m"] + 0.03, yi, f"{d['sim3_ate_mean_m']:.2f} m  ·  peak {d['fuel_per_frame_max'] / 1e6:.0f} M fuel",
            va="center", fontsize=9, color=INK)
ax.set_yticks(y, [s for s, _ in steps]); ax.set_xlim(0, 3.0); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean Sim(3) ATE, m (public set)")
ax.set_title("Each change, added one at a time")
save(fig, "fig2_attribution.png")

# 3. Public vs held-out --------------------------------------------------------------------------------------------
rows = [("Constant pose", B["minimal"], "minimal"), ("Dead reckoning", B["dead_reckoning"], "dead_reckoning"),
        ("KLT + IMU", B["klt_imu"], "klt_imu"), ("Stage 1 (v23)", V["v23_quarter_c40"], "v23_quarter_c40"),
        ("Stage 2 (v56)", V["v56_cs_par025_coast"], "v56_cs_par025_coast"), ("Persistent (e3)", V["e3_persist05"], "e3_persist05"),
        ("+ FB check (e4)", V["e4_p05_fb1"], "e4_p05_fb1"), ("+ 3-frame (e5)", V["e5_p05_base3"], "e5_p05_base3")]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
x = np.arange(len(rows)); w = 0.38
pub = [d["sim3_ate_mean_m"] for _, d, _ in rows]
ho = [held["approaches"][k]["sim3_ate_mean_m"] for _, _, k in rows]
ax.bar(x - w / 2 - 0.01, pub, w, color=BLUE, edgecolor=SURFACE, linewidth=2, label="Public (10 episodes)")
ax.bar(x + w / 2 + 0.01, ho, w, color=AQUA, edgecolor=SURFACE, linewidth=2, label="Held-out (8 episodes, not used for tuning)")
for xi, a, b in zip(x, pub, ho):
    ax.text(xi - w / 2, a + 0.04, f"{a:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.text(xi + w / 2, b + 0.04, f"{b:.2f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x, [r[0] for r in rows], rotation=18, ha="right"); ax.grid(axis="x", visible=False)
ax.set_ylabel("Mean Sim(3) ATE, m"); ax.set_ylim(0, 3.6)
ax.set_title("Generalization: public set against held-out episodes")
ax.legend(loc="upper right", fontsize=9)
save(fig, "fig3_generalization.png")

# 4. Per-frame fuel on the hardest public episode ------------------------------------------------------------------
episode = "60ed4323f3aab9c83a854f30062cb4ffbefeac12"
traces = [("Upstream KLT + IMU", ROOT / "outputs/local-benchmark/klt_imu-wasm", ORANGE),
          ("Stage 1 (v23)", ROOT / "outputs/variants/v23_quarter_c40-wasm", GRAY),
          ("Persistent tracks (e3)", ROOT / "outputs/variants/e3_persist05-wasm", BLUE)]
fig, ax = plt.subplots(figsize=(8.6, 3.8))
for name, d, c in traces:
    f = np.load(d / "logs" / episode / "fuel.npy") / 1e6
    ax.plot(np.arange(len(f)) / 20.0, f, color=c, lw=1.2, label=f"{name}: peak {f.max():.0f} M, median {np.median(f[12:]):.0f} M")
ax.set_yscale("log"); ax.set_ylim(0.5, 400)
ax.set_xlabel("Episode time, s"); ax.set_ylabel("Fuel per frame, millions (log)")
ax.set_title("Per-frame fuel on public episode 60ed4323")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9, ncol=1)
save(fig, "fig4_fuel_trace.png")

# 5. Direction accuracy ---------------------------------------------------------------------------------------------
# From scripts/direction_accuracy.py on episodes 9b6a9279 and 60ed4323 (v23 = essential matrix, v32 = known rotation).
labels = ["Episode 9b6a9279", "Episode 60ed4323"]
essential_err, known_err = [50.2, 45.7], [12.2, 11.9]
essential_n, known_n = [90, 191], [240, 247]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
x = np.arange(2); w = 0.38
for ax, (a, b, title, unit) in zip(axes, [(essential_err, known_err, "Median direction error", "degrees"),
                                          (essential_n, known_n, "Frames with a direction", "frames")]):
    ax.bar(x - w / 2 - 0.01, a, w, color=ORANGE, edgecolor=SURFACE, linewidth=2, label="5-point essential matrix")
    ax.bar(x + w / 2 + 0.01, b, w, color=BLUE, edgecolor=SURFACE, linewidth=2, label="Known rotation (ours)")
    for xi, va, vb in zip(x, a, b):
        ax.text(xi - w / 2, va * 1.02, f"{va:g}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + w / 2, vb * 1.02, f"{vb:g}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, labels); ax.set_ylabel(unit); ax.set_title(title); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(a + b) * 1.18)
axes[0].legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
save(fig, "fig5_direction.png")

# 6. Study: experimental options against the seed-noise band -----------------------------------------------------------
H = host["variants"]
seeds = [H[k] for k in ["v56_cs_par025_coast", "s56_seed1", "s56_seed2", "s56_seed3", "s56_seed4"]]
study = [("v56 (reference)", "v56_cs_par025_coast"), ("E1 flow speed, α=1", "e1_flow_a1"), ("E1 flow speed, α=0.5", "e1_flow_a05"),
         ("E2 tilt correction, 5 s", "e2_tilt5"), ("E3 persistent tracks", "e3_persist05"), ("E4 + FB check 1 px", "e4_p05_fb1"),
         ("E5 + 3-frame baseline", "e5_p05_base3"), ("E5 + 5-frame baseline", "e5_p05_base5")]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
y = np.arange(len(study))[::-1]
for tag, color, mk, label, dy in [("public", BLUE, "o", "Public", 0.13), ("heldout", AQUA, "s", "Held-out ProcTHOR (7)", -0.13)]:
    lo = min(s[f"{tag}_sim3_ate_mean_m"] for s in seeds); hi = max(s[f"{tag}_sim3_ate_mean_m"] for s in seeds)
    ax.axvspan(lo, hi, color=color, alpha=0.12, lw=0, label=f"v56 seed range, {label.split(' ')[0].lower()}")
    ax.scatter([H[k][f"{tag}_sim3_ate_mean_m"] for _, k in study], y + dy, s=64, marker=mk, color=color, edgecolor=SURFACE,
               linewidth=2, zorder=3, label=label)
ax.set_yticks(y, [s for s, _ in study]); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean Sim(3) ATE, m (host screening)"); ax.set_xlim(0.4, 1.1)
ax.set_title("Experimental options, with run-to-run noise shaded")
ax.legend(loc="lower right", fontsize=9)
save(fig, "fig6_study.png")
