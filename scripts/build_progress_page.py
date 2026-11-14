"""Build the progress page (one self-contained HTML file) from local WASM evaluation reports.

    .venv-vio/bin/python scripts/build_progress_page.py      # writes outputs/progress/index.html

Numbers come from outputs/local-benchmark/<baseline>-wasm/report.json (upstream examples, the stand-in baseline set)
and outputs/variants/<name>-wasm/report.json (our variants). The improvement score is the upstream server's formula
(viobench.tools.server.evaluations.dominated_area) applied to those local points. STATUS and NOTES are maintained by
hand: update them when the state of the project changes.
"""
import html
import json
import math
from datetime import datetime
from pathlib import Path

from viobench.tools.server.evaluations import ERROR_FLOOR, FUEL_FLOOR, dominated_area

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs" / "local-benchmark"
VARIANTS = ROOT / "outputs" / "variants"
VARIANT_SPEC = ROOT / "solutions" / "klt_bounded" / "variants.json"
OUT = ROOT / "outputs" / "progress" / "index.html"
HELDOUT = ROOT / "outputs" / "heldout"

BASELINES = [  # (report directory name, label, colour token, description)
    ("minimal", "Minimal (constant pose)", "--c-min", "Always outputs the identity pose."),
    ("dead_reckoning", "Dead reckoning (IMU only)", "--c-dr", "Integrates accelerometer and gyroscope; no vision."),
    ("klt_imu", "KLT + IMU", "--c-klt", "Upstream baseline: KLT tracks, essential matrix for direction, IMU for rotation and scale."),
    ("openvins", "OpenVINS (native)", "--c-ov", "Upstream native example: OpenVINS 2.7 frontends, fixed-lag Ceres optimization, consensus."),
    ("vins_mono", "VINS-Mono (native)", "--c-vm", "Upstream native example: sliding-window Ceres optimization with IMU preintegration."),
    ("orbslam3", "ORB-SLAM3 (native)", "--c-orb", "Upstream native example: ORB features, inertial initialization, local bundle adjustment."),
]
NATIVE = ROOT / "outputs" / "native"

# status: done | running | blocked | todo
STATUS = [
    ("done", "Baselines in the WASM sandbox", "minimal, dead reckoning and KLT + IMU on all 10 public episodes; OpenVINS and VINS-Mono native examples built and measured (VINS-Mono failed partway through 1 episode). ORB-SLAM3 built and passed check; its evaluation was stopped before finishing."),
    ("done", "Stage 1: bounded compute", "Peaks come from essential-matrix RANSAC and first-use costs. Quarter resolution, binding RANSAC limits and a full warm-up in init: 2.003 m at 35.7 M (KLT: 2.042 m at 142 M)."),
    ("done", "Stage 2: accuracy at the bounded budget", "Known-rotation translation direction (12° vs 46–50° median error) and constant-speed fusion with coasting and a 0.25 px gate: 0.637 m at 14.7 M."),
    ("done", "Evaluation environment check", "The server runs the same viobench evaluate command with the same pinned wasmtime, and the upload carries our runtime: results reproduce for the same episodes and limits. Unknown: the server's episode selection (a public+private-10 preset exists) and limit changes."),
    ("done", "Generalization check, 8 held-out episodes", "7 new ProcTHOR episodes and 1 HSSD episode, simulated with the CPU renderer and never used for tuning. Gains hold; 2 HSSD episodes were stopped (CPU renderer memory and time)."),
    ("done", "Broader study", "Noise floor (seed spread ±0.02 m), E1 flow-based speed (worse), E2 tilt correction (no effect), E3 persistent tracks (better and cheaper), E4 forward-backward check and E5 3-frame baseline (small gains), fuel probe (4.5 M per frame is Python/IMU/state, not vision)."),
    ("todo", "Next", "Resolution × corner sweep on the combined best (defined as w_* in variants.json, not run); native C++ port to remove the 4.5 M Python floor; finish the HSSD held-out episodes on a GPU machine; get the server's /config and submit."),
    ("blocked", "tiny_cnn training", "GPU simulation needs OptiX, unavailable under WSL. The CPU renderer works but is far too slow for a training set."),
]

NOTES = []  # filled in by hand after reading the results; see NOTES_FILE
NOTES_FILE = ROOT / "outputs" / "progress" / "notes.json"


def load_report(path):
    report = json.loads(path.read_text())
    return report, datetime.fromtimestamp(path.stat().st_mtime)


def load():
    rows = []
    for key, label, colour, desc in BASELINES:
        path = BENCH / f"{key}-wasm" / "report.json"
        if not path.exists():
            path = NATIVE / f"{key}-wasm" / "report.json"
        if path.exists():
            report, mtime = load_report(path)
            rows.append({"key": key, "label": label, "colour": colour, "desc": desc, "kind": "baseline",
                         "report": report, "summary": report["summary"], "mtime": mtime})
    spec = json.loads(VARIANT_SPEC.read_text()) if VARIANT_SPEC.exists() else {}
    for name, params in spec.items():
        path = VARIANTS / f"{name}-wasm" / "report.json"
        if name.startswith("_") or name.startswith("probe") or not path.exists():
            continue
        report, mtime = load_report(path)
        desc = ", ".join(f"{k}={v}" for k, v in params.items()) or "upstream settings (reproduction check)"
        rows.append({"key": name, "label": name, "colour": "--c-var", "desc": desc, "kind": "variant",
                     "report": report, "summary": report["summary"], "mtime": mtime})
    return rows


def point(r):
    return (max(r["summary"]["fuel_per_frame_max"], FUEL_FLOOR), max(r["summary"]["sim3_ate_rmse_mean"], ERROR_FLOOR))


def frontier(rows):
    keep = []
    for a in rows:
        fa, ea = point(a)
        if not any(point(b)[0] <= fa and point(b)[1] <= ea and point(b) != (fa, ea) for b in rows):
            keep.append(a)
    return sorted(keep, key=lambda r: point(r)[0])


def scoring(rows, step_fuel):
    """The server's improvement score: extra dominated log-area beyond the baselines, as % of the reference box."""
    base = [point(r) for r in rows if r["kind"] == "baseline"]
    if not base:
        return None
    fuel_ref, err_ref = 1.2 * step_fuel, 1.2 * max(e for _, e in base)
    box = math.log(fuel_ref / FUEL_FLOOR) * math.log(err_ref / ERROR_FLOOR)
    base_area = dominated_area(base, fuel_ref, err_ref)
    pct = lambda pts: 100 * max(0.0, dominated_area(base + pts, fuel_ref, err_ref) - base_area) / box
    return {"all": pct([point(r) for r in rows if r["kind"] == "variant"]),
            "each": {r["key"]: pct([point(r)]) for r in rows if r["kind"] == "variant"}}


def fmt_fuel(x):
    return f"{x / 1e9:.2f} G" if x >= 1e9 else f"{x / 1e6:.1f} M" if x >= 1e6 else f"{x / 1e3:.0f} k"


def staircase(points, X, Y, right, top):
    """Polygon of the region dominated by points (up and to the right), clipped to the plot."""
    pts = sorted(points)
    best, poly = None, []
    for f, e in pts:
        if best is not None and e >= best:
            continue
        if best is None:
            poly.append((X(f), top))
        else:
            poly.append((X(f), Y(best)))
        poly.append((X(f), Y(e)))
        best = e
    poly += [(right, Y(best)), (right, top)]
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in poly)


def pareto_svg(rows, limit):
    W, H, L, R, T, B = 760, 400, 64, 24, 20, 48
    ys = [r["summary"]["sim3_ate_rmse_mean"] for r in rows]
    x0, x1 = 1e5, 10 ** math.ceil(math.log10(limit * 1.2))
    y0, y1 = math.floor(min(ys) * 2) / 2 - 0.5, math.ceil(max(ys) * 2) / 2 + 0.25
    y0 = max(0.0, y0)
    X = lambda v: L + (math.log10(max(v, x0)) - math.log10(x0)) / (math.log10(x1) - math.log10(x0)) * (W - L - R)
    Y = lambda v: T + (1 - (min(v, y1) - y0) / (y1 - y0)) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean Sim(3) ATE against maximum fuel per frame, log scale">']
    allp = [(r["summary"]["fuel_per_frame_max"], r["summary"]["sim3_ate_rmse_mean"]) for r in rows]
    basep = [p for p, r in zip(allp, rows) if r["kind"] == "baseline"]
    if any(r["kind"] == "variant" for r in rows):
        out.append(f'<polygon class="gain" points="{staircase(allp, X, Y, W - R, T)}"/>')
    out.append(f'<polygon class="base" points="{staircase(basep, X, Y, W - R, T)}"/>')
    for e in range(int(math.log10(x0)), int(math.log10(x1)) + 1):
        v = 10 ** e
        out.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - B}"/>')
        out.append(f'<text class="tick" x="{X(v):.1f}" y="{H - B + 18}" text-anchor="middle">{fmt_fuel(v)}</text>')
    v = y0
    while v <= y1 + 1e-9:
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        out.append(f'<text class="tick" x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.1f} m</text>')
        v += 0.5
    out.append(f'<line class="limit" x1="{X(limit):.1f}" x2="{X(limit):.1f}" y1="{T}" y2="{H - B}"/>')
    out.append(f'<text class="tick" x="{X(limit) - 6:.1f}" y="{T + 12}" text-anchor="end">step limit {fmt_fuel(limit)}</text>')
    front = {r["key"] for r in frontier(rows)}
    for r in sorted(rows, key=lambda r: r["kind"] == "baseline"):
        f, e = r["summary"]["fuel_per_frame_max"], r["summary"]["sim3_ate_rmse_mean"]
        x, y = X(f), Y(e)
        if r["kind"] == "baseline":
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" style="fill:var({r["colour"]})"><title>{html.escape(r["label"])}: {e:.3f} m, {fmt_fuel(f)}</title></circle>')
            out.append(f'<text class="pt" x="{x + 10:.1f}" y="{y - 10:.1f}">{html.escape(r["label"].split(" (")[0])}</text>')
        else:
            cls = "var on" if r["key"] in front else "var"
            out.append(f'<rect class="{cls}" x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" transform="rotate(45 {x:.1f} {y:.1f})"><title>{html.escape(r["label"])}: {e:.3f} m, {fmt_fuel(f)}</title></rect>')
            if r["key"] in front:
                out.append(f'<text class="pt small" x="{x + 9:.1f}" y="{y + 16:.1f}">{html.escape(r["label"].split("_")[0])}</text>')
    out.append(f'<text class="axis" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">max fuel per frame, log scale (lower is better)</text>')
    out.append(f'<text class="axis" transform="translate(16 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">mean Sim(3) ATE (lower is better)</text>')
    out.append("</svg>")
    return "\n".join(out)


def build():
    rows = load()
    if not rows:
        raise SystemExit("No WASM reports found.")
    limit = rows[0]["report"]["limits"]["step_fuel"]
    score = scoring(rows, limit)
    front = {r["key"] for r in frontier(rows)}
    base_front = {r["key"] for r in frontier([r for r in rows if r["kind"] == "baseline"])}
    variants = [r for r in rows if r["kind"] == "variant"]
    updated = max(r["mtime"] for r in rows)
    klt = next((r for r in rows if r["key"] == "klt_imu"), None)

    def row_html(r):
        s = r["summary"]
        if r["kind"] == "baseline":
            chip = "<span class=chip-muted>baseline</span>" if r["key"] in base_front else "<span class=chip-muted>dominated</span>"
        else:
            gain = score["each"].get(r["key"], 0.0) if score else 0.0
            chip = f"<span class=chip-ok>+{gain:.2f}% area</span>" if gain > 0 else "<span class=chip-muted>dominated</span>"
        dot = f'<span class="dot" style="background:var({r["colour"]})"></span>' if r["kind"] == "baseline" else '<span class="dia"></span>'
        return (f'<tr class="{r["kind"]}"><td>{dot}{html.escape(r["label"])}<div class="sub">{html.escape(r["desc"])}</div></td>'
                f'<td class="num">{s["sim3_ate_rmse_mean"]:.3f}</td><td class="num">{fmt_fuel(s["fuel_per_frame_max"])}</td>'
                f'<td class="num">{fmt_fuel(s["fuel_per_frame_p99"])}</td><td class="num">{fmt_fuel(s["fuel_per_frame_median"])}</td>'
                f'<td class="num">{s["ate_rmse_median"]:.1f}</td><td class="num">{s["completed"]}/{s["episodes"]}</td><td>{chip}</td></tr>')

    ordered = [r for r in rows if r["kind"] == "baseline"] + sorted(variants, key=lambda r: r["summary"]["fuel_per_frame_max"])
    notes = json.loads(NOTES_FILE.read_text()) if NOTES_FILE.exists() else NOTES
    total = score["all"] if score else 0.0
    best_var = min(variants, key=lambda r: r["summary"]["sim3_ate_rmse_mean"]) if variants else None
    cheapest_good = None
    if klt and variants:
        ok = [r for r in variants if r["summary"]["sim3_ate_rmse_mean"] <= klt["summary"]["sim3_ate_rmse_mean"]]
        cheapest_good = min(ok, key=lambda r: r["summary"]["fuel_per_frame_max"]) if ok else None
    if total > 0:
        verdict_class, verdict = "good", (f"<strong>Local improvement: +{total:.2f}% of the scoring area</strong> beyond the stand-in baseline frontier. "
                                          "This is measured against the three upstream examples, not the live server's baselines, so it is a local result until "
                                          "the server's /config and baseline points are matched.")
    elif variants:
        verdict_class, verdict = "warn", "<strong>No local improvement yet.</strong> Every variant measured so far is dominated by a baseline."
    else:
        verdict_class, verdict = "warn", "<strong>Not there yet.</strong> Baselines are measured; no variants yet."
    tiles = [
        (f"+{total:.2f}%", "local area improvement (server formula, stand-in baselines)"),
        (f"{fmt_fuel(klt['summary']['fuel_per_frame_max'])} → {fmt_fuel(cheapest_good['summary']['fuel_per_frame_max'])}" if cheapest_good else "n/a",
         f"KLT peak fuel → cheapest variant at least as accurate{' (' + cheapest_good['key'] + ')' if cheapest_good else ''}"),
        (f"{best_var['summary']['sim3_ate_rmse_mean']:.3f} m" if best_var else "n/a", f"best variant ATE{' (' + best_var['key'] + ')' if best_var else ''}; KLT {klt['summary']['sim3_ate_rmse_mean']:.3f} m" if klt else ""),
        (str(len(variants)), "variants measured, 10 public episodes each"),
    ]
    r0 = rows[0]["report"]
    held_rows = []
    by_key = {r["key"]: r for r in rows}
    pooled = {}  # approach -> episodes pooled over every outputs/heldout*/ directory (one per held-out group)
    for path in sorted(ROOT.glob("outputs/heldout*/*-wasm/report.json")):
        pooled.setdefault(path.parent.name[:-5], {}).update({e["hash"]: e for e in json.loads(path.read_text())["episodes"]})
    n_held = max((len(v) for v in pooled.values()), default=0)
    for key, eps in pooled.items():
        eps = list(eps.values())
        ate = sum(e["metrics"]["sim3_ate_rmse"] for e in eps) / len(eps)
        fuel = max(e["fuel_per_frame_max"] for e in eps)
        unal = sorted(e["metrics"]["ate_rmse"] for e in eps)[len(eps) // 2]
        filled = sum(e["metrics"]["filled_frames"] for e in eps)
        done = sum(1 for e in eps if e["completed"])
        pub = by_key.get(key)
        held_rows.append((ate, f'<tr><td>{html.escape(pub["label"] if pub else key)}</td>'
            f'<td class="num">{pub["summary"]["sim3_ate_rmse_mean"]:.3f}</td>' if pub else f'<tr><td>{html.escape(key)}</td><td class="num">n/a</td>'))
        held_rows[-1] = (ate, held_rows[-1][1] +
            f'<td class="num">{ate:.3f}</td><td class="num">{fmt_fuel(pub["summary"]["fuel_per_frame_max"]) if pub else "n/a"}</td>'
            f'<td class="num">{fmt_fuel(fuel)}</td><td class="num">{unal:.1f}</td><td class="num">{done}/{len(eps)}</td><td class="num">{filled}</td></tr>')
    held_rows.sort(key=lambda x: x[0])
    heldout = ("" if not held_rows else
        '<section class="panel"><h2>Generalization check: held-out episodes</h2>'
        f'<p class="meta">{n_held} of 10 planned one-minute episodes scored so far. They were simulated for this check and never used for tuning. Plan: 4 other ProcTHOR scenes, 3 HSSD scenes '
        '(a different scene catalog), and the public scene with a new seed, a 3 s hover start, and a fixed 90° field of view. Same .wasm files as above.</p>'
        '<div class="scroll"><table><thead><tr><th>Approach</th><th class="num">Public ATE (m)</th><th class="num">Held-out ATE (m)</th>'
        '<th class="num">Public max fuel</th><th class="num">Held-out max fuel</th><th class="num">Held-out median unaligned ATE (m)</th>'
        '<th class="num">Completed</th><th class="num">Filled frames</th></tr></thead><tbody>'
        + "\n".join(r for _, r in held_rows) + "</tbody></table></div></section>")
    page = TEMPLATE.format(heldout=heldout, 
        updated=updated.strftime("%B %-d, %Y %H:%M"), n_eps=rows[0]["summary"]["episodes"],
        verdict_class=verdict_class, verdict=verdict,
        tiles="\n".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>' for a, b in tiles),
        pareto=pareto_svg(rows, limit),
        legend=" ".join(f'<span><span class="dot" style="background:var({r["colour"]})"></span>{html.escape(r["label"])}</span>' for r in rows if r["kind"] == "baseline"),
        rows="\n".join(row_html(r) for r in ordered),
        status="\n".join(f'<li class="st-{k}"><span class="pill">{k}</span><div><strong>{html.escape(t)}</strong><p>{html.escape(d)}</p></div></li>' for k, t, d in STATUS),
        notes="\n".join(f"<li>{html.escape(n)}</li>" for n in notes),
        env=html.escape(f'wasmtime {r0["wasmtime"]} · {r0["host"]} · default local limits (step {fmt_fuel(limit)}, init {fmt_fuel(r0["limits"]["init_fuel"])})'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(OUT, f"improvement {total:.3f}%", {k: round(v, 3) for k, v in (score["each"] if score else {}).items()})


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EECS 116 VIO Progress</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Fraunces:opsz,wght@9..144,600&display=swap" rel="stylesheet">
<style>
:root {{
  /* lab-notebook dashboard: verdict first, then the frontier chart, the numbers, the reading, the work log */
  --bg:#f5f6f8; --panel:#ffffff; --ink:#1b2230; --muted:#5d6678; --line:#dde1e8; --accent:#2f5bd3;
  --c-min:#9aa3b2; --c-dr:#d9822b; --c-klt:#2f5bd3; --c-ov:#8b5cf6; --c-vm:#c2410c; --c-orb:#be185d; --c-var:#0f8a7e; --gain:rgba(15,138,126,.18); --base-fill:rgba(93,102,120,.10);
  --ok:#1f8a5b; --warn:#b7791f; --bad:#c2410c; --todo:#6b7280;
  --display:"Fraunces", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#12161e; --panel:#1a202b; --ink:#e6e9ef; --muted:#9aa3b5; --line:#2c3443; --accent:#7f9cf5;
  --c-min:#7c8597; --c-dr:#f0a050; --c-klt:#7f9cf5; --c-ov:#a78bfa; --c-vm:#f07a4a; --c-orb:#f472b6; --c-var:#3cc7b5; --gain:rgba(60,199,181,.20); --base-fill:rgba(154,163,181,.10);
  --ok:#4cc38a; --warn:#e0b04a; --bad:#f07a4a; --todo:#9aa3b5; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg:#12161e; --panel:#1a202b; --ink:#e6e9ef; --muted:#9aa3b5; --line:#2c3443; --accent:#7f9cf5;
  --c-min:#7c8597; --c-dr:#f0a050; --c-klt:#7f9cf5; --c-ov:#a78bfa; --c-vm:#f07a4a; --c-orb:#f472b6; --c-var:#3cc7b5; --gain:rgba(60,199,181,.20); --base-fill:rgba(154,163,181,.10);
  --ok:#4cc38a; --warn:#e0b04a; --bad:#f07a4a; --todo:#9aa3b5; color-scheme: dark; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 var(--body); }}
main {{ max-width:1080px; margin:0 auto; padding:32px 20px 64px; display:grid; gap:24px; }}
h1 {{ font:600 2rem/1.15 var(--display); margin:0; text-wrap:balance; }}
h2 {{ font:600 1.05rem/1.3 var(--body); margin:0 0 12px; }}
.meta {{ color:var(--muted); font:13px var(--mono); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:20px; }}
.verdict {{ border-left:4px solid var(--warn); }} .verdict.good {{ border-left-color:var(--ok); }}
.verdict p {{ margin:6px 0 0; max-width:78ch; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.tile b {{ display:block; font:500 1.35rem/1.25 var(--mono); font-variant-numeric:tabular-nums; }}
.tile span {{ color:var(--muted); font-size:13px; }}
svg {{ width:100%; height:auto; display:block; }}
svg .grid {{ stroke:var(--line); stroke-width:1; }}
svg .limit {{ stroke:var(--bad); stroke-dasharray:4 4; }}
svg .gain {{ fill:var(--gain); stroke:var(--c-var); stroke-width:1.5; }}
svg .base {{ fill:var(--panel); stroke:var(--muted); stroke-dasharray:6 4; stroke-width:1.5; }}
svg .base {{ fill-opacity:1; }}
svg .var {{ fill:var(--panel); stroke:var(--c-var); stroke-width:2; }} svg .var.on {{ fill:var(--c-var); }}
svg .tick {{ fill:var(--muted); font:11px var(--mono); }}
svg .axis {{ fill:var(--muted); font:12px var(--body); }}
svg .pt {{ fill:var(--ink); font:500 12px var(--body); }} svg .pt.small {{ font:11px var(--mono); fill:var(--c-var); }}
.legend {{ display:flex; flex-wrap:wrap; gap:16px; color:var(--muted); font-size:13px; margin-top:8px; align-items:center; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
.dia {{ display:inline-block; width:9px; height:9px; background:var(--c-var); transform:rotate(45deg); margin-right:8px; }}
.sw {{ display:inline-block; width:14px; height:10px; margin-right:6px; vertical-align:middle; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; min-width:820px; }}
th, td {{ text-align:left; padding:9px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:12px; color:var(--muted); font-weight:500; }}
tr.baseline td {{ background:color-mix(in srgb, var(--line) 25%, transparent); }}
td.num, th.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.sub {{ color:var(--muted); font:12px var(--mono); max-width:46ch; }}
.chip-ok, .chip-muted {{ font-size:12px; padding:2px 8px; border-radius:99px; white-space:nowrap; }}
.chip-ok {{ color:var(--ok); border:1px solid var(--ok); }} .chip-muted {{ color:var(--muted); border:1px solid var(--line); }}
ul.status {{ list-style:none; margin:0; padding:0; display:grid; gap:12px; }}
ul.status li {{ display:grid; grid-template-columns:88px 1fr; gap:12px; align-items:start; }}
ul.status p {{ margin:2px 0 0; color:var(--muted); max-width:82ch; }}
.pill {{ font:500 11px var(--mono); text-transform:uppercase; padding:3px 0; border-radius:4px; text-align:center; color:#fff; }}
.st-done .pill {{ background:var(--ok); }} .st-running .pill {{ background:var(--accent); }}
.st-blocked .pill {{ background:var(--bad); }} .st-todo .pill {{ background:var(--todo); }}
ol.notes {{ margin:0; padding-left:20px; display:grid; gap:8px; max-width:82ch; }}
@media (max-width:640px) {{ ul.status li {{ grid-template-columns:1fr; }} }}
</style></head>
<body><main>
<header>
  <h1>VIOBench progress: accuracy vs. peak fuel</h1>
  <div class="meta">EECS 116 final · data updated {updated} · {n_eps} public episodes · local WASM sandbox</div>
</header>

<section class="panel verdict {verdict_class}">
  <h2>Where we stand on the objective</h2>
  <p><strong>Objective:</strong> add points beyond the shared baseline frontier on mean Sim(3) ATE and the <em>maximum</em> fuel of any
  single frame, both lower is better. The server scores the extra dominated area in log space, as a percentage of a reference box.</p>
  <p>{verdict}</p>
</section>

<div class="tiles">{tiles}</div>

<section class="panel">
  <h2>Pareto view</h2>
  {pareto}
  <div class="legend">{legend}<span><span class="dia"></span>variant (filled: on the frontier)</span>
  <span><span class="sw" style="background:var(--gain);border:1px solid var(--c-var)"></span>area gained over baselines</span>
  <span><span class="sw" style="border:1px dashed var(--muted)"></span>baseline frontier</span></div>
</section>

<section class="panel">
  <h2>Measured results</h2>
  <div class="scroll"><table>
    <thead><tr><th>Approach</th><th class="num">Mean Sim(3) ATE (m)</th><th class="num">Max fuel</th><th class="num">p99 fuel</th>
    <th class="num">Median fuel</th><th class="num">Median unaligned ATE (m)</th><th class="num">Completed</th><th>Area alone</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <p class="meta">{env}</p>
</section>

{heldout}

<section class="panel">
  <h2>What the numbers say</h2>
  <ol class="notes">{notes}</ol>
</section>

<section class="panel">
  <h2>Work log</h2>
  <ul class="status">{status}</ul>
</section>
</main></body></html>
"""

if __name__ == "__main__":
    build()
