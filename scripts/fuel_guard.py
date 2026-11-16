"""Fuel guard: warn (or refuse) before submitting a .wasm whose measured fuel is close to the server's limits.

Reads one or more report.json files written by `viobench evaluate ... -o <dir>` and compares the peak fuel per frame
and the initialization fuel with the limits. Limits come from the live server (`/config`) when VIOBENCH_TOKEN is set,
otherwise from the local defaults (step 2e10, init 2e11).

    python scripts/fuel_guard.py report-c_best/report.json report-w_s0500_c60/report.json
    VIOBENCH_TOKEN=... python scripts/fuel_guard.py report-*/report.json      # use the server's current limits

Exit code: 0 = all safe, 1 = at least one WARN, 2 = at least one STOP (do not submit).
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_LIMITS = {"step_fuel": 2e10, "init_fuel": 2e11}
SERVER = "https://viobench.rl.tools"


def server_limits(token):
    """The server's current evaluation limits; instructors can change them, so prefer these over local defaults."""
    request = urllib.request.Request(f"{SERVER}/config", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)["evaluation"]["limits"]


def classify(used, limit, warn, stop):
    share = used / limit
    level = "STOP" if share >= stop else "WARN" if share >= warn else "ok"
    return level, share


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", help="report.json files from viobench evaluate")
    parser.add_argument("--warn", type=float, default=0.10, help="warn at this share of the limit (default 0.10)")
    parser.add_argument("--stop", type=float, default=0.33, help="refuse at this share of the limit (default 0.33)")
    args = parser.parse_args()

    token = os.environ.get("VIOBENCH_TOKEN")
    limits, source = dict(DEFAULT_LIMITS), "local defaults"
    if token:
        try:
            limits.update(server_limits(token))
            source = "server /config"
        except Exception as error:  # offline or bad token: fall back, but say so
            print(f"! could not read server limits ({error}); using local defaults")
    print(f"limits ({source}): step {limits['step_fuel']:.3g}, init {limits['init_fuel']:.3g}; "
          f"WARN at {args.warn:.0%}, STOP at {args.stop:.0%}\n")

    worst = 0
    for path in args.reports:
        report = json.loads(Path(path).read_text())
        summary = report.get("summary") or {}
        episodes = report.get("episodes") or []
        step_peak = summary.get("fuel_per_frame_max")
        init_peak = max((e.get("fuel_init", 0) for e in episodes), default=None)
        failed = [e.get("hash", "?")[:8] for e in episodes if e.get("completed") is False]

        print(f"{path}")
        if step_peak is None:
            print("  ! no fuel_per_frame_max in this report: is it a viobench evaluate report?")
            worst = max(worst, 1)
            continue
        level, share = classify(step_peak, limits["step_fuel"], args.warn, args.stop)
        print(f"  step peak {step_peak:,.0f}  = {share:.2%} of limit  [{level}]")
        worst = max(worst, {"ok": 0, "WARN": 1, "STOP": 2}[level])
        if init_peak is not None:
            level, share = classify(init_peak, limits["init_fuel"], args.warn, args.stop)
            print(f"  init peak {init_peak:,.0f}  = {share:.2%} of limit  [{level}]")
            worst = max(worst, {"ok": 0, "WARN": 1, "STOP": 2}[level])
        if failed:
            print(f"  ! {len(failed)} episode(s) did not complete: {', '.join(failed)}  [STOP]")
            worst = 2
        print(f"  episodes measured: {summary.get('completed', '?')}/{summary.get('episodes', '?')}"
              f"  (peak can be higher on unseen episodes; measure more than one if you can)")

    print("\n" + {0: "OK: safe to submit.", 1: "WARNING: close to a limit, think before submitting.",
                  2: "STOP: do not submit this build."}[worst])
    sys.exit(worst)


if __name__ == "__main__":
    main()
