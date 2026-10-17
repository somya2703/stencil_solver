#!/usr/bin/env python3
"""
scripts/repeat_bench.py -- run naive-vs-tiled N times and report mean +/- sd and a 95% CI.

Each run of `stencil_tiled` measures naive and then tiled in one process, so every repeat
is already an alternating pair (the same protocol as the 5x5 repeat in docs/CASE_STUDY.md
section 3.2). The speed ratio is computed per pair (naive time / tiled time) and then
summarised, which is what the README headline should quote.

Usage (needs the GPU build):
    python3 scripts/repeat_bench.py --binary build/bin/stencil_tiled --nx 512 --runs 10

Options:
    --runs N        number of pairs (default 10)
    --pause S       seconds to idle between runs so clocks/thermals settle (default 5)
    --steps/--warmup  passed through (defaults 200 / 20, as in the guide)
    --outdir DIR    per-run JSONs and the summary are written here (default results/repeat)

Output: a printed table, <outdir>/repeat_<nx>.json, and a ready-to-paste Markdown line.
Note: *.json is git-ignored, so commit the printed Markdown line, not the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time

# Two-sided 95% Student-t critical values by degrees of freedom (n - 1).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
       17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t_crit(df: int) -> float:
    if df in T95:
        return T95[df]
    keys = sorted(T95)
    for k in keys:
        if k >= df:
            return T95[k]
    return 1.96


def summarise(xs: list[float]) -> dict:
    n = len(xs)
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    half = t_crit(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": mean, "sd": sd, "ci95_half_width": half, "min": min(xs), "max": max(xs)}


def read_ms(path: pathlib.Path) -> float:
    data = json.loads(path.read_text())
    results = data.get("results", [])
    if len(results) != 1:
        raise SystemExit(f"{path}: expected exactly one result entry, got {len(results)}")
    return float(results[0]["time_ms"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--binary", default="build/bin/stencil_tiled")
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--pause", type=float, default=5.0)
    ap.add_argument("--outdir", default="results/repeat")
    ap.add_argument("--label", default="", help="free text stored with the summary (e.g. TILE_Y/PENCIL_Z)")
    a = ap.parse_args()

    if a.nx == 256:
        sys.exit("--nx 256 triggers the 128/256/512 sweep in stencil_tiled; use another size (e.g. 255)")
    if a.runs < 2:
        sys.exit("--runs must be at least 2")

    binary = pathlib.Path(a.binary)
    if not binary.exists():
        sys.exit(f"binary not found: {binary} (build first, see BUILD_AND_RUN.md)")
    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    naive_ms, tiled_ms, ratios = [], [], []
    for i in range(1, a.runs + 1):
        tiled_json = outdir / f"run{i:02d}_tiled.json"
        naive_json = outdir / f"run{i:02d}_tiled_naive.json"  # stencil_tiled derives this name
        cmd = [str(binary), "--nx", str(a.nx), "--steps", str(a.steps),
               "--warmup", str(a.warmup), "--output", str(tiled_json)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"run {i} failed ({proc.returncode}):\n{proc.stderr or proc.stdout}")
        n_ms, t_ms = read_ms(naive_json), read_ms(tiled_json)
        naive_ms.append(n_ms)
        tiled_ms.append(t_ms)
        ratios.append(n_ms / t_ms)
        print(f"run {i:2d}/{a.runs}: naive {n_ms:8.3f} ms   tiled {t_ms:8.3f} ms   "
              f"ratio {n_ms / t_ms:.3f}x", flush=True)
        if i < a.runs and a.pause > 0:
            time.sleep(a.pause)

    summary = {
        "grid": a.nx, "steps": a.steps, "warmup": a.warmup, "runs": a.runs, "label": a.label,
        "naive_ms": summarise(naive_ms), "tiled_ms": summarise(tiled_ms),
        "tiled_speed_vs_naive": summarise(ratios),
    }
    (outdir / f"repeat_{a.nx}.json").write_text(json.dumps(summary, indent=2))

    r = summary["tiled_speed_vs_naive"]
    print()
    print(f"naive : {summary['naive_ms']['mean']:.3f} ms  (sd {summary['naive_ms']['sd']:.3f})")
    print(f"tiled : {summary['tiled_ms']['mean']:.3f} ms  (sd {summary['tiled_ms']['sd']:.3f})")
    print(f"ratio : {r['mean']:.3f}x  sd {r['sd']:.3f}  95% CI "
          f"[{r['mean'] - r['ci95_half_width']:.3f}, {r['mean'] + r['ci95_half_width']:.3f}]  "
          f"(range {r['min']:.3f}-{r['max']:.3f})")
    print()
    print("Markdown for the README / CASE_STUDY:")
    print(f"> Tiled runs at **{r['mean']:.3f}x** of naive at {a.nx}^3 "
          f"(mean of {a.runs} alternating naive/tiled pairs, sd {r['sd']:.3f}, "
          f"95% CI {r['mean'] - r['ci95_half_width']:.3f}-{r['mean'] + r['ci95_half_width']:.3f}; "
          f"naive {summary['naive_ms']['mean']:.2f} ms, tiled {summary['tiled_ms']['mean']:.2f} ms).")


if __name__ == "__main__":
    main()
