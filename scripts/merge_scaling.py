#!/usr/bin/env python3
"""
scripts/merge_scaling.py

Merges per-GPU-count scaling JSON files produced by run_scaling.sh
into a single combined JSON for plot_results.py.

run_scaling.sh writes one file per (mode, ngpu) pair:
  results/weak_gpu1_20240101_120000.json
  results/weak_gpu2_20240101_120001.json
  results/strong_gpu1_20240101_120002.json
  ...

Each file has the schema written by main_multigpu.cpp:
  { "results": [{ "mode": "multigpu", "ngpu": N, "time_ms": X, "efficiency": Y }] }

This script:
  1. Reads all matching files from --indir
  2. Computes parallel efficiency relative to the 1-GPU baseline
  3. Writes a single combined JSON with the schema expected by plot_results.py:
     { "results": [{ "mode": "weak"|"strong", "ngpu": N, "time_ms": X, "efficiency": Y }] }

Usage:
    python3 scripts/merge_scaling.py \\
        --indir results \\
        --output results/scaling_combined.json

    # Merge only weak scaling results:
    python3 scripts/merge_scaling.py \\
        --indir results --mode weak \\
        --output results/scaling_weak_combined.json
"""

from __future__ import annotations
import argparse
import glob
import json
import pathlib
import sys
from typing import Optional


def load_scaling_file(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", [])
        if not results:
            return None
        return results[0]   # each file has exactly one result entry
    except Exception as e:
        print(f"Warning: skipping {path}: {e}", file=sys.stderr)
        return None


def infer_mode(filename: str) -> Optional[str]:
    """Infer 'weak' or 'strong' from the filename prefix written by run_scaling.sh."""
    name = pathlib.Path(filename).stem.lower()
    if name.startswith("weak"):
        return "weak"
    if name.startswith("strong"):
        return "strong"
    return None


def compute_efficiency(runs: list[dict]) -> list[dict]:
    """
    Set efficiency = time_1gpu / (ngpu × time_Ngpu) for each entry.
    For weak scaling:   ideal is constant time → efficiency = time_1 / time_N
    For strong scaling: ideal is linear speedup → efficiency = time_1 / (N × time_N)
    We use the strong-scaling formula for both (it's the standard definition).
    """
    if not runs:
        return runs
    # Sort by ngpu
    runs = sorted(runs, key=lambda r: r["ngpu"])
    baseline_ms = runs[0]["time_ms"]
    baseline_ngpu = runs[0]["ngpu"]
    for r in runs:
        if r["time_ms"] > 0 and baseline_ms > 0:
            # Speedup relative to baseline (usually 1 GPU)
            speedup = baseline_ms / r["time_ms"] * (r["ngpu"] / baseline_ngpu)
            r["efficiency"] = round(speedup / (r["ngpu"] / baseline_ngpu), 4) \
                if r["ngpu"] != baseline_ngpu else 1.0
        else:
            r["efficiency"] = 0.0
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description="Merge scaling JSON files")
    p.add_argument("--indir",  required=True,
                   help="Directory containing per-GPU scaling JSON files")
    p.add_argument("--output", required=True,
                   help="Output combined JSON path")
    p.add_argument("--mode",   choices=["weak", "strong", "both"],
                   default="both",
                   help="Which scaling mode to include (default: both)")
    args = p.parse_args()

    indir = pathlib.Path(args.indir)
    if not indir.is_dir():
        print(f"Error: {indir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Collect all JSON files in the directory
    json_files = sorted(glob.glob(str(indir / "*.json")))
    if not json_files:
        print(f"Error: no JSON files found in {indir}", file=sys.stderr)
        sys.exit(1)

    weak_runs:   list[dict] = []
    strong_runs: list[dict] = []

    for path in json_files:
        mode = infer_mode(path)
        if mode is None:
            continue   # skip bench_naive.json, bench_tiled.json, etc.

        if args.mode != "both" and mode != args.mode:
            continue

        entry = load_scaling_file(path)
        if entry is None:
            continue

        # Normalise mode field to "weak" or "strong"
        entry["mode"] = mode

        if mode == "weak":
            weak_runs.append(entry)
        else:
            strong_runs.append(entry)

    if not weak_runs and not strong_runs:
        print("Error: no scaling data found — check file naming "
              "(files must start with 'weak_' or 'strong_')",
              file=sys.stderr)
        sys.exit(1)

    # Compute efficiency relative to 1-GPU baseline
    weak_runs   = compute_efficiency(weak_runs)
    strong_runs = compute_efficiency(strong_runs)

    combined = {"results": weak_runs + strong_runs}

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(combined, f, indent=2)

    total = len(weak_runs) + len(strong_runs)
    print(f"Merged {total} scaling results → {output}")
    if weak_runs:
        gpus = [r["ngpu"] for r in weak_runs]
        print(f"  Weak   scaling: GPUs {gpus}")
    if strong_runs:
        gpus = [r["ngpu"] for r in strong_runs]
        print(f"  Strong scaling: GPUs {gpus}")


if __name__ == "__main__":
    main()
