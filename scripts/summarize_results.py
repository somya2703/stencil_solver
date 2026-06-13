#!/usr/bin/env python3
"""
scripts/summarize_results.py
Reads naive + tiled JSON results and prints a Markdown summary table
to stdout (appended to GITHUB_STEP_SUMMARY by benchmark.yml).

Usage:
    python3 scripts/summarize_results.py \
        --naive  results/naive.json \
        --tiled  results/tiled.json
"""

import argparse, json, sys

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not load {path}: {e}", file=sys.stderr)
        return {}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--naive",  required=True)
    p.add_argument("--tiled",  required=True)
    args = p.parse_args()

    naive = load(args.naive)
    tiled = load(args.tiled)

    n_results = naive.get("results", [])
    t_results = tiled.get("results", [])

    gpu = naive.get("gpu_name", tiled.get("gpu_name", "unknown"))
    print(f"**GPU:** {gpu}  ")
    print(f"**Stencil radius:** {naive.get('stencil_radius', '?')}  ")
    print(f"**Precision:** {naive.get('precision', 'fp32')}  ")
    print()
    print("| Grid | Naive BW (GB/s) | Tiled BW (GB/s) | Speedup |")
    print("|------|----------------|----------------|---------|")

    for n, t in zip(n_results, t_results):
        sp = n["time_ms"] / t["time_ms"] if t.get("time_ms", 0) > 0 else 0
        print(f"| {n['grid_size']}³ "
              f"| {n['bandwidth_gbs']:.1f} "
              f"| {t['bandwidth_gbs']:.1f} "
              f"| **{sp:.2f}×** |")

if __name__ == "__main__":
    main()
