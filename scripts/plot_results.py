#!/usr/bin/env python3
"""
scripts/plot_results.py — generate performance comparison plots.

Usage:
    python3 scripts/plot_results.py \
        --naive  results/bench_naive.json \
        --tiled  results/bench_tiled.json \
        --scaling results/bench_scaling.json \
        --outdir results/plots \
        --report results/BENCHMARKS.md
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required. Run: pip3 install matplotlib numpy")
    sys.exit(1)

STYLE = {
    "naive":  {"color": "#E24B4A", "marker": "o", "label": "Naive (global mem)"},
    "tiled":  {"color": "#185FA5", "marker": "s", "label": "Tiled (shared mem)"},
    "weak":   {"color": "#1D9E75", "marker": "^", "label": "Weak scaling"},
    "strong": {"color": "#BA7517", "marker": "D", "label": "Strong scaling"},
}


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def plot_throughput_comparison(naive: dict, tiled: dict, outdir: pathlib.Path) -> None:
    """Bar chart: naive vs tiled bandwidth and GFLOPS across grid sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Naive vs Tiled Kernel Performance", fontsize=14, fontweight="bold")

    sizes = [r["grid_size"] for r in naive.get("results", [])]
    naive_bw   = [r["bandwidth_gbs"]  for r in naive.get("results", [])]
    tiled_bw   = [r["bandwidth_gbs"]  for r in tiled.get("results", [])]
    naive_gf   = [r["gflops"]         for r in naive.get("results", [])]
    tiled_gf   = [r["gflops"]         for r in tiled.get("results", [])]

    x = np.arange(len(sizes))
    w = 0.35

    # Bandwidth
    ax = axes[0]
    ax.bar(x - w/2, naive_bw, w, label="Naive", color=STYLE["naive"]["color"], alpha=0.85)
    ax.bar(x + w/2, tiled_bw, w, label="Tiled", color=STYLE["tiled"]["color"], alpha=0.85)
    ax.set_xlabel("Grid size (N³)")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title("Memory bandwidth")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # GFLOPS
    ax = axes[1]
    ax.bar(x - w/2, naive_gf, w, label="Naive", color=STYLE["naive"]["color"], alpha=0.85)
    ax.bar(x + w/2, tiled_gf, w, label="Tiled", color=STYLE["tiled"]["color"], alpha=0.85)
    ax.set_xlabel("Grid size (N³)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Compute throughput")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = outdir / "throughput_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_scaling(data: dict, outdir: pathlib.Path) -> None:
    """Weak and strong scaling efficiency curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Multi-GPU Scaling", fontsize=14, fontweight="bold")

    for ax, mode in zip(axes, ["weak", "strong"]):
        runs = [r for r in data.get("results", []) if r.get("mode") == mode]
        if not runs:
            continue
        gpus   = [r["ngpu"]      for r in runs]
        eff    = [r["efficiency"] for r in runs]
        s      = STYLE[mode]
        ax.plot(gpus, eff, marker=s["marker"], color=s["color"],
                linewidth=2, markersize=8, label=s["label"])
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5, label="Ideal")
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel("Parallel efficiency")
        ax.set_title(f"{mode.capitalize()} scaling")
        ax.set_ylim(0, 1.1)
        ax.set_xticks(gpus)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = outdir / "scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def write_markdown_report(naive: dict, tiled: dict, scaling: dict,
                          outdir: pathlib.Path, report_path: str) -> None:
    speedups = []
    for n_r, t_r in zip(naive.get("results", []), tiled.get("results", [])):
        if n_r["time_ms"] > 0:
            speedups.append(t_r["time_ms"] / n_r["time_ms"])

    avg_speedup = np.mean(speedups) if speedups else 0.0

    lines = [
        "# Performance Benchmarks",
        "",
        f"**GPU:** {naive.get('gpu_name', 'unknown')}  ",
        f"**CUDA:** {naive.get('cuda_version', 'unknown')}  ",
        f"**Stencil radius:** {naive.get('stencil_radius', '?')}  ",
        "",
        "## Kernel comparison (naive vs tiled)",
        "",
        "| Grid | Naive BW (GB/s) | Tiled BW (GB/s) | Speedup |",
        "|------|----------------|----------------|---------|",
    ]
    for n_r, t_r in zip(naive.get("results", []), tiled.get("results", [])):
        sp = n_r["time_ms"] / t_r["time_ms"] if t_r["time_ms"] > 0 else 0
        lines.append(
            f"| {n_r['grid_size']}³ "
            f"| {n_r['bandwidth_gbs']:.1f} "
            f"| {t_r['bandwidth_gbs']:.1f} "
            f"| **{sp:.2f}×** |"
        )

    lines += [
        "",
        f"**Average speedup: {avg_speedup:.2f}×**",
        "",
        "## Plots",
        "",
        f"![Throughput comparison](plots/throughput_comparison.png)",
        f"![Scaling curves](plots/scaling.png)",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stencil solver benchmark results")
    parser.add_argument("--naive",   required=False, help="Naive benchmark JSON")
    parser.add_argument("--tiled",   required=False, help="Tiled benchmark JSON")
    parser.add_argument("--scaling", required=False, help="Scaling benchmark JSON")
    parser.add_argument("--outdir",  default="results/plots", help="Output directory")
    parser.add_argument("--report",  default=None, help="Markdown report path")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    naive   = load_json(args.naive)   if args.naive   else {}
    tiled   = load_json(args.tiled)   if args.tiled   else {}
    scaling = load_json(args.scaling) if args.scaling else {}

    if naive and tiled:
        plot_throughput_comparison(naive, tiled, outdir)

    if scaling:
        plot_scaling(scaling, outdir)

    if args.report and naive and tiled:
        write_markdown_report(naive, tiled, scaling, outdir, args.report)


if __name__ == "__main__":
    main()
