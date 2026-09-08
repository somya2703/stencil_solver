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

#from networkx import radius

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


def compute_arithmetic_intensity(stencil_radius: int, precision: str) -> float:
    """AI = FLOP/byte for the stencil, given radius and precision.

    Matches the formula in include/stencil/types.hpp:
      flops_per_point = 3*(4R+1) + 5
      bytes_per_point = (6R+4) * sizeof(real_t)
    """
    bytes_per_elem = 8 if precision == "fp64" else 4
    R = stencil_radius
    flops = 3 * (4 * R + 1) + 5
    bytes_moved = (6 * R + 4) * bytes_per_elem
    return flops / bytes_moved

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

def plot_roofline(naive: dict, tiled: dict, peak_bw: float, peak_flops_tflops: float,
                   outdir: pathlib.Path) -> None:
    """Roofline chart: memory-bound ceiling vs compute-bound ceiling,
    with naive/tiled operating points plotted at the kernel's AI."""
    fig, ax = plt.subplots(figsize=(7, 6))

    radius    = naive.get("stencil_radius", tiled.get("stencil_radius", 4))
    precision = naive.get("precision", tiled.get("precision", "fp32"))
    ai = compute_arithmetic_intensity(radius, precision)

    peak_flops_gflops = peak_flops_tflops * 1000.0
    ridge_ai = peak_flops_gflops / peak_bw

    ai_range = np.logspace(-3, 3, 200)
    roof = np.minimum(peak_bw * ai_range, peak_flops_gflops)
    ax.plot(ai_range, roof, color="black", linewidth=1.5, label="Roofline")
    ax.axvline(ridge_ai, linestyle=":", color="gray", alpha=0.6,
               label=f"Ridge point (AI={ridge_ai:.1f})")
    ax.axvline(ai, linestyle="--", color="#666666", alpha=0.5,
               label=f"Kernel AI={ai:.2f}")

    for data, key in [(naive, "naive"), (tiled, "tiled")]:
        results = data.get("results", [])
        if not results:
            continue
        gflops_pts = [r["gflops"] for r in results]
        ai_pts = [ai] * len(gflops_pts)
        s = STYLE[key]
        ax.scatter(ai_pts, gflops_pts, color=s["color"], marker=s["marker"],
                   s=90, label=s["label"], zorder=5, edgecolor="black", linewidth=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Attained performance (GFLOP/s)")
    ax.set_title("Roofline Model")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = outdir / "roofline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_speedup(naive: dict, tiled: dict, outdir: pathlib.Path) -> None:
    """Line chart: naive/tiled speedup ratio across grid sizes."""
    fig, ax = plt.subplots(figsize=(7, 5))

    n_results = naive.get("results", [])
    t_results = tiled.get("results", [])

    sizes    = []
    speedups = []
    for n_r, t_r in zip(n_results, t_results):
        if t_r.get("time_ms", 0) > 0:
            sizes.append(n_r["grid_size"])
            speedups.append(n_r["time_ms"] / t_r["time_ms"])

    if not speedups:
        print("Warning: no matching naive/tiled entries — skipping speedup plot")
        return

    x = np.arange(len(sizes))
    ax.plot(x, speedups, marker="o", color=STYLE["tiled"]["color"],
            linewidth=2, markersize=8, label="Tiled speedup over naive")

    for xi, sp in zip(x, speedups):
        ax.annotate(f"{sp:.2f}×", (xi, sp), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    avg = np.mean(speedups)
    ax.axhline(avg, linestyle="--", color="gray", alpha=0.6,
               label=f"Average: {avg:.2f}×")

    ax.set_xlabel("Grid size (N³)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Tiled vs Naive Speedup")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = outdir / "speedup.png"
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
    parser.add_argument("--peak-bw",    type=float, default=None,
                        help="Peak memory bandwidth in GB/s (enables roofline plot)")
    parser.add_argument("--peak-flops", type=float, default=None,
                        help="Peak FP throughput in TFLOP/s (enables roofline plot)")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    naive   = load_json(args.naive)   if args.naive   else {}
    tiled   = load_json(args.tiled)   if args.tiled   else {}
    scaling = load_json(args.scaling) if args.scaling else {}

    if naive and tiled:
        plot_throughput_comparison(naive, tiled, outdir)
        plot_speedup(naive, tiled, outdir)

        if args.peak_bw and args.peak_flops:
            plot_roofline(naive, tiled, args.peak_bw, args.peak_flops, outdir)

    if scaling:
        plot_scaling(scaling, outdir)

    if args.report and naive and tiled:
        write_markdown_report(naive, tiled, scaling, outdir, args.report)


if __name__ == "__main__":
    main()
