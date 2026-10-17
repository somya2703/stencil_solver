#!/usr/bin/env python3
"""
scripts/make_story_plots.py -- the four "story" charts used in the README.

All numbers are the measured values recorded in docs/CASE_STUDY.md (section
references are in the comments next to each table), so the charts regenerate
without a GPU:

    python3 scripts/make_story_plots.py --outdir docs/plots

Charts:
  tiled_vs_naive_by_grid.png   tiled/naive speed ratio vs grid size, three configs
  kernel_time_vs_floor.png     512^3 time per step: naive, v1, v2, vs the DRAM floor
  stall_breakdown.png          where warps wait: naive vs v1 vs v2
  prediction_ledger.png        the 39 written-down predictions, held vs failed
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

NAIVE = "#E24B4A"
V1 = "#8FB4DB"
V2 = "#185FA5"
OK = "#1D9E75"
BAD = "#E24B4A"
GREY = "#666666"

SIZES = [64, 128, 192, 256, 320, 384, 512]

# Tiled / naive speed ratio per grid size (CASE_STUDY 3.1, 3.1b, 3.8).
RATIO = {
    "v1, TILE_Y=4 / PENCIL_Z=8 (old guide default)": [0.42, 0.68, 0.65, 0.60, 0.59, 0.59, 0.58],
    "v1, TILE_Y=8 / PENCIL_Z=12 (best v1)": [0.49, 0.79, 0.79, 0.77, 0.78, 0.71, 0.72],
    "v2, TILE_Y=8 / PENCIL_Z=20 (current default)": [0.49, 0.97, 1.02, 0.95, 0.88, 0.86, 0.87],
}
RATIO_COLORS = [NAIVE, V1, V2]
V2_KEY = "v2, TILE_Y=8 / PENCIL_Z=20 (current default)"

# 512^3 time per step in ms (CASE_STUDY 3.1b / 3.8) and the DRAM floor (3.3).
TIME_MS = {"naive": 11.463, "tiled v1\n(8/12)": 15.992, "tiled v2\n(8/20)": 13.116}
FLOOR_PEAK_MS = 10.96  # compulsory traffic at the derived ~192 GB/s peak

# Warp stall reasons, % of warp-active cycles, one 512^3 launch each
# (CASE_STUDY 3.4 for naive and v1 8/12, 3.8 for v2). Different launches, so
# read the shape, not the second digit.
STALLS = {
    "naive": {"barrier": 0.0, "long_sb": 53.6},
    "tiled v1": {"barrier": 47.0, "long_sb": 22.6},
    "tiled v2": {"barrier": 20.2, "long_sb": 59.7},
}

# Prediction ledger outcomes by number (CASE_STUDY section 4): 39 rows, 15 failed.
FAILED = {2, 8, 10, 16, 18, 21, 23, 24, 25, 32, 33, 34, 35, 37, 39}
N_PRED = 39


def save(fig, outdir: pathlib.Path, name: str) -> None:
    out = outdir / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_ratio_by_grid(outdir):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhspan(0, 1, color="#fbeaea", zorder=0)
    ax.axhline(1.0, color="black", lw=1.2, zorder=1)
    x = list(range(len(SIZES)))
    ax.text(-0.3, 1.012, "tiled faster", fontsize=9, color=GREY, va="bottom")
    ax.text(-0.3, 0.988, "tiled slower", fontsize=9, color=GREY, va="top")
    for (label, ys), col in zip(RATIO.items(), RATIO_COLORS):
        ax.plot(x, ys, marker="o", lw=2.2, ms=6, color=col, label=label, zorder=3)
    for xi, y in zip(x, RATIO[V2_KEY]):
        dy = -15 if xi == 0 else 8  # 64^3: v1 and v2 coincide, put the label below
        ax.annotate(f"{y:.2f}", (xi, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8, color=V2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}³" for s in SIZES])
    ax.set_xlim(-0.4, len(SIZES) - 0.6)
    ax.set_ylim(0.3, 1.12)
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Tiled speed relative to naive\n(naive time / tiled time)")
    ax.set_title("Tiling never clearly wins on this GPU (RTX 4050 Laptop, 8th-order stencil)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3)
    save(fig, outdir, "tiled_vs_naive_by_grid.png")


def plot_time_vs_floor(outdir):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    names = list(TIME_MS)
    vals = list(TIME_MS.values())
    bars = ax.bar(names, vals, color=[NAIVE, V1, V2], width=0.55, zorder=3)
    ax.axhline(FLOOR_PEAK_MS, color="black", ls="--", lw=1.4, zorder=4,
               label=f"DRAM floor at peak bandwidth ({FLOOR_PEAK_MS:.2f} ms)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    for b, v in zip(bars, vals):
        ratio = vals[0] / v
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f} ms\n({ratio:.2f}× of naive)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 21)
    ax.set_ylabel("Time per step at 512³ (ms, lower is better)")
    ax.set_title("Naive already sits about 4.5% above the memory floor", fontsize=11,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    save(fig, outdir, "kernel_time_vs_floor.png")


def plot_stalls(outdir):
    fig, ax = plt.subplots(figsize=(7, 4.9))
    names = list(STALLS)
    barrier = [STALLS[n]["barrier"] for n in names]
    longsb = [STALLS[n]["long_sb"] for n in names]
    other = [max(0.0, 100 - b - l) for b, l in zip(barrier, longsb)]
    ax.bar(names, longsb, color="#BA7517", label="Waiting on global-memory loads", width=0.55)
    ax.bar(names, barrier, bottom=longsb, color="#7F4FA3",
           label="Waiting at __syncthreads() barriers", width=0.55)
    ax.bar(names, other, bottom=[l + b for l, b in zip(longsb, barrier)], color="#DDDDDD",
           label="Everything else (issuing + other stalls)", width=0.55)
    for i, (l, b) in enumerate(zip(longsb, barrier)):
        ax.text(i, l / 2, f"{l:.0f}%", ha="center", va="center", color="white", fontsize=10,
                fontweight="bold")
        if b > 4:
            ax.text(i, l + b / 2, f"{b:.0f}%", ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of warp-active cycles")
    ax.set_title("Why tiling lost: v1 spent half its time at barriers; v2 cut that by more than half",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8.5, frameon=False)
    save(fig, outdir, "stall_breakdown.png")


def plot_ledger(outdir):
    cols = 13
    rows = (N_PRED + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for n in range(1, N_PRED + 1):
        r, c = divmod(n - 1, cols)
        failed = n in FAILED
        ax.add_patch(Rectangle((c, rows - 1 - r), 0.88, 0.88, color=BAD if failed else OK))
        ax.text(c + 0.44, rows - 1 - r + 0.44, str(n), ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")
    ax.set_xlim(-0.1, cols)
    ax.set_ylim(-0.1, rows + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    held = N_PRED - len(FAILED)
    ax.set_title(f"{N_PRED} predictions written down before running: "
                 f"{held} held (green), {len(FAILED)} failed (red)", fontsize=10.5,
                 fontweight="bold")
    ax.text(cols / 2, -0.45,
            "Failures include the build-time estimate, the 255³ alignment theory, and what the\n"
            "stall profile would show. Details: docs/CASE_STUDY.md section 4.",
            ha="center", va="top", fontsize=8, color=GREY)
    save(fig, outdir, "prediction_ledger.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--outdir", default="docs/plots")
    args = ap.parse_args()
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    plot_ratio_by_grid(out)
    plot_time_vs_floor(out)
    plot_stalls(out)
    plot_ledger(out)


if __name__ == "__main__":
    main()
