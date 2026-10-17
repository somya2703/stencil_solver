#!/usr/bin/env python3
"""
scripts/make_slice_gif.py -- turn slice files into an animated GIF.

Input  : <prefix>.f32 + <prefix>.json, written by either
           ./build/bin/stencil_slice --nx 128 --steps 240 --every 4 --slice-out results/wave
           python3 scripts/reference_slice.py --out results/wave      (no GPU needed)
Output : an animated GIF of the mid-Z XY plane (red = positive, blue = negative).

Usage:
    python3 scripts/make_slice_gif.py --prefix results/wave --out docs/media/wave_slice.gif

Colour scaling: each frame is scaled by its own peak so the rings stay visible as the
pulse spreads out and weakens (--norm global uses one fixed scale for all frames).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


def load(prefix: str):
    meta = json.loads(pathlib.Path(prefix + ".json").read_text())
    nx, ny, frames = meta["nx"], meta["ny"], meta["frames"]
    data = np.fromfile(prefix + ".f32", dtype=np.float32)
    if data.size != frames * nx * ny:
        raise SystemExit(f"size mismatch: {data.size} floats, expected {frames * nx * ny}")
    return meta, data.reshape(frames, ny, nx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prefix", required=True, help="path prefix of the .f32/.json pair")
    ap.add_argument("--out", default="docs/media/wave_slice.gif")
    ap.add_argument("--norm", choices=["frame", "global"], default="frame")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--size", type=float, default=3.6, help="figure size in inches")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--title", default="3D wave pulse, middle slice")
    a = ap.parse_args()

    meta, frames = load(a.prefix)
    dt = meta.get("dt", 0.0)
    global_peak = float(np.abs(frames[1:]).max()) if len(frames) > 1 else 1.0

    fig, ax = plt.subplots(figsize=(a.size, a.size + 0.35))
    ax.set_axis_off()
    im = ax.imshow(frames[0], cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", interpolation="bilinear")
    title = ax.set_title("", fontsize=9)
    fig.suptitle(a.title, fontsize=10, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(i: int):
        f = frames[i]
        peak = global_peak if a.norm == "global" else max(float(np.abs(f).max()), 1e-12)
        im.set_data(f)
        im.set_clim(-peak, peak)
        step = i * meta["every"]
        title.set_text(f"step {step}   t = {step * dt * 1000:.0f} ms")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=a.fps), dpi=a.dpi)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
