#!/usr/bin/env python3
"""
scripts/reference_slice.py -- NumPy version of the wave solver that writes the
same slice files as the GPU tool `stencil_slice`, so the animation can be made
on any machine (no GPU needed). It is a deliberately plain re-implementation of
what the CUDA kernels compute:

    p_next = 2*p_cur - p_prev + dt^2 * v^2 * laplacian(p_cur)

8th-order finite-difference Laplacian (R = 4), a Gaussian pulse in the centre,
fixed (Dirichlet zero) boundary cells, same defaults as the C++ code
(dx = 10 m, v = 1500 m/s, dt = 0.45*dx/v, sigma = 5 cells). It mirrors
cpu_reference_step() in tests/integration/test_kernel_correctness.cpp.

Usage:
    python3 scripts/reference_slice.py --n 128 --steps 240 --every 4 --out results/wave
    python3 scripts/make_slice_gif.py  --prefix results/wave --out docs/media/wave_slice.gif

Output: <out>.f32 (float32 frames of the mid-Z XY plane, frame-major) and <out>.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

R = 4
C0 = -205.0 / 72.0
C = (8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)
CFL = 0.45


def gaussian(n: int, cx: float, cy: float, cz: float, sigma: float) -> np.ndarray:
    """Same as init_gaussian(): only interior cells are filled, halos stay zero."""
    p = np.zeros((n, n, n), dtype=np.float32)
    z, y, x = np.meshgrid(*(np.arange(R, n - R, dtype=np.float32),) * 3, indexing="ij")
    r2 = (x - cx * (n - 1)) ** 2 + (y - cy * (n - 1)) ** 2 + (z - cz * (n - 1)) ** 2
    p[R:-R, R:-R, R:-R] = np.exp(-r2 / (2.0 * sigma * sigma))
    return p


def laplacian(p: np.ndarray, inv_h2: float) -> np.ndarray:
    """8th-order Laplacian on the interior (isotropic spacing)."""
    n = p.shape[0]
    s = slice(R, n - R)
    centre = p[s, s, s]
    lap = np.float32(3.0 * C0) * centre
    for r in range(1, R + 1):
        lo, hi = slice(R - r, n - R - r), slice(R + r, n - R + r)
        lap += np.float32(C[r - 1]) * (
            p[s, s, hi] + p[s, s, lo] + p[s, hi, s] + p[s, lo, s] + p[hi, s, s] + p[lo, s, s]
        )
    return lap * np.float32(inv_h2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=128, help="grid size N (N^3 cells)")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--every", type=int, default=4, help="save a frame every N steps")
    ap.add_argument("--velocity", type=float, default=1500.0)
    ap.add_argument("--dx", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--out", default="results/wave", help="output prefix")
    a = ap.parse_args()

    n, dx = a.n, a.dx
    dt = CFL * dx / a.velocity
    v2 = np.float32(a.velocity * a.velocity)
    dt2 = np.float32(dt * dt)
    inv_h2 = 1.0 / (dx * dx)

    cur = gaussian(n, 0.5, 0.5, 0.5, a.sigma)
    prev = cur.copy()  # quiescent start, as in SolverBase::init()
    mid = n // 2

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    t0 = time.time()
    with open(out.with_suffix(".f32"), "wb") as f:

        def capture() -> None:
            nonlocal frames
            f.write(np.ascontiguousarray(cur[mid]).astype(np.float32).tobytes())
            frames += 1

        capture()
        s = slice(R, n - R)
        for step in range(1, a.steps + 1):
            nxt = np.zeros_like(cur)
            nxt[s, s, s] = np.float32(2.0) * cur[s, s, s] - prev[s, s, s] \
                + dt2 * v2 * laplacian(cur, inv_h2)
            prev, cur = cur, nxt
            if step % a.every == 0:
                capture()

    meta = {
        "nx": n, "ny": n, "nz": n, "plane": "xy", "plane_index": mid, "frames": frames,
        "every": a.every, "steps": a.steps, "dt": dt, "dx": dx, "velocity": a.velocity,
        "physics": "wave", "kernel": "numpy-reference", "dtype": "float32",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {frames} frames ({n}x{n}) to {out}.f32 / .json in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
