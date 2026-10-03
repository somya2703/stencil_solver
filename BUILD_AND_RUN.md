# Build and Run Guide (corrected)

Complete step-by-step instructions with expected output for every command.

> **This copy is tailored to a laptop-class GPU** — specifically the
> hardware detected in this environment:
>
> | | |
> |---|---|
> | **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU |
> | **Compute capability** | 8.9 |
> | **SMs** | 20 |
> | **Memory** | 6.1 GB |
> | **Peak memory bandwidth** | ~192 GB/s (measured via Nsight Compute; the commonly-quoted 96 GB/s for this chip omits GDDR6's ×2 DDR transfer factor — see the Troubleshooting table) |
> | **Peak FP32 (approx.)** | ~12 TFLOP/s *(estimate — check `nvidia-smi -q -d CLOCK` or your OEM spec sheet for the exact sustained figure)* |
>
> This revision fixes several inaccuracies found in the previous guide:
> a wrong unit-test count, a CLI flag that silently triggers a 3-way
> grid sweep instead of the single run shown, a sign error in
> `plot_results.py`'s average-speedup line, and missing tuning flags
> for the tiled kernel that this specific GPU needs. See the "What
> changed" box at the end for the full list.
>
> **A second correction pass** (also listed at the end) replaced every
> *illustrative* expected-output block below with numbers actually
> measured on the reference hardware above, fixed the peak-bandwidth
> formula, and corrected Section 4's GPU-test IDs. A fuller,
> counter-backed investigation of the naive-vs-tiled result lives in
> `docs/CASE_STUDY.md` — several notes below point to it instead of
> repeating its detail here.

---

## Prerequisites

```bash
nvidia-smi                  # must show your GPU
nvcc --version               # must be ≥ 12.0
cmake --version              # must be ≥ 3.24
ninja --version              # optional but recommended
python3 --version            # ≥ 3.9  (for plotting scripts)
```

Expected `nvidia-smi` header on this machine:
```
NVIDIA GeForce RTX 4050 Laptop GPU    Driver Version: 5xx.xx    CUDA Version: 12.x
```

A newer driver may report a higher `CUDA Version` here (e.g. `13.x`) —
that field is the newest CUDA runtime the *driver* supports, not the
installed toolkit. Check `nvcc --version` for the toolkit version
actually used to build.

If you're missing anything, use the Docker path in Section 1B — it
handles all dependencies automatically.

---

## Section 1A — Local build (you have CUDA + CMake installed)

### Step 1 — Clone and enter the repo

```bash
git clone https://github.com/somya2703/stencil_solver.git
cd stencil_solver
```

### Step 2 — (Optional) Install Conan for dependency management

```bash
pip3 install "conan>=2.0"
conan profile detect --force
conan install . --output-folder=build --build=missing \
    --settings=build_type=Release
```

If you skip Conan, CMake falls back to `FetchContent` to download
GoogleTest automatically.

### Step 3 — Configure CMake, targeting this GPU specifically

Compile only for Ada Lovelace (SM 8.9) instead of the default
architecture list (`70;75;80;86;90`) to save build time.

Also pass the tiled-kernel tuning knobs. The best configuration measured
on this GPU is `TILE_Y=8, PENCIL_Z=12`: tiled runs at about 0.71x of naive
at 512^3, so it is still *slower* than naive (see Section 5B and
`docs/CASE_STUDY.md` section 3.2 for why). The previous recommendation of
`TILE_Y=4, PENCIL_Z=8` measures worse (0.58x at 512^3), because a smaller
`TILE_Y` shrinks the useful fraction of each block (Section 8B).
`CMakeLists.txt` exposes the knobs:

```bash
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTENCIL_BUILD_TESTS=ON \
    -DSTENCIL_BUILD_BENCHMARKS=ON \
    -DSTENCIL_ENABLE_NVTX=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DSTENCIL_TILE_Y=8 \
    -DSTENCIL_PENCIL_Z=12
```

**Expected output:**
```
-- The CXX compiler identification is GNU 12.x
-- The CUDA compiler identification is NVIDIA 12.x
-- Found CUDAToolkit: .../include (found version "12.x")
...
╔══════════════════════════════════════╗
║  stencil-solver 0.1.0              ║
╠══════════════════════════════════════╣
║  Build type   : Release
║  CUDA archs   : 89
║  Precision    : FP32
║  Tile Y       : 8
║  Pencil Z     : 12
║  CPU fallback : OFF
║  MPI          : OFF
║  NCCL         : OFF
║  NVTX         : ON
║  Tests        : ON
║  Benchmarks   : ON
║  Coverage     : OFF
╚══════════════════════════════════════╝
-- Configuring done
-- Build files have been written to: .../build
```

> **If you see**: `No CMAKE_CUDA_COMPILER could be found`
> → `export PATH=/usr/local/cuda/bin:$PATH`

> **If you see**: `cmake_minimum_required ... CMake 3.24`
> → `pip3 install cmake --upgrade`

To confirm SM 8.9 is actually correct for your card:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# → 8.9
```

> **If you see**: `fatal error: nvtx3/nvtx3.hpp: No such file or directory`
> during Step 4 — `STENCIL_ENABLE_NVTX` defaults to `ON`, but the NVTX3
> header isn't part of a standard host CUDA toolkit install; it's only
> fetched automatically inside `Dockerfile.dev`. Fetch it onto your host
> once, matching what the Docker image does:
> ```bash
> sudo mkdir -p /usr/local/cuda/include/nvtx3
> sudo curl -fsSL https://raw.githubusercontent.com/NVIDIA/NVTX/v3.1.0/c/include/nvtx3/nvtx3.hpp \
>     -o /usr/local/cuda/include/nvtx3/nvtx3.hpp
> ```
> then reconfigure/rebuild — no CMake flag changes needed. Alternatively,
> configure with `-DSTENCIL_ENABLE_NVTX=OFF` if you don't need the
> Nsight Systems NVTX range markers used in Section 8.
>
> This header ships with the toolkit on some CUDA 12.8+ installs — if
> `nvtx3/nvtx3.hpp` is already present under your CUDA include path,
> this step is unnecessary; the configure/build will simply succeed.

### Step 4 — Build

```bash
cmake --build build --parallel $(nproc)
```

Measured on the reference hardware: well under a minute (4.3 s
wall-clock, 33 s of CPU time across `$(nproc)` parallel jobs). Earlier
estimates of "3–8 minutes" were too pessimistic for this codebase's
size; your own time will still vary with core count and whether a
build cache is warm.

**Build tree after completion:**
```
build/
├── bin/
│   ├── stencil_naive
│   ├── stencil_tiled
│   ├── tests/
│   │   └── ... (unit + GPU integration test binaries)
│   └── bench/
│       ├── bench_naive
│       ├── bench_tiled
│       └── bench_scaling
```

### Step 5 — Make the helper scripts executable

```bash
chmod +x scripts/*.sh
```

(Or prefix any script call with `bash`, e.g. `bash scripts/foo.sh ...`,
without changing permissions.)

---

## Section 1B — Docker build (no local CUDA toolchain needed)

```bash
docker build -f infra/docker/Dockerfile.dev -t stencil-solver:dev .

docker run --gpus all --rm -it \
    -v $(pwd):/workspace \
    stencil-solver:dev
```

Inside the container, run Steps 3–5 above identically (including the
`-DSTENCIL_TILE_Y=8 -DSTENCIL_PENCIL_Z=12` flags).

> ⚠️ **Files created inside the container are root-owned on your
> host.** By default `docker run` runs as root, so anything the
> container writes into the bind-mounted repo (`build/`,
> `nsight-output/`, etc.) can't be deleted afterward by your normal
> host user — `rm -rf build` will fail with `Permission denied` on
> individual files. Either use `sudo rm -rf <dir>` from the host when
> cleaning up, or avoid the problem entirely by running the container
> as your own user:
> ```bash
> docker run --gpus all --rm -it \
>     --user "$(id -u):$(id -g)" \
>     -v $(pwd):/workspace \
>     stencil-solver:dev
> ```

> ⚠️ **Don't reuse the same `build/` directory between this container
> path and a local host build (Section 1A).** Inside the container the
> repo is mounted at `/workspace`, so `build/CMakeCache.txt` records
> that path. If you later run `cmake -B build -S .` from the host at
> e.g. `/home/you/stencil_solver`, CMake will refuse with a
> `CMakeCache.txt directory is different` error, because it caches
> absolute source/binary paths and detects the mismatch. Either wipe
> `build/` (`sudo rm -rf build`) before switching contexts, or keep
> separate directories per context, e.g. `build-docker/` and `build/`.

---

## Section 2 — CPU-fallback build (no GPU required)

```bash
cmake -B build-cpu -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTENCIL_CPU_FALLBACK=ON \
    -DSTENCIL_BUILD_TESTS=ON \
    -DSTENCIL_BUILD_BENCHMARKS=ON

cmake --build build-cpu --parallel $(nproc)
```

The tile/pencil flags are irrelevant here — the CPU fallback path
delegates the tiled launcher straight to the naive OpenMP
implementation (see `stencil_naive.cu`'s `#else` branch), so there's
no shared-memory tiling to tune.

---

## Section 3 — Run unit tests

```bash
ctest --test-dir build \
    --output-on-failure \
    --label-regex "unit" \
    --parallel $(nproc)
```

**Expected output:**
```
Test project .../build
    Start  1: test_fd_coeffs.FDCoeffs.ConsistencyRadius1
    ...
100% tests passed, 0 tests failed out of 59
Total Test time (real) = 3–6 sec
```

(59 is the correct count across `test_fd_coeffs`, `test_grid`,
`test_timer`, `test_solver_cpu`, `test_solver`, and `test_kernels` —
matching the repo's own initial-commit message. If your run reports a
different total, it usually means one of these files wasn't compiled
into the `unit`-labeled targets in `tests/CMakeLists.txt`.)

Unit tests use grid sizes ≤ 32³ — negligible on any GPU.

---

## Section 4 — Run GPU integration tests

```bash
ctest --test-dir build \
    --output-on-failure \
    --label-regex "gpu" \
    --timeout 300
```

**Expected output:**
```
    Start 60: test_wave_3d.WaveSolver3DTest.EnergyDoesNotGrow
    ...
    Start 74: test_kernel_correctness.KernelCorrectness.LinearFieldZeroLaplacian

100% tests passed, 0 tests failed out of 15
Total Test time (real) = 1–3 sec
```

(IDs 60–74 follow directly from Section 3's 59 unit tests — CTest
numbers every test in the project sequentially, regardless of label.
Measured real time on the reference hardware was ~2 sec; the earlier
"30–60 sec" estimate was too pessimistic for grids this small.)

These use grids ≤ 64³, so memory and bandwidth headroom are not a
concern on this card.

---

## Section 5 — Run the solvers directly

### 5A — Naive solver (baseline)

⚠️ **Watch the grid-size flags here.** `main_naive.cpp` only runs the
single size you pass if it differs from the *default* 256×256×256:

```cpp
if (cfg.nx != 256 || cfg.ny != 256 || cfg.nz != 256) {
    grid_sizes = { cfg.nx };          // honors your explicit size
} else {
    grid_sizes = { 128, 256, 512 };   // default sweep — three runs!
}
```

Passing `--nx 256 --ny 256 --nz 256` (as older docs did) is
indistinguishable from passing nothing at all, so you'll silently get
a 128³/256³/512³ sweep plus a summary table instead of one run. To get
exactly one run at 256³, pick a size that isn't literally the default,
e.g.:

```bash
./build/bin/stencil_naive \
    --nx 255 --ny 255 --nz 255 \
    --steps 100 --warmup 10 \
    --physics wave \
    --verbose
```

**Measured output (reference hardware, this exact command):**
```
=== stencil-solver: naive kernel ===

GPU [0]: NVIDIA GeForce RTX 4050 Laptop GPU
  Compute: 8.9   SMs: 20   Memory: 6.1 GB
  Peak bandwidth: 192.0 GB/s

Grid: 255³   steps: 100   warmup: 10

── Grid: wave/naive ─────────────────────
  Total cells   : 255 × 255 × 255  =  16.58 M
  Interior      : 247 × 247 × 247  =  15.07 M
  Spacing       : dx=10.00 m  dy=10.00 m  dz=10.00 m
  Time step     : dt=3.0000e-03 s
  Stencil radius: R=4  (8th order)
  Memory/field  : 66.33 MB
  Fields (3×p + vel2): 265.30 MB

  wave/naive               1.572 ms/step   1073.6 GB/s    536.8 GFLOP/s
```

`--warmup 10` is short: a cold process or one following a smaller run
can read up to ~12% slower here than the steady state reached with a
longer warmup (e.g. `--warmup 300`). Treat single-digit-warmup timings
as approximate; Section 6's benchmarks use `--warmup 20` over 200
timed steps and are the more reliable figures for comparison.

**Use `--warmup 100` or more for steady-state numbers.** A cold process
with `--warmup 10` can read up to ~12% slower than the steady state
(256³: ~1.57 ms at warmup 10 vs 1.40 ms at warmup 300). The command above
keeps `--warmup 10` only so the output matches the block shown.

If you *do* want the 128³/256³/512³ sweep on purpose, just omit
`--nx/--ny/--nz` entirely (or set exactly 256/256/256), and expect
three result lines plus a "── Summary ──" table, not the single-line
output shown above.

> With 20 SMs vs. an A100's 108 (~5.4×) and ~192 GB/s vs. 2,039 GB/s
> (~10.6×), a naive kernel result in the low-millisecond range for a
> 256-ish grid is expected, not a regression. What matters is the
> **relative** naive→tiled speedup, not the absolute ms/step compared
> to datacenter hardware.

### 5B — Tiled solver (naive vs tiled comparison)

At 512³, four fp32 fields cost roughly
`512³ × 4 bytes × 4 fields ≈ 2.1 GB` — comfortably inside 6.1 GB, but
close unnecessary GPU consumers (browser, display compositor) before
large runs.

```bash
./build/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 \
    --steps 200 --warmup 20
```

> **Note on tile configuration.** The 5B, 6B and 6C output blocks below
> were captured with `TILE_Y=4, PENCIL_Z=8`, the guide's earlier Step 3
> setting. Step 3 now builds `8/12`, which is faster for the tiled kernel
> (512^3: 16.03 ms vs 19.91 ms, 0.72x vs 0.58x of naive). With the new
> build expect the banner to read `Tile dims : 32 x 8 (SMEM block: 40 x 16)`
> and `Z pencil depth : 12`, the tiled rows to be roughly 20% lower, and the
> naive rows unchanged. The other grid sizes have not been re-measured at
> 8/12 yet; re-run them and replace these blocks when convenient.

**Measured output (reference hardware, captured at `TILE_Y=4, PENCIL_Z=8`):**
```
=== stencil-solver: naive vs tiled comparison ===

GPU [0]: NVIDIA GeForce RTX 4050 Laptop GPU  (6.1 GB  |  peak BW: 192.0 GB/s)

Stencil radius : 4  (8th-order)
Tile dims      : 32 × 4  (SMEM block: 40 × 12)
Z pencil depth : 8

Grid        Naive ms   BW (GB/s)    Tiled ms   BW (GB/s)   Speedup
--------  ----------  ----------  ----------  ----------  --------
512         11.467 ms    1250.5       19.907 ms     720.3       0.58×

── Naive ────────────────────────────────────────────────

Kernel                         ms/step        GB/s       GFLOP/s   Speedup
--------------------------  ----------  ----------  ------------  --------
  naive                       11.467 ms    1250.5          625.2      1.00×

── Tiled ────────────────────────────────────────────────

Kernel                         ms/step        GB/s       GFLOP/s   Speedup
--------------------------  ----------  ----------  ------------  --------
  tiled                       19.907 ms     720.3          360.1      1.00×
```

**Tiled is slower than naive on this GPU — at this configuration, and
at every `TILE_Y`/`PENCIL_Z` combination tested, including after a
wider sweep.** This is a reproducible, counter-confirmed result, not a
sign that something is misconfigured. The best configuration found so
far is `TILE_Y=8, PENCIL_Z=12` at ≈0.71× (still a loss). Naive is
already within ~4.5% of this GPU's practical DRAM-bandwidth floor, so
no amount of tile tuning should be expected to make tiled win at this
grid size and radius. For the counter-level explanation (barrier
stalls, DRAM-traffic increase, where occupancy does and doesn't
matter), see `docs/CASE_STUDY.md` §3.

> **Memory ceiling on this card:** don't jump straight to 1024³ —
> `1024³ × 4 bytes × 4 fields ≈ 17.2 GB`, which will not fit in 6.1 GB.
> Stay at or below roughly 640³ for single-run headroom
> (`640³ × 4 × 4 ≈ 4.2 GB`).

---

## Section 6 — Run the benchmarks

### 6A — Naive benchmark (grid size sweep)

The default sweep (`64, 128, 192, 256, 320, 384, 512`) is safe on
6.1 GB.

```bash
mkdir -p results
./build/bin/bench/bench_naive \
    --steps 200 --warmup 20 \
    --output results/bench_naive.json
```

**Measured output (reference hardware):**
```
=== bench_naive — global-memory stencil ===

Device : NVIDIA GeForce RTX 4050 Laptop GPU
Memory : 6.1 GB   Peak BW: 192.0 GB/s

Radius : 4  (8th order)
Steps  : 200  Warmup: 20

Grid         ms/step        GB/s       GFLOP/s
--------  ----------  ----------  ------------
64           0.012 ms    1665.4          832.7
128          0.187 ms    1036.7          518.4
192          0.664 ms    1050.4          525.2
256          1.403 ms    1217.4          608.7
320          2.779 ms    1224.2          612.1
384          4.797 ms    1241.2          620.6
512         11.462 ms    1250.9          625.5

Results written to: results/bench_naive.json
```

> ⚠️ **Small-grid GB/s numbers are inflated, not a real memory-bandwidth
> measurement — and this holds even after the peak-bandwidth fix
> above.** `bandwidth_bytes_per_step()` in `types.hpp` assumes zero
> cache reuse between steps. At 64³ the whole working set fits
> comfortably in L2, so most traffic is served from cache, not DRAM —
> the reported figure (1665 GB/s above) is roughly 9× the corrected
> ~192 GB/s physical peak. This isn't just a small-grid artifact: even
> at 512³, Nsight Compute shows naive's *measured* DRAM throughput at
> 95.7% of peak while this same model-based number reads 1250 GB/s —
> about 6.5× too high. Treat every "GB/s" and "BW Eff." figure in this
> repo's console/JSON output as a model number (closer to L2 traffic),
> not a DRAM measurement; see `docs/CASE_STUDY.md` §3.3 and
> Troubleshooting below.

### 6B — Tiled benchmark (naive vs tiled comparison)

```bash
./build/bin/bench/bench_tiled \
    --steps 200 --warmup 20 \
    --output results/bench_tiled.json
```

**Measured output (reference hardware, captured at `TILE_Y=4, PENCIL_Z=8`):**
```
=== bench_tiled — naive vs shared-memory tiled ===

Device : NVIDIA GeForce RTX 4050 Laptop GPU
Memory : 6.1 GB   Peak BW: 192.0 GB/s

Radius : 4  (8th order)
Steps  : 200  Warmup: 20

Grid        Naive ms  Naive GB/s    Tiled ms  Tiled GB/s   Speedup
--------  ----------  ----------  ----------  ----------  --------
64           0.012 ms    1663.5        0.028 ms     706.8       0.42×
128          0.187 ms    1036.5        0.276 ms     702.4       0.68×
192          0.632 ms    1104.0        0.975 ms     715.9       0.65×
256          1.401 ms    1219.8        2.353 ms     725.9       0.60×
320          2.759 ms    1233.0        4.654 ms     730.8       0.59×
384          4.799 ms    1240.6        8.102 ms     734.8       0.59×
512         11.477 ms    1249.3       19.892 ms     720.8       0.58×

Results written to: results/bench_tiled.json
Results written to: results/bench_tiled_naive.json
```

**Tiled is below 1.0× at every grid size shown above — this is the
expected, reproducible result on this GPU class**, not a tuning
mistake you haven't found yet (it holds across a wider sweep; see
Section 8B and `docs/CASE_STUDY.md` §3.2). The per-run "Summary"
tables (the `── Naive ──` / `── Tiled ──` blocks a full sweep run
prints below the main table) have a separate, known bug: their
own Speedup column compares every row against the *first* row
regardless of grid size, producing meaningless values like `0.06×`,
`0.02×`, `0.00×`. Ignore that column; use the `ms/step` and `GB/s`
columns per row instead (tracked as checklist item M1).

### 6C — Scaling benchmark (single-GPU roofline)

```bash
./build/bin/bench/bench_scaling \
    --steps 200 --warmup 20 \
    --output results/bench_scaling.json
```

**Measured output (reference hardware):**
```
=== bench_scaling — single-GPU roofline ===

Peak BW   : 192.0 GB/s
Kernel    : tiled
Radius    : 4  (8th-order)

Grid         ms/step        GB/s     BW Eff.
--------  ----------  ----------  ----------
64           0.028 ms     710.9        370.2%
96           0.096 ms     791.2        412.1%
128          0.278 ms     696.5        362.8%
192          0.977 ms     714.0        371.9%
256          2.370 ms     720.8        375.4%
384          8.109 ms     734.2        382.4%
512         19.935 ms     719.3        374.6%

Scaling results written to: results/bench_scaling.json
```

> ⚠️ **`BW Eff.` is not a trustworthy efficiency percentage, even
> after the peak-bandwidth fix.** It divides the model byte-count
> above (closer to L2 traffic than DRAM traffic) by the peak, so it
> still reads well over 100%. This is a known, unfixed limitation
> (checklist item L1) — don't use this column as a real
> bandwidth-efficiency figure until the numerator is replaced with a
> measured DRAM-byte count. See `docs/CASE_STUDY.md` §3.3.

---

## Section 7 — Generate plots and report

```bash
python3 scripts/plot_results.py \
    --naive   results/bench_tiled_naive.json \
    --tiled   results/bench_tiled.json \
    --peak-bw 192 \
    --peak-flops 12 \
    --outdir  docs/plots \
    --report  docs/BENCHMARKS.md
```

**Expected output:**
```
Saved: docs/plots/throughput_comparison.png
Saved: docs/plots/speedup.png
Saved: docs/plots/roofline.png
Report written: docs/BENCHMARKS.md
```

The average-speedup sign bug that affected earlier versions of
`scripts/plot_results.py` (the summary line computing `tiled/naive`
instead of `naive/tiled`, so it could print a misleadingly positive
number — e.g. "1.56×" — over a table where every row showed tiled
slower) is **fixed** as of this repo's current `plot_results.py`. The
per-row table and the "Average speedup" line at the bottom of
`docs/BENCHMARKS.md` now agree: given the 6B data above, both describe
tiled as roughly 0.59× of naive on average. If you ever see them
disagree again, check that `write_markdown_report()` reads
`speedups.append(n_r["time_ms"] / t_r["time_ms"])`, not the reciprocal.

> Use this card's real peak bandwidth/FLOPs (`192`/`12`), not the
> A100 defaults (`2039`/`312`) — otherwise the roofline ceiling is
> drawn for hardware you don't have.
>
> Even with the corrected peak-bw, the roofline plot will show every
> naive and tiled point sitting *above* the drawn memory roof. That's
> the same model-vs-measured-bytes issue as `BW Eff.` above (checklist
> item L1) — the plotted "GFLOP/s" values are real, but the x-axis
> arithmetic intensity is computed against modeled bytes, not measured
> DRAM bytes, so the roof itself is drawn too low. Treat the roofline
> plot as illustrative only until L1 is fixed.

---

## Section 8 — Nsight profiling

### 8A — One-time setup

```bash
chmod +x scripts/*.sh
```

If `ncu` refuses to attach with a permissions error:
```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
```

On the reference hardware (driver 595.84), `ncu` worked immediately
with no sysctl change needed — `RmProfilingAdminOnly` in
`/proc/driver/nvidia/params` was already `0`. That driver parameter,
not `perf_event_paranoid`, is what actually gates `ncu`'s GPU
performance-counter access; `perf_event_paranoid` instead governs
CPU-side sampling (relevant to `nsys`, not `ncu`). Try `ncu` first;
only apply the fix above if you hit `ERR_NVGPUCTRPERM` in its output.

### 8B — Parameter sweep before profiling

Don't jump straight into `ncu`/`nsys` on the default tuning — establish
which `TILE_Y`/`PENCIL_Z` is actually best for your card first, since
profiling a mistuned config just documents the mistuning. Only test
`TILE_Y ≥ 8`: the kernel pays a fixed `2·STENCIL_RADIUS = 8`-cell halo
border in both X and Y regardless of tile size, so shrinking `TILE_Y`
below 8 shrinks the *useful* fraction of every block faster than it
shrinks the overhead — `TILE_Y=4` measures worse than the default, not
better. Also respect the hard ceiling: `TILE_X=32` is fixed (warp size),
and CUDA caps blocks at 1024 threads, so `40 × (TILE_Y+8) ≤ 1024` means
`TILE_Y ≤ 17` — anything at or above `32` fails at launch with
`invalid configuration argument`.

```bash
mkdir -p tuning_results
echo "tile_y,pencil_z,naive_ms,tiled_ms,speedup" > tuning_results/sweep.csv

for ty in 8 16; do
  for pz in 8 12 16 20 24; do
    echo "=== TILE_Y=$ty PENCIL_Z=$pz ==="
    rm -rf build-tune
    cmake -B build-tune -S . -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="89" \
        -DSTENCIL_TILE_Y=$ty -DSTENCIL_PENCIL_Z=$pz \
        -DSTENCIL_BUILD_TESTS=OFF -DSTENCIL_BUILD_BENCHMARKS=OFF > /dev/null 2>&1
    cmake --build build-tune --parallel $(nproc) --target stencil_tiled > /dev/null 2>&1

    OUT=$(./build-tune/bin/stencil_tiled --nx 512 --ny 512 --nz 512 --steps 200 --warmup 20)
    NAIVE_MS=$(echo "$OUT" | grep "^  naive" | awk '{print $2}')
    TILED_MS=$(echo "$OUT" | grep "^  tiled" | awk '{print $2}')
    echo "$ty,$pz,$NAIVE_MS,$TILED_MS,$(echo "$NAIVE_MS $TILED_MS" | awk '{print $1/$2}')" \
        >> tuning_results/sweep.csv
  done
done

column -s, -t tuning_results/sweep.csv
```

(`sudo` before `rm -rf build-tune` has been dropped — a normal local
build doesn't need it. If an earlier Docker run left `build-tune/` or
`build-tuned/` root-owned, fix that once with
`sudo chown -R "$USER:$USER" build-tune*` rather than running every
sweep iteration as root.)

**Measured result on this GPU:** every configuration in the swept
range showed tiled *slower* than naive (speedup < 1.0×). Across a
wider sweep than the loop above (`TILE_Y` ∈ {4, 8, 16}, `PENCIL_Z`
stepped from 8 to 32), the best found was **`TILE_Y=8, PENCIL_Z=12`
at ≈0.71×** (confirmed over five repeated 200-step runs) — not
`TILE_Y=16, PENCIL_Z=24` (≈0.69×), which is close but not optimal.
This is not a broken sweep; it's a real, reproducible finding on this
Ada-class laptop GPU. The full sweep table, repeat-run confirmation,
and the counter-level explanation (barrier stalls, DRAM-traffic
increase, where occupancy does and doesn't matter, and two
still-unexplained steps at `PENCIL_Z=16→20` and `28→32`) are in
`docs/CASE_STUDY.md` §3.2–§3.6 — start there before re-deriving this
from scratch.

### 8C — Isolating a single kernel for Nsight Compute

`stencil_tiled`'s binary always runs `kernel_naive` first, then
`kernel_tiled`, in one process (that's how it prints its own
naive-vs-tiled comparison). This matters for profiling: `ncu`'s
`--launch-count 1` (used by `scripts/profile_nsight.sh` to avoid the
kernel-replay timing inflation described below) stops after the
**first** launch encountered in the whole process — which is always
`kernel_naive` — so running the wrapper script against `stencil_tiled`
never actually profiles the tiled kernel itself.

To profile `kernel_tiled` specifically, skip the first launch and run
the binary at `--steps 1 --warmup 0` so exactly two launches occur
(one naive, one tiled) with no ambiguity about which is which:

```bash
ncu --launch-skip 1 --launch-count 1 \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
launch__registers_per_thread,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    -o nsight-output/tiled_only \
    ./build-best/bin/stencil_tiled --nx 512 --steps 1 --warmup 0

ncu --launch-count 1 -o nsight-output/naive_only \
    ./build-best/bin/stencil_naive --nx 512 --steps 1 --warmup 0
```

(`build-best` here means whatever `TILE_Y`/`PENCIL_Z` your Section 8B
sweep found best — reconfigure a dedicated build directory with those
flags first if you haven't already. A kernel-name filter,
`ncu -k kernel_tiled --launch-count 1 ...`, is an equally valid
alternative to `--launch-skip`/`--launch-count` and was used
throughout `docs/CASE_STUDY.md`.)

> ⚠️ **The `ms/step` console output during either of these `ncu` runs
> is not real timing** — Nsight Compute replays the kernel multiple
> times (you'll see `"kernel_tiled" - 0 (1/1): ... - 7 passes` or
> similar) to collect the full metric set, and that replay overhead is
> counted inside the binary's own `GpuTimer`. A tiled kernel that
> normally takes ~17ms can print `500+ ms` here — that's an artifact of
> profiling instrumentation, not a real regression. The metrics stored
> in the `.ncu-rep` file are unaffected and are what you actually want;
> ignore the console `ms/step`/`GB/s`/speedup line entirely when `ncu`
> is attached. Use Section 6's benchmark binaries (no `ncu`) for real
> timing. (Nsight Compute's own single-launch *reported* duration,
> `gpu__time_duration.sum`, is a separate, trustworthy number — but
> even that can vary run-to-run by several percent and shouldn't be
> used alone to rank tile configurations; use Section 8B's benchmark
> sweep for ranking and `ncu` for counter ratios.)

### 8D — Comparing the two reports in `ncu-ui`

```bash
ncu-ui nsight-output/naive_only.ncu-rep
```

You may see `qt.glx: ... Could not initialize GLX` /
`Falling back to Mesa software rendering` — this is a hardware-GL
context warning, not a failure; the UI works, just slower (software
rendered).

`docs/nsight_workflow.md` describes adding a baseline via
**File → Add Baseline**, and an earlier revision of this guide
described a specific "Add Current Result as Baseline" toolbar tooltip
— neither was found in Nsight Compute UI 2025.1.1.0 on the reference
hardware. The exact mechanism is version-dependent; open the
**Baselines** panel at the bottom of the window and explore its
toolbar directly rather than following a fixed set of steps.

A more reliable, version-independent route, validated on this
hardware, is the CLI/text output instead of the baseline UI:

```bash
# Rule-engine findings (source-level optimization hints) for a report:
ncu -i nsight-output/naive_only.ncu-rep --page details \
    2>&1 | grep -A4 -E "^\s+(OPT|WRN)"

# Per-SASS-line stall attribution (needs --set full --import-source yes
# when generating the report):
ncu -i nsight-output/tiled_only.ncu-rep --page source \
    --print-source sass --csv > tiled_only.sass.csv
```

The second command's CSV has a `Source` column and a
`Warp Stall Sampling (All Samples)` column; sorting it descending (or
scripting it) shows exactly which instructions the profiler caught
warps waiting on, which is more informative than the Summary tab's
headline diff numbers.

**What this showed on the reference hardware** (full detail in
`docs/CASE_STUDY.md` §3.4–§3.5): tiled kernels read **more** DRAM than
naive (+12% to +38%, not less — the kernel's own design goal), spend
33–51% of warp-active cycles stalled at `__syncthreads()` barriers
(vs. 0% for naive), and execute 1.6–2.4× the warp instructions per
output cell. Registers and occupancy are **not** a consistent
explanation: the lowest-occupancy configuration swept (62.5%, at
`TILE_Y=16`) is *faster* than several higher-occupancy configurations,
and occupancy only becomes the dominant factor once `PENCIL_Z ≥ 28`
(where the register count jumps and a 640-thread block drops from two
per SM to one). Don't expect a single clean "more registers → lower
occupancy → slower" story across every configuration; the real
picture is more specific than that, and is written up with evidence
tags (verified / derived / inferred / unverified) in the case study.

### 8E — Nsight Systems: end-to-end timeline, no replay overhead

Unlike `ncu`, Nsight Systems does **not** replay kernels — timings
here are real and can be trusted directly, including for host-side
scheduling analysis that `ncu`'s single-launch snapshots can't provide.

```bash
./scripts/profile_nsight.sh systems ./build-best/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 --steps 50 --warmup 10
```

Pull the aggregate kernel timing and host-API summary straight from
the generated `.sqlite`, without needing the GUI:

```bash
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
    nsight-output/<timestamp>/stencil_tiled.nsys-rep
```

In `cuda_gpu_kern_sum`, check `kernel_naive` vs `kernel_tiled`'s
**Avg (ns)**, aggregated across all 60 launches (10 warmup + 50
timed) — this should closely match the ratio from your Section 8B
sweep and confirms it isn't a single-run fluke. On the reference
hardware, tiled's aggregate mean (16.64 ms) matched the Section 8B
sweep closely (16.7 ms); naive's aggregate mean (12.19 ms) ran a few
percent above Section 6's steady-state benchmark (11.46 ms) because
the mean includes the slower early launches in the 10-launch warmup
window — use the median (11.84 ms here) or Section 6's dedicated
benchmark binary for the cleanest headline number.

In `cuda_api_sum`, specifically check `cudaStreamSynchronize` and
`cudaEventSynchronize` call counts: on a correctly-implemented timed
loop (per `SolverBase::run()`'s design) there should be only a
handful of calls total — matching one-time IC-upload syncs and the
final blocking timer read — and **none** occurring between the
per-step kernel launches themselves. Confirmed on the reference
hardware: exactly 4 `cudaStreamSynchronize` and 2
`cudaEventSynchronize` calls across the full 120-launch run (both
kernels, warmup + timed), none interleaved with the launches. This
rules out host-side scheduling stalls as a contributor to the measured
slowdown, isolating the cause to the kernels' own execution efficiency
(registers/occupancy/stalls, per 8D) rather than CPU-GPU round-trip
overhead.

To view the timeline visually instead of (or in addition to) the CLI
stats:
```bash
nsys-ui nsight-output/<timestamp>/stencil_tiled.nsys-rep
```
Same Mesa software-rendering fallback as `ncu-ui` is likely, same
non-issue. Zoom into the GPU kernel row (right-click-drag, or
scroll-wheel) during the timed loop — you should see kernel blocks
packed edge-to-edge with no visible idle gaps between them, the
visual counterpart of the "zero in-loop syncs" finding above. A
GPU-metrics-sampled capture (`nsys profile --gpu-metrics-device=0 ...`,
not enabled by the wrapper script by default) would additionally show
a time-resolved SM/warp-occupancy graph, if you want that level of
detail.

---

## Section 9 — Multi-GPU build and scaling

This is a single-GPU laptop, so `stencil_multigpu` will only ever run
with `-n 1` — there's no second device to exercise the halo exchange
path meaningfully. To smoke-test the MPI code path:

```bash
cmake -B build-mpi -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTENCIL_ENABLE_MPI=ON \
    -DSTENCIL_ENABLE_NVTX=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DSTENCIL_TILE_Y=8 \
    -DSTENCIL_PENCIL_Z=12

cmake --build build-mpi --parallel $(nproc)

mpirun -n 1 ./build-mpi/bin/stencil_multigpu \
    --nx 256 --ny 256 --nz 256 --steps 50 --warmup 10
```

Real weak/strong scaling curves require multiple GPUs and aren't
reproducible on this machine — skip `run_scaling.sh` and
`merge_scaling.py` unless you have access to a multi-GPU host or cloud
instance.

*(This section, and the multi-GPU code path it exercises, has not
been validated in the current investigation — see `docs/CASE_STUDY.md`
§7 for several suspected issues found by reading the source, none of
which have been run or confirmed.)*

---

## Section 10 — CI locally (reproduce what GitHub Actions runs)

```bash
docker build -f infra/docker/Dockerfile.ci -t stencil-solver:ci .
docker run --rm stencil-solver:ci
```

CI uses `STENCIL_CPU_FALLBACK=ON`, so this step is identical
regardless of what GPU you have.

*Note: the workflow files that are supposed to run this automatically
on GitHub currently live under `infra/.github/workflows/`, which
GitHub does not read (it only looks at a root-level `.github/`). CI is
not actually running on pushes/PRs until that's fixed — out of scope
for this guide; tracked separately.*

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `CMake Error: The current CMakeCache.txt directory ... is different than the directory ... where CMakeCache.txt was created` | The same `build/` directory was configured from two different absolute paths — e.g. once inside the Docker container (mounted at `/workspace`) and once on the host (`/home/you/stencil_solver`). CMake caches absolute source/binary paths and refuses to reuse a mismatched cache | `rm -rf build` and reconfigure from scratch. Going forward, use separate build directories for Docker vs. host builds (e.g. `build-docker/` and `build/`) so their caches never collide |
| `./scripts/*.sh: Permission denied` | Scripts aren't marked executable in a fresh checkout | `chmod +x scripts/*.sh`, or run via `bash scripts/foo.sh ...` |
| `stencil_naive` prints three grid sizes and a summary table when you only wanted one | `--nx/--ny/--nz` all equal the default 256, so `main_naive.cpp` falls into its sweep branch | Pass a size that differs from 256 (e.g. `--nx 255`), or accept the sweep |
| Tiled kernel benchmarks slower than naive across the board | This is the measured, reproducible result on this GPU class, confirmed across a wide `TILE_Y`/`PENCIL_Z` sweep — not a tuning mistake | See `docs/CASE_STUDY.md` for the full sweep and the counter-level cause (barrier stalls, extra DRAM traffic); the best configuration found so far is `TILE_Y=8, PENCIL_Z=12` (≈0.71×, still a loss) |
| Device banner prints `Peak bandwidth: 96.0 GB/s` | The `memoryClockRate * memoryBusWidth / 8` formula in `src/main_*.cpp` and `tests/perf/bench_*.cpp` omits GDDR's ×2 DDR transfer factor | Fixed in the current repo (look for a commit titled "Fix peak DRAM bandwidth..."); expect `192.0 GB/s` after rebuilding. If you still see `96.0`, you're on an older commit or a stale build directory |
| `docs/BENCHMARKS.md` "Average speedup" doesn't match the per-row table's sign | `write_markdown_report()` in `plot_results.py` averaged `tiled/naive` instead of `naive/tiled` | Fixed in the current repo; if you still see this, check `scripts/plot_results.py` for `speedups.append(n_r["time_ms"] / t_r["time_ms"])` |
| A multi-size sweep's "Summary" table shows a Speedup column of `1.00×, 0.12×, 0.02×, 0.00×, ...` | `format_perf_table()` compares every row to row 0 regardless of grid size — not a real speedup series | Known limitation (checklist item M1), not yet fixed; read the `ms/step`/`GB/s` columns per row instead and ignore that column |
| `ctest --label-regex unit` doesn't report 59 tests | A test file wasn't wired into `add_unit_test(...)` in `tests/CMakeLists.txt`, or you're comparing against an older/incorrect expected count | Re-check `tests/CMakeLists.txt`; 59 is the correct total for `test_fd_coeffs`, `test_grid`, `test_timer`, `test_solver_cpu`, `test_solver`, `test_kernels` |
| Reported GB/s exceeds this GPU's peak at small grid sizes (or even at 512³) | `bandwidth_bytes_per_step()` assumes zero cache reuse; the resulting number tracks L2 traffic, not DRAM bandwidth, at any grid size | Treat every "GB/s"/"BW Eff." figure as a model number, not a DRAM measurement, until checklist item L1 lands; see `docs/CASE_STUDY.md` §3.3 |
| `bench_scaling`'s `BW Eff.` column reads well above 100% (even 700%+) | Same model-vs-measured-bytes issue as above; the peak-bandwidth fix alone doesn't correct the numerator | Known limitation (checklist item L1); don't trust this column yet |
| `docs/plots/*.png` or `docs/BENCHMARKS.md` fails to write with `PermissionError` | A prior Docker run (without `--user "$(id -u):$(id -g)"`) left `docs/plots/` (or another output directory) root-owned | `sudo chown -R "$USER:$USER" docs/plots` (or whichever directory is affected) |
| `./scripts/profile_nsight.sh compare` prints a diff table with every metric showing `—` | `ncu --csv` emits long-format CSV (one row per metric, generic `"Metric Name"`/`"Metric Value"` columns), but the script's `extract()` looks up metric names as if they were column headers, so every lookup silently returns nothing | Use the manual `ncu`/`ncu-ui` workflow in Section 8C–8D instead; if patching the script, pivot parsed rows into `{row["Metric Name"]: row["Metric Value"]}` before any `.get()` lookups |
| `ncu --launch-count 1` on `stencil_tiled` only ever profiles `kernel_naive`, never `kernel_tiled` | `stencil_tiled`'s binary always launches naive before tiled in one process; `--launch-count 1` stops at the very first launch in the whole run | Use `--launch-skip 1 --launch-count 1` (or `-k kernel_tiled`) with `--steps 1 --warmup 0` on the binary, per Section 8C |
| `ncu: unrecognised option '--output'` | Old script used `--output` instead of `-o` | Already fixed in current `scripts/profile_nsight.sh` — verify you're on the current version |
| `ncu: Failed to find metric regex ...l1tex__hit_rate...` | Bare `l1tex__hit_rate` needs a rollup suffix | Already fixed to `l1tex__hit_rate.pct` in the current script |
| `ModuleNotFoundError: No module named 'networkx'` | Stray `from networkx import radius` import | Already removed from the current `plot_results.py` |
| `CUDA_ARCH not supported` | Built for archs that don't include your card | `-DCMAKE_CUDA_ARCHITECTURES="89"` for this card |
| `cudaErrorMemoryAllocation` on large grids | 6.1 GB VRAM ceiling — `N³ × 4 bytes × 4 fields` grows fast | Stay at or below ~640³; close other GPU consumers first |
| `nvcc: error: unrecognized option '--expt-relaxed-constexpr'` | nvcc < 11 | Upgrade CUDA toolkit to ≥ 12.0 |
| `MPI_Init failed` | OpenMPI not installed | `apt install libopenmpi-dev openmpi-bin` |
| `ncu: permission denied` / `ERR_NVGPUCTRPERM` | Nsight Compute needs elevated GPU performance-counter access (gated by the driver's `RmProfilingAdminOnly` parameter, not `perf_event_paranoid`) | `sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'` as a first try; if `ncu` still reports `ERR_NVGPUCTRPERM`, the driver-level setting needs changing instead (see NVIDIA's developer tools docs) — not needed at all on some driver versions (see Section 8A) |

---

## Summary: what each binary does

| Binary | Input | Output |
|--------|-------|--------|
| `stencil_naive` | Grid size, steps, physics flags | Console: ms/step, GB/s, GFLOP/s. Optional: JSON. **Runs a 3-size sweep if grid == 256³** |
| `stencil_tiled` | Same | Console: naive vs tiled table with speedup. Two JSON files |
| `stencil_multigpu` | Same + MPI rank count | Single-rank smoke test only on this hardware |
| `bench_naive` | Grid sweep or single size | Console: 7-row sweep table (safe up to 512³ on 6.1 GB). JSON |
| `bench_tiled` | Same | Console: naive+tiled side-by-side table. Two JSON files |
| `bench_scaling` | Grid sweep | Console: BW efficiency % table (relative to ~192 GB/s here; see the `BW Eff.` caveat in 6C) |
| `test_*` | (none) | GoogleTest pass/fail output — 59 unit tests, 15 GPU tests |

---

## Quick reference

```bash
# One-time setup
chmod +x scripts/*.sh

# Configure for this specific GPU, with tiled-kernel tuning
cmake -B build -S . -DSTENCIL_BUILD_TESTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DSTENCIL_TILE_Y=8 -DSTENCIL_PENCIL_Z=12
cmake --build build --parallel $(nproc)

# Unit tests (expect 59/59)
ctest --test-dir build --output-on-failure --label-regex unit --parallel $(nproc)

# Single naive run at a size that isn't the 256 default
./build/bin/stencil_naive --nx 255 --ny 255 --nz 255 --steps 100 --warmup 10 --verbose

# Benchmarks (safe grid range: up to 512³)
./build/bin/bench/bench_tiled --steps 200 --warmup 20 --output results/bench_tiled.json

# Plots + report, using this card's real peak BW/FLOPs
python3 scripts/plot_results.py \
    --naive results/bench_tiled_naive.json \
    --tiled results/bench_tiled.json \
    --peak-bw 192 --peak-flops 12 \
    --outdir docs/plots --report docs/BENCHMARKS.md

# Nsight Compute — isolate and compare a single kernel launch (see Section 8B–8D)
ncu --launch-skip 1 --launch-count 1 -o nsight-output/tiled_only \
    ./build-best/bin/stencil_tiled --nx 512 --steps 1 --warmup 0
ncu --launch-count 1 -o nsight-output/naive_only \
    ./build-best/bin/stencil_naive --nx 512 --steps 1 --warmup 0
ncu-ui nsight-output/naive_only.ncu-rep   # open tiled_only.ncu-rep too; see Section 8D

# Nsight Systems — real aggregate timing, no replay overhead (see Section 8E)
./scripts/profile_nsight.sh systems ./build-best/bin/stencil_tiled --nx 512 --steps 50 --warmup 10
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum nsight-output/<timestamp>/stencil_tiled.nsys-rep
```

For the full counter-backed investigation behind the numbers in this
guide — including the two still-unexplained performance steps and a
prediction-vs-measurement ledger — see `docs/CASE_STUDY.md`.

---

## What changed from the previous version

1. **Section 1A** — added `-DSTENCIL_TILE_Y=4 -DSTENCIL_PENCIL_Z=8` to
   the configure step, with an explanation of why the A100-sized
   defaults are a poor fit for a 20-SM part.
2. **Section 3** — corrected expected unit-test count from 43 to 59.
3. **Section 5A** — replaced the `--nx 256 --ny 256 --nz 256` example
   (which silently triggers a 3-size sweep) with `--nx 255 --ny 255
   --nz 255`, and explained the sweep-vs-single-run branch in
   `main_naive.cpp`.
4. **Section 5B / 6B** — added a note that a consistently sub-1×
   tiled/naive speedup indicates mistuned `TILE_Y`/`PENCIL_Z`, not a
   broken kernel.
5. **Section 6A** — added a caveat that small-grid GB/s figures are
   inflated by L2 cache reuse and can exceed the card's physical peak.
6. **Section 7** — flagged the sign error in
   `write_markdown_report()`'s average-speedup calculation
   (`tiled/naive` instead of `naive/tiled`), with the one-line fix.
7. **Troubleshooting table** — added rows for all of the above,
   keeping the previously-fixed `ncu`/`networkx` issues for reference.
8. **Section 1B / Troubleshooting** — added a warning against reusing
   the same `build/` directory across a Docker-container build and a
   local host build, since CMake caches absolute source/binary paths
   and will error out (`CMakeCache.txt directory is different`) if
   they don't match — a trap when following Section 1B then 1A
   back-to-back.
9. **Section 1A** — added instructions for fetching the `nvtx3.hpp`
   header manually on host builds, since `STENCIL_ENABLE_NVTX=ON`
   (the default) fails to compile without it and the header is only
   auto-fetched inside `Dockerfile.dev`.
10. **Section 1B** — added a warning and fix for root-owned files left
    behind by container writes into the bind-mounted repo, including
    the `--user "$(id -u):$(id -g)"` prevention tip.
11. **Section 8 — substantially rewritten.** The original 8B
    (`profile_nsight.sh compare`) diff table is broken — it always
    prints `—` for every metric due to a CSV long/wide-format parsing
    bug in the script, and even when working, `--launch-count 1`
    silently profiles the wrong kernel (`kernel_naive`, not
    `kernel_tiled`) on the `stencil_tiled` binary specifically, since
    that binary always launches naive first. Replaced with: a
    parameter-sweep step (8B) to find the best `TILE_Y`/`PENCIL_Z`
    before profiling at all; a corrected single-kernel isolation
    method using `--launch-skip`/`--launch-count`/`--warmup 0` (8C);
    a corrected `ncu-ui` baseline-comparison walkthrough matching the
    current UI, since `docs/nsight_workflow.md`'s "File → Add
    Baseline" instruction is stale (8D); and a new Nsight Systems
    subsection (8E) covering aggregate multi-launch timing via
    `nsys stats` and using the host-API call counts to rule out
    CPU-GPU scheduling stalls as a cause of any measured slowdown.

**Second correction pass (counter-backed re-measurement on the
reference hardware):**

12. **Peak memory bandwidth fixed.** Every device-banner line printed
    "96.0 GB/s"; the underlying `memoryClockRate * memoryBusWidth / 8`
    formula was missing GDDR's ×2 DDR transfer factor. The real peak
    is ≈192 GB/s, confirmed by Nsight Compute (naive measures 95.7% of
    it at 512³). All expected-output blocks, the roofline `--peak-bw`
    argument, and related prose below were updated accordingly.
13. **Sections 5A/5B/6A/6B/6C — every "illustrative" expected-output
    block replaced with values actually measured** on the reference
    hardware for the exact commands shown (not projections). The old
    numbers (e.g. naive at 82.4 ms/35.4 GB/s for 512³) were off by
    roughly 7–27× depending on grid size and were never measured on
    any GPU; the real naive figure is 11.46 ms at ≈1250 "GB/s" (model
    units — see the `BW Eff.` caveats).
14. **Section 5B/6B/8B — the "mistuned defaults" framing is now
    qualified, not removed.** On this exact hardware, tiled is slower
    than naive at *every* `TILE_Y`/`PENCIL_Z` combination tested,
    including the guide's own Step 3 recommendation (`4/8`, 0.58×) and
    the best found so far (`8/12`, 0.71×). The guidance to try other
    tunings is still correct advice, but readers should not expect a
    configuration that beats naive to exist at this grid size and
    radius on this GPU class — see `docs/CASE_STUDY.md`.
15. **Section 4 — GPU test IDs corrected** from `Start 44`…`Start 58`
    to `Start 60`…`Start 74` (CTest numbers tests sequentially across
    both labels; 59 unit tests precede the 15 GPU tests), and the
    expected wall time corrected from "30–60 sec" to "1–3 sec"
    (measured 2.06 sec).
16. **Step 4 build time corrected** from "3–8 minutes" to "well under
    a minute" (measured 4.3 s wall-clock).
17. **Section 7 — confirmed fixed, not just flagged.** The sign error
    described in item 6 above is fixed in the current repository; the
    guide now says so directly instead of describing it as an open bug
    to patch.
18. **Section 8A — the `perf_event_paranoid` fix was unnecessary** on
    the reference hardware/driver; clarified that `ncu`'s actual gate
    is the driver's `RmProfilingAdminOnly` setting.
19. **Section 8B — sweep grid widened to include `PENCIL_Z=12`**,
    which the original sweep's `{8, 16, 24}` list skipped and which
    turned out to be the best configuration found (≈0.71×, vs. ≈0.69×
    for `TILE_Y=16, PENCIL_Z=24`). Removed an unnecessary `sudo` from
    the sweep loop.
20. **Section 8D — baseline-UI instructions replaced.** Neither the
    original `docs/nsight_workflow.md` instructions nor the previous
    revision of this guide's specific toolbar-tooltip description
    matched Nsight Compute UI 2025.1.1.0 on the reference hardware.
    Replaced with a validated CLI alternative (rule-engine text and
    per-SASS-line stall export) and a summary of what that analysis
    actually showed — which does not match a simple "more registers,
    lower occupancy, more DRAM saved" story.
21. **Section 8E — added confirmed numbers** from running the
    described `nsys stats` workflow (aggregate kernel means, exact
    host-API call counts), rather than only describing what to expect.
22. **Troubleshooting table — several rows added** for the
    peak-bandwidth fix, the `BW Eff.`/roofline model-vs-measured-bytes
    issue, the multi-size "Summary" table's broken Speedup column, and
    root-owned `docs/plots/` from a prior Docker run.
23. **Added pointers throughout to `docs/CASE_STUDY.md`**, a
    counter-backed investigation (Nsight Compute/Systems, SASS
    inspection, a prediction-vs-measurement ledger) that goes well
    beyond what this guide covers, rather than duplicating that detail
    inline.

**Checklist item D2:**

24. **Step 3 vs Section 8B contradiction resolved.** Step 3 no longer
    recommends `TILE_Y=4, PENCIL_Z=8` (which Section 8B and the
    measurements show is the slowest valid configuration, 0.58x); it now
    configures the best measured `TILE_Y=8, PENCIL_Z=12` (about 0.71x).
    The "mistuned A100 defaults" explanation was removed, since tiled is
    slower than naive at every configuration tested. Added the
    `--warmup >= 100` steady-state advice to Section 5A. The 5B/6B/6C
    output blocks are still the 4/8 captures and are labelled as such
    pending a re-run at 8/12. Section 7 already used `--peak-bw 192`.
