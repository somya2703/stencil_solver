# Build and Run Guide

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
> | **Peak memory bandwidth** | 96.0 GB/s |
> | **Peak FP32 (approx.)** | ~12 TFLOP/s *(estimate — laptop power limits vary by vendor/chassis; check `nvidia-smi -q -d CLOCK` or your OEM spec sheet for the exact sustained figure)* |
>
> The original guide's example output was written against an A100
> (2,039 GB/s, 108 SMs, 80 GB). Every number in that version is roughly
> **20× optimistic** for this card on bandwidth alone, and grid sizes
> that comfortably fit on an A100 can exceed 6.1 GB here. All commands
> and expected-output blocks below have been adjusted accordingly.

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

If you're missing anything, use the Docker path in Section 1B — it
handles all dependencies automatically.

---

## Section 1A — Local build (you have CUDA + CMake installed)

### Step 1 — Clone and enter the repo

```bash
git clone https://github.com/YOUR_USERNAME/stencil-solver.git
cd stencil-solver
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

Compiling for every architecture in the default list
(`70;75;80;86;90`) wastes time when you only have one card. Target
Ada Lovelace (SM 8.9) directly:

```bash
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTENCIL_BUILD_TESTS=ON \
    -DSTENCIL_BUILD_BENCHMARKS=ON \
    -DSTENCIL_ENABLE_NVTX=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89"
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

### Step 4 — Build

```bash
cmake --build build --parallel $(nproc)
```

This is a laptop CPU as well as a laptop GPU — expect the first build
to take longer than a workstation (roughly 3–8 minutes depending on
core count), mostly in `nvcc` compiling the tiled kernel's unrolled
loops.

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

The shell scripts in this repo are not marked executable by default
in a fresh checkout. Set that once, up front, or every `./scripts/*.sh`
invocation below will fail with `Permission denied`:

```bash
chmod +x scripts/*.sh
```

(Alternatively, prefix any script call with `bash`, e.g.
`bash scripts/profile_nsight.sh ...`, without changing permissions.)

---

## Section 1B — Docker build (no local CUDA toolchain needed)

```bash
docker build -f infra/docker/Dockerfile.dev -t stencil-solver:dev .

docker run --gpus all --rm -it \
    -v $(pwd):/workspace \
    stencil-solver:dev
```

Inside the container, run Steps 3–5 above identically — use
`-DCMAKE_CUDA_ARCHITECTURES="89"` there too, and `chmod +x scripts/*.sh`
still applies (permission bits aren't preserved by every `docker build`
context depending on your `.dockerignore`/git settings).

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

Useful on this machine specifically if you want to run the full test
suite while the GPU is busy with something else (e.g. a display
compositor eating VRAM) — CPU fallback doesn't touch the 6.1 GB budget
at all.

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
100% tests passed, 0 tests failed out of 43
Total Test time (real) = 3–6 sec
```

Unit tests use grid sizes ≤ 32³ — negligible on any GPU, laptop or
otherwise, so timing here won't meaningfully differ from a workstation.

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
    Start 44: test_wave_3d.WaveSolver3DTest.EnergyDoesNotGrow
    ...
    Start 58: test_kernel_correctness.KernelCorrectness.LinearFieldZeroLaplacian

100% tests passed, 0 tests failed out of 15
Total Test time (real) = 30–60 sec
```

These use grids ≤ 64³, so memory and bandwidth headroom are not a
concern on this card.

---

## Section 5 — Run the solvers directly

### 5A — Naive solver (baseline)

```bash
./build/bin/stencil_naive \
    --nx 256 --ny 256 --nz 256 \
    --steps 100 --warmup 10 \
    --physics wave \
    --verbose
```

**Expected output (illustrative for this GPU class — your exact
numbers will vary):**
```
=== stencil-solver: naive kernel ===

GPU [0]: NVIDIA GeForce RTX 4050 Laptop GPU
  Compute: 8.9   SMs: 20   Memory: 6.1 GB
  Peak bandwidth: 96.0 GB/s

Grid: 256³   steps: 100   warmup: 10

── Grid: wave/naive ─────────────────────
  Total cells   : 256 × 256 × 256  =  16.78 M
  Interior      : 248 × 248 × 248  =  15.25 M
  Spacing       : dx=10.00 m  dy=10.00 m  dz=10.00 m
  Time step     : dt=3.0000e-03 s
  Stencil radius: R=4  (8th order)
  Memory/field  : 67.11 MB
  Fields (3×p + vel2): 268.44 MB

  wave/naive              22.4 ms/step    76.3 GB/s     36.7 GFLOP/s
```

> With 20 SMs vs. an A100's 108 (~5.4×) and 96 GB/s vs. 2,039 GB/s
> (~21×), a naive kernel result in the tens-of-ms range for 256³ is
> expected, not a regression. What matters is the **relative**
> naive→tiled speedup, not the absolute ms/step compared to datacenter
> hardware.

### 5B — Tiled solver (naive vs tiled comparison)

At 512³, four fp32 fields cost roughly:
`512³ × 4 bytes × 4 fields ≈ 2.1 GB` — comfortably inside 6.1 GB, but
leaves less headroom than an 80 GB card if anything else is using
VRAM (browser, display compositor, etc.). Close unnecessary GPU
consumers before large runs.

```bash
./build/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 \
    --steps 200 --warmup 20
```

**Expected output (illustrative):**
```
=== stencil-solver: naive vs tiled comparison ===

GPU [0]: NVIDIA GeForce RTX 4050 Laptop GPU  (6.1 GB  |  peak BW: 96.0 GB/s)

Stencil radius : 4  (8th-order)
Tile dims      : 32 × 8  (SMEM block: 40 × 16)
Z pencil depth : 16

Grid     Naive ms    BW (GB/s)   Tiled ms   BW (GB/s)    Speedup
-------- ----------  ----------  ----------  ----------  --------
512³      82.4 ms     71.6         26.1 ms     55.9        3.16×

── Naive ────────────────────────────────────────────
  wave/naive               82.4 ms     71.6      34.5    1.00×

── Tiled ────────────────────────────────────────────
  wave/tiled                26.1 ms     55.9      26.7    1.00×
```

> The **speedup ratio** (~3×) should land in a similar ballpark to the
> A100 story even though absolute ms/step is much higher here — the
> tiled kernel's DRAM-traffic reduction is architecture-independent;
> only the *scale* of the numbers changes with your bandwidth/SM count.

> **Memory ceiling on this card:** don't jump straight to 1024³ —
> `1024³ × 4 bytes × 4 fields ≈ 17.2 GB`, which will not fit in 6.1 GB
> and will fail with a CUDA out-of-memory error. Stay at or below
> roughly 640³ for single-run headroom on this GPU
> (`640³ × 4 × 4 ≈ 4.2 GB`).

---

## Section 6 — Run the benchmarks

### 6A — Naive benchmark (grid size sweep)

The default sweep (`64, 128, 192, 256, 320, 384, 512`) is safe on
6.1 GB — the largest point (512³) still fits as shown above.

```bash
mkdir -p results
./build/bin/bench/bench_naive \
    --steps 200 --warmup 20 \
    --output results/bench_naive.json
```

**Expected output (illustrative):**
```
=== bench_naive — global-memory stencil ===

Device : NVIDIA GeForce RTX 4050 Laptop GPU
Memory : 6.1 GB   Peak BW: 96.0 GB/s

Radius : 4  (8th order)
Steps  : 200  Warmup: 20

Grid      ms/step       GB/s       GFLOP/s
--------  ----------  ----------  ------------
64³         0.32 ms      18.9        9.1
128³        1.9 ms       28.7       13.8
192³        5.7 ms       33.9       16.3
256³       13.4 ms       35.0       16.8
320³       25.9 ms       35.2       16.9
384³       44.1 ms       35.9       17.3
512³       82.4 ms       35.4       17.0

Results written to: results/bench_naive.json
```

### 6B — Tiled benchmark (naive vs tiled comparison)

```bash
./build/bin/bench/bench_tiled \
    --steps 200 --warmup 20 \
    --output results/bench_tiled.json
```

**Expected output (illustrative):**
```
=== bench_tiled — naive vs shared-memory tiled ===

Device : NVIDIA GeForce RTX 4050 Laptop GPU
Memory : 6.1 GB   Peak BW: 96.0 GB/s

Radius : 4  (8th order)
Steps  : 200  Warmup: 20

Grid      Naive ms    Naive GB/s  Tiled ms    Tiled GB/s  Speedup
--------  ----------  ----------  ----------  ----------  --------
64³        0.32 ms      18.9       0.13 ms      14.2       2.46×
128³       1.9 ms       28.7       0.61 ms      21.3       3.11×
192³       5.7 ms       33.9       1.75 ms      26.1       3.26×
256³      13.4 ms       35.0       4.11 ms      27.8       3.26×
320³      25.9 ms       35.2       7.90 ms      28.6       3.28×
384³      44.1 ms       35.9      13.4 ms       29.1       3.29×
512³      82.4 ms       35.4      26.1 ms       28.9       3.16×

Results written to: results/bench_tiled.json
Results written to: results/bench_tiled_naive.json
```

### 6C — Scaling benchmark (single-GPU roofline)

```bash
./build/bin/bench/bench_scaling \
    --steps 200 --warmup 20 \
    --output results/bench_scaling.json
```

**Expected output (illustrative):**
```
=== bench_scaling — single-GPU roofline ===

Peak BW   : 96.0 GB/s
Kernel    : tiled
Radius    : 4  (8th-order)

Grid      ms/step       GB/s       BW Eff.
--------  ----------  ----------  ----------
64³         0.13 ms      14.2       14.8%
96³         0.42 ms      19.6       20.4%
128³        0.61 ms      21.3       22.2%
192³        1.75 ms      26.1       27.2%
256³        4.11 ms      27.8       29.0%
384³       13.4 ms       29.1       30.3%
512³       26.1 ms       28.9       30.1%

Scaling results written to: results/bench_scaling.json
```

> BW efficiency is now relative to this card's **96 GB/s** peak, not
> an A100's 2,039 GB/s — a ~30% efficiency figure here is a
> *healthier* result relative to peak than the ~17% the same kernel
> reports on an A100, because a laptop chip's lower SM count makes it
> easier to approach its (much lower) memory ceiling.

---

## Section 7 — Generate plots and report

`scripts/plot_results.py` now supports `--peak-bw` and `--peak-flops`
(roofline plot) and always generates a naive-vs-tiled speedup chart
alongside the throughput bar chart, in addition to the Markdown report.

Use this card's real peak bandwidth and your best FP32 estimate —
**do not** reuse the A100 defaults (`2039`/`312`) from the original
guide, or the roofline ceiling will be drawn for hardware you don't
have and every data point will appear artificially far below the roof:

```bash
python3 scripts/plot_results.py \
    --naive   results/bench_tiled_naive.json \
    --tiled   results/bench_tiled.json \
    --peak-bw 96 \
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

**Generated files:**
```
docs/
├── plots/
│   ├── throughput_comparison.png  ← grouped bar chart: naive vs tiled BW + GFLOP/s
│   ├── speedup.png                ← speedup vs grid size line chart
│   └── roofline.png               ← roofline with naive/tiled operating points
└── BENCHMARKS.md                  ← report with tables filled in
```

> If you don't know your card's exact sustained FP32 TFLOP/s, `12` is
> a reasonable estimate for this chip based on core count × boost
> clock, but laptop power/thermal limits (Optimus, dynamic boost, dock
> vs. battery) can shift the real number meaningfully. Treat the
> roofline compute ceiling as approximate unless you've measured it
> directly (e.g. with a dedicated FP32 GEMM microbenchmark).

---

## Section 8 — Nsight profiling

### 8A — One-time setup: script permissions and possible perf permissions

If you haven't already done Step 5 in Section 1A:
```bash
chmod +x scripts/*.sh
```

If `ncu` refuses to attach with a permissions error:
```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
```

### 8B — Profile both kernels and see the diff table

```bash
./scripts/profile_nsight.sh compare ./build/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 --steps 1
```

**Expected output (illustrative):**
```
Profiling naive:  ./build/bin/stencil_naive
Profiling tiled:  ./build/bin/stencil_tiled
Output dir:       nsight-output/20260714_120000

── Nsight Compute: stencil_naive --nx 512 --ny 512 --nz 512 --steps 1 ──
   Output: nsight-output/20260714_120000/stencil_naive.ncu-rep
==PROF== Connected to process ...
==PROF== Profiling "kernel_naive" - 0: 0%....50%....100% - 39 passes
...
  wave/naive              2800-3000 ms/step      ~5 GB/s      ~2.5 GFLOP/s
==PROF== Report: nsight-output/20260714_120000/stencil_naive.ncu-rep

── Nsight Compute: stencil_tiled --nx 512 --ny 512 --nz 512 --steps 1 ──
   ...

── Metric comparison (naive vs tiled) ────────────────────────────
  Metric                           Naive        Tiled
  ------------------------------  ------------  ------------
  DRAM reads (GB)                    14.72         0.54
  DRAM writes (GB)                    0.54         0.54
  L1 hit rate                         4-6%        70-80%
  Mem throughput % peak              70-90%        30-45%
  Achieved occupancy                 40-55%        45-60%

── Next steps ──────────────────────────────────────────────────
  ncu-ui nsight-output/20260714_120000/stencil_naive.ncu-rep
  ncu-ui nsight-output/20260714_120000/stencil_tiled.ncu-rep
  (File → Add Baseline for side-by-side diff view)
```

> **⚠️ The `ms/step` numbers under `--set full` are not real
> performance numbers.** Nsight Compute replays each kernel dozens of
> times (you'll see `"kernel_naive" - 0` through roughly `- 10`, each
> taking ~38–39 passes) to collect the full metric set. That inflates
> wall time by two to three orders of magnitude and can even make the
> tiled kernel look *slower* than naive in the printed table — that's
> a profiling artifact, not a regression. **Use Section 6's benchmark
> binaries (`bench_naive`/`bench_tiled`, no `ncu` involved) for actual
> timing numbers.** Use `ncu`'s output only for the DRAM-traffic,
> L1-hit-rate, and occupancy metrics in the diff table above.

> **DRAM-GB figures above are still meaningful even on this card** —
> `dram__bytes_read.sum` reflects actual bytes moved per kernel launch,
> which doesn't scale with SM count or peak bandwidth. Expect roughly
> the same ~27× DRAM-traffic reduction (naive → tiled) here as on an
> A100, because that reduction comes from the shared-memory reuse
> pattern in the kernel itself, not from GPU class.

### 8C — Nsight Systems timeline

```bash
./scripts/profile_nsight.sh systems ./build/bin/stencil_tiled \
    --nx 512 --steps 50 --warmup 10
```

This does **not** use kernel replay, so timings here are close to
real (`nsys` samples the actual execution rather than re-running the
kernel dozens of times). Expect the tiled `step` NVTX regions to be
back-to-back with no gaps, same as the A100 story.

---

## Section 9 — Multi-GPU build and scaling

This is a single-GPU laptop, so `stencil_multigpu` will only ever run
with `-n 1` here — there's no second device to exercise the halo
exchange path meaningfully. If you want to exercise the MPI/NCCL code
paths at all, build with MPI enabled and run a single rank as a smoke
test rather than a scaling experiment:

```bash
cmake -B build-mpi -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTENCIL_ENABLE_MPI=ON \
    -DSTENCIL_ENABLE_NVTX=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89"

cmake --build build-mpi --parallel $(nproc)

mpirun -n 1 ./build-mpi/bin/stencil_multigpu \
    --nx 256 --ny 256 --nz 256 --steps 50 --warmup 10
```

Real weak/strong scaling curves (Section 9 in the original guide)
require multiple GPUs and aren't reproducible on this machine — skip
`run_scaling.sh` and `merge_scaling.py` unless you have access to a
multi-GPU host or cloud instance.

---

## Section 10 — CI locally (reproduce what GitHub Actions runs)

```bash
docker build -f infra/docker/Dockerfile.ci -t stencil-solver:ci .
docker run --rm stencil-solver:ci
```

CI uses `STENCIL_CPU_FALLBACK=ON`, so this step is identical
regardless of what GPU you have — no laptop-specific adjustment
needed here.

---

## Troubleshooting (updated with issues found on this setup)

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `./scripts/*.sh: Permission denied` | Scripts aren't marked executable in a fresh checkout | `chmod +x scripts/*.sh`, or run via `bash scripts/foo.sh ...` |
| `ncu: unrecognised option '--output'` | `scripts/profile_nsight.sh` used `--output`, which isn't valid `ncu` CLI syntax (unlike `nsys`) | Change to `-o` in `run_compute()`'s `ncu` invocation |
| `ncu: Failed to find metric regex:^l1tex__hit_rate\.(sum\|min\|max\|avg\|pct\|ratio\|max_rate)$` | Bare `l1tex__hit_rate` isn't a resolvable metric on current `ncu` — it needs a rollup suffix | Use `l1tex__hit_rate.pct` everywhere the script references it (the `METRICS` array, the diff-table `--metrics` string, and the printed-row key) |
| `plot_results.py: error: unrecognized arguments: --peak-bw ... --peak-flops ...` | Older version of the script didn't define these flags or the roofline function | Add `--peak-bw`/`--peak-flops` args and a `plot_roofline()` call in `main()` |
| `ModuleNotFoundError: No module named 'networkx'` | A stray `from networkx import radius` line, unrelated to this codebase, ended up in `plot_results.py` (likely an editor auto-import on the unresolved local variable `radius`) | Delete that import line entirely — `radius` here is just `stencil_radius` from the benchmark JSON |
| Diff table prints `'NoneType' object has no attribute 'strip'` and falls back to "could not read .ncu-rep files" | `.ncu-rep` contains multiple profiled launches (warmup + timed); `csv.DictReader` returns ragged rows with `None` for missing columns, and the script blindly calls `.strip()` on every value | Guard the dict comprehension against `None` keys/values, and use the **last** CSV row (the timed launch), not `rows[0]` (a warmup pass) |
| `ncu --set full` reports thousands-of-ms `ms/step` that seem absurdly slow | Kernel replay for full metric collection (~39 passes per launch) inflates wall time by 2–3 orders of magnitude — expected `ncu` behavior, not a real regression | Get real timing from `bench_naive`/`bench_tiled`/`stencil_tiled` directly (no `ncu`); use `ncu` output only for DRAM/occupancy/hit-rate metrics |
| `CUDA_ARCH not supported` | GPU older than SM 70, or you built for archs that don't include yours | `-DCMAKE_CUDA_ARCHITECTURES="89"` for this card specifically |
| `cudaErrorMemoryAllocation` / out-of-memory on large grids | 6.1 GB VRAM ceiling — `N³ × 4 bytes × 4 fields` grows fast (1024³ ≈ 17 GB) | Stay at or below ~640³ for headroom; close other GPU consumers (browser, compositor) before large runs |
| `nvcc: error: unrecognized option '--expt-relaxed-constexpr'` | nvcc < 11 | Upgrade CUDA toolkit to ≥ 12.0 |
| `MPI_Init failed` | OpenMPI not installed | `apt install libopenmpi-dev openmpi-bin` |
| `ncu: permission denied` | Nsight Compute needs elevated perf counter access | `sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'` |
| `nsys: error opening shared library` | Missing nsys library path | `export LD_LIBRARY_PATH=/opt/nvidia/nsight-systems/.../lib:$LD_LIBRARY_PATH` |

---

## Summary: what each binary does

| Binary | Input | Output |
|--------|-------|--------|
| `stencil_naive` | Grid size, steps, physics flags | Console: ms/step, GB/s, GFLOP/s. Optional: JSON |
| `stencil_tiled` | Same | Console: naive vs tiled table with speedup. Two JSON files |
| `stencil_multigpu` | Same + MPI rank count | Single-rank smoke test only on this hardware (no second GPU) |
| `bench_naive` | Grid sweep or single size | Console: 7-row sweep table (safe up to 512³ on 6.1 GB). JSON for plot_results.py |
| `bench_tiled` | Same | Console: naive+tiled side-by-side table. Two JSON files |
| `bench_scaling` | Grid sweep | Console: BW efficiency % table (relative to 96 GB/s peak here, not 2,039 GB/s) |
| `test_*` | (none) | GoogleTest pass/fail output — grid sizes small enough to be VRAM-agnostic |

---

## Quick reference: commands adjusted for this GPU

```bash
# One-time setup
chmod +x scripts/*.sh

# Configure for this specific GPU
cmake -B build -S . -DSTENCIL_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build --parallel $(nproc)

# Benchmarks (safe grid range: up to 512³)
./build/bin/bench/bench_tiled --steps 200 --warmup 20 --output results/bench_tiled.json

# Plots + report, using this card's real peak BW/FLOPs (not A100 defaults)
python3 scripts/plot_results.py \
    --naive results/bench_tiled_naive.json \
    --tiled results/bench_tiled.json \
    --peak-bw 96 --peak-flops 12 \
    --outdir docs/plots --report docs/BENCHMARKS.md

# Nsight Compute — metrics only, ignore the inflated ms/step in this mode
./scripts/profile_nsight.sh compare ./build/bin/stencil_tiled --nx 512 --steps 1

# Nsight Systems — real timing, no replay overhead
./scripts/profile_nsight.sh systems ./build/bin/stencil_tiled --nx 512 --steps 50 --warmup 10
```
