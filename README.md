# stencil-solver

A GPU-accelerated 3D finite-difference stencil solver demonstrating
Nsight-driven optimisation for seismic and heat-diffusion workloads.

Implements the acoustic wave equation and heat diffusion over a 3D grid
using 8th-order finite differences, progressing from a naive global-memory
kernel through shared-memory tiling and multi-GPU domain decomposition.

This is the exact computational pattern underlying seismic RTM/FWI,
reservoir simulation, and power-grid transient solvers.

---

## What's in this repo

| Component | Description |
|-----------|-------------|
| `stencil_naive` | Baseline global-memory kernel — Nsight "before" target |
| `stencil_tiled` | Shared-memory tiled kernel — Nsight "after" target |
| `stencil_multigpu` | MPI + NCCL domain-decomposed multi-GPU solver |
| `bench_naive/tiled/scaling` | Benchmark executables with JSON output |
| `scripts/profile_nsight.sh` | Nsight Compute + Systems wrapper with compare mode |
| `scripts/plot_results.py` | Roofline, throughput, speedup, scaling charts |
| `docs/BENCHMARKS.md` | Performance report (fill in after profiling) |
| `docs/nsight_workflow.md` | Step-by-step Nsight profiling guide |

---

## Quick start

### With Docker (recommended)

```bash
# Build the dev image
docker build -f infra/docker/Dockerfile.dev -t stencil-solver:dev .

# Interactive shell with GPU access
docker run --gpus all --rm -it -v $(pwd):/workspace stencil-solver:dev

# Inside the container — build and test
cmake -B build -S . -DSTENCIL_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure --label-regex unit
```

### Local build (requires CUDA ≥ 12.0, CMake ≥ 3.24)

```bash
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTENCIL_BUILD_TESTS=ON \
    -DSTENCIL_BUILD_BENCHMARKS=ON

cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure --label-regex unit
```

### CPU-only mode (no GPU required)

```bash
cmake -B build -S . \
    -DSTENCIL_CPU_FALLBACK=ON \
    -DSTENCIL_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

---

## The optimisation story

The core of this project is a Nsight-driven before/after comparison
of two kernel implementations of the same stencil:

```
∂²p/∂t² = v²(x) · (∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²)
```

Discretised as: `p_next = 2·p_cur - p_prev + dt²·v²·∇²p_cur`

### Naive kernel (`src/kernels/stencil_naive.cu`)

Every thread reads all `6R+1 = 25` stencil neighbours directly from
global memory. Adjacent threads in X load overlapping ranges —
each interior point is loaded ~8 times across a warp in the Y/Z directions.

```
Expected Nsight Compute (512³, fp32, A100):
  DRAM reads:          ~14.7 GB/step
  L1 hit rate:         ~5%
  Memory throughput:   ~45% of peak
  Time per step:       ~3.8 ms
```

### Tiled kernel (`src/kernels/stencil_tiled.cu`)

XY plane loaded once into shared memory per tile. Z direction served
from a per-thread register pencil. Single global memory load per cell.

```
Block:   SMEM_X × SMEM_Y = 40 × 16 = 640 threads
SMEM:    40 × 16 × 4 B = 2,560 B per block
Pencil:  PENCIL_Z + 2R = 24 registers per thread

Expected Nsight Compute (512³, fp32, A100):
  DRAM reads:          ~0.54 GB/step  (27× less)
  L1 hit rate:         ~75%
  Memory throughput:   ~22% of peak
  Time per step:       ~1.1 ms  (~3.5× speedup)
```

### Roofline position

```
Arithmetic intensity = 56 FLOP / 112 bytes = 0.5 FLOP/byte (R=4, fp32)
```

At AI = 0.5 FLOP/byte the kernel is deep in the memory-bound regime.
The tiled kernel improves by reducing effective DRAM traffic, not by
changing the arithmetic.

---

## Profiling

```bash
# Nsight Compute — naive vs tiled comparison (recommended first step)
./scripts/profile_nsight.sh compare ./build/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 --steps 1

# Nsight Compute — single kernel, full metric set
./scripts/profile_nsight.sh compute ./build/bin/stencil_naive \
    --nx 512 --steps 1

# Nsight Systems — end-to-end timeline with NVTX annotations
./scripts/profile_nsight.sh systems ./build/bin/stencil_tiled \
    --nx 512 --steps 50 --warmup 10
```

See [docs/nsight_workflow.md](docs/nsight_workflow.md) for the full
step-by-step guide and interpretation notes.

---

## Benchmarks

```bash
# Single-GPU: naive vs tiled across grid sizes
./build/bin/bench/bench_tiled --steps 200 --warmup 20 \
    --output results/bench_tiled.json

# Generate report and plots
python3 scripts/plot_results.py \
    --naive  results/bench_tiled_naive.json \
    --tiled  results/bench_tiled.json \
    --peak-bw 2039 --peak-flops 312 \
    --outdir docs/plots \
    --report docs/BENCHMARKS.md
```

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the latest results.

---

## Multi-GPU scaling

```bash
# Build with MPI + NCCL
cmake -B build-mpi -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTENCIL_ENABLE_MPI=ON \
    -DSTENCIL_ENABLE_NCCL=ON
cmake --build build-mpi --parallel

# Weak + strong scaling curves (1, 2, 4, 8 GPUs)
./scripts/run_scaling.sh --type both --max-gpus 4 --outdir results

# Merge per-GPU JSONs and plot
python3 scripts/merge_scaling.py --indir results \
    --output results/scaling_combined.json
python3 scripts/plot_results.py \
    --scaling results/scaling_combined.json \
    --outdir docs/plots --report docs/BENCHMARKS.md
```

**Decomposition**: Z-slab domain decomposition. Each rank owns
`NZ/nranks` interior Z-planes plus `STENCIL_RADIUS` halo planes on
each face. Halo exchange uses NCCL (GPU-direct, single-node) or
MPI_Sendrecv (multi-node, host-staged or CUDA-aware).

---

## Build options

| Option | Default | Description |
|--------|---------|-------------|
| `STENCIL_BUILD_TESTS` | `ON` | Unit + integration tests |
| `STENCIL_BUILD_BENCHMARKS` | `ON` | Benchmark executables |
| `STENCIL_ENABLE_MPI` | `OFF` | MPI multi-node path |
| `STENCIL_ENABLE_NCCL` | `OFF` | NCCL GPU-direct halo exchange |
| `STENCIL_ENABLE_NVTX` | `ON` | NVTX range markers for Nsight |
| `STENCIL_CPU_FALLBACK` | `OFF` | OpenMP fallback (no GPU needed) |
| `STENCIL_FP64` | `OFF` | Double precision (default: fp32) |
| `STENCIL_ENABLE_COVERAGE` | `OFF` | gcov/lcov coverage |
| `CMAKE_CUDA_ARCHITECTURES` | `70;75;80;86;90` | Target SM arches |

---

## CI

Every push runs:
1. `clang-format` check
2. CPU-fallback build (no GPU required)
3. Unit tests via CTest (`--label-regex unit`)
4. `clang-tidy` static analysis (on `main`)
5. Coverage report → Codecov (on `main`)

GPU integration tests run on `[self-hosted, gpu, cuda]` runners
for PRs targeting `main`.

Benchmarks run on release tags and post results as job summaries.

---

## Project structure

```
stencil-solver/
├── src/
│   ├── kernels/         naive.cu, tiled.cu, multi_gpu.cu
│   ├── solver/          solver_base, wave_prop, heat_diffusion
│   ├── comm/            nccl_exchange, mpi_halo
│   ├── utils/           grid, timer, io
│   └── main_{naive,tiled,multigpu}.cpp
├── include/stencil/     types, config, grid, timer, kernels, solver, comm, io
├── tests/
│   ├── unit/            test_fd_coeffs, test_grid, test_timer, test_solver*
│   ├── integration/     test_wave_3d, test_heat_3d, test_kernel_correctness
│   └── perf/            bench_naive, bench_tiled, bench_scaling
├── docs/                BENCHMARKS.md, nsight_workflow.md
├── scripts/             profile_nsight.sh, run_scaling.sh, plot_results.py,
│                        merge_scaling.py, summarize_results.py, format.sh
├── infra/docker/        Dockerfile.dev, Dockerfile.ci, docker-compose.yml
├── cmake/               FindNCCL.cmake, CUDAArchUtils.cmake
└── .github/workflows/   ci.yml, benchmark.yml, release.yml
```
