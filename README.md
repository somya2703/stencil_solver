# stencil-solver

A GPU-accelerated 3D finite-difference stencil solver demonstrating
Nsight-driven optimisation for seismic and heat-diffusion workloads.

Implements the acoustic wave equation and heat diffusion over a 3D grid
using 8th-order finite differences, progressing from a naive global-memory
kernel through shared-memory tiling. A multi-GPU domain-decomposed path
is implemented but has not been run or validated (see `docs/CASE_STUDY.md` section 7).

This is the exact computational pattern underlying seismic RTM/FWI,
reservoir simulation, and power-grid transient solvers.

## Key findings (RTX 4050 Laptop GPU, Ada SM 8.9, CUDA 12.8)

The expected result for this project was "shared-memory tiling beats the naive kernel."
On this GPU it does not, and the investigation explains why.

| | Naive | Tiled (best config, TILE_Y=8, PENCIL_Z=12) |
|---|---|---|
| Time per step, 512^3 | about 11.5 ms | about 16.0 ms (0.72x of naive) |
| Effective bandwidth (compulsory traffic / time) | about 178 GB/s | about 128 GB/s |
| DRAM read per step | 1.59 GB | 1.98 GB |

- **Naive is already at the DRAM limit.** Nsight Compute reports about 96% of peak DRAM
  throughput, and its time is within roughly 2-5% of the traffic floor. Reducing
  redundant reads cannot help much when L2 already absorbs them.
- **The peak is independently checked.** A standalone read-only kernel reaches about
  188 GB/s (98% of the derived 192 GB/s peak).
- **Tiling moves more DRAM bytes, not fewer**, and spends 33-51% of warp cycles at
  barriers (naive: 0%). Occupancy explains only part of the slowdown.
- **Two kernel edits were tried and did not help:** replacing a redundant `p_cur`
  reload was neutral, and prefetching `p_prev`/`vel2` before the barrier was about
  5% slower. Two performance steps in the PENCIL_Z sweep remain unexplained.
- **Limits:** one GPU only (the tiled design targeted A100-class parts and has not
  been run on one); the multi-GPU path was never run.

Details, evidence tags (verified / derived / inferred / unverified) and the full
prediction ledger, including the predictions that failed, are in
[docs/CASE_STUDY.md](docs/CASE_STUDY.md).

---

## What's in this repo

| Component | Description |
|-----------|-------------|
| `stencil_naive` | Baseline global-memory kernel — Nsight "before" target |
| `stencil_tiled` | Shared-memory tiled kernel — Nsight "after" target |
| `stencil_multigpu` | MPI + NCCL domain-decomposed solver (implemented, **not validated**) |
| `bench_naive/tiled/scaling` | Benchmark executables with JSON output |
| `scripts/profile_nsight.sh` | Nsight Compute + Systems wrapper with compare mode |
| `scripts/plot_results.py` | Roofline, throughput, speedup, scaling charts |
| `docs/BENCHMARKS.md` | Naive-vs-tiled benchmark table and plots (RTX 4050 Laptop) |
| `docs/CASE_STUDY.md` | Counter-backed investigation of why tiling loses on this GPU |
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
Measured Nsight Compute (512³, fp32, RTX 4050 Laptop GPU -- this repo's
actual reference hardware; see docs/CASE_STUDY.md section 3 for the full
counter-backed methodology):
  DRAM reads:          ~1.59 GB/step
  DRAM throughput:     ~95.7% of peak (~192 GB/s, ncu --clock-control none)
  Time per step:       ~11.46 ms
```

The ~14.7 GB/step figure quoted in earlier drafts of this README was a
theoretical worst-case estimate (zero L2 reuse assumed) and was never
measured on any GPU. On this card the L2 cache absorbs most of the
redundant neighbour reads, so actual DRAM traffic is far lower -- naive
is already within ~4.5% of this GPU's practical DRAM-bandwidth floor.
DRAM byte counts like this depend on the GPU's cache behaviour and
won't necessarily transfer to other hardware (e.g. an A100's larger L2).

### Tiled kernel (`src/kernels/stencil_tiled.cu`)

XY plane loaded once into shared memory per tile. Z direction served
from a per-thread register pencil. Single global memory load per cell.

```
Block:   SMEM_X × SMEM_Y = 40 × 16 = 640 threads
SMEM:    40 × 16 × 4 B = 2,560 B per block
Pencil:  PENCIL_Z + 2R = 20 live values in the sliding window at this
         repo's current default (PENCIL_Z=12); the compiler's actual
         total register count per thread is 48 (the pencil array isn't
         the only thing using registers) -- see docs/CASE_STUDY.md 3.2

Measured Nsight Compute (512³, fp32, RTX 4050 Laptop GPU, TILE_Y=8,
PENCIL_Z=12 -- this repo's current default; see docs/CASE_STUDY.md 3.2):
  DRAM reads:          ~1.98 GB/step  (MORE than naive's ~1.59 GB, not
                       less -- see the note below)
  L1 hit rate:         ~28.8%
  DRAM throughput:     ~89.0% of peak
  Time per step:       ~16.1 ms  (0.71× of naive -- tiled is SLOWER on
                       this GPU, confirmed across every TILE_Y/PENCIL_Z
                       configuration tested; the best found is this one)
```

### Roofline position

```
Arithmetic intensity = 56 FLOP / 16 bytes = 3.5 FLOP/byte (R=4, fp32, compulsory traffic)
```

(Bytes are the compulsory traffic per point, 4 x 4 B; the earlier
no-reuse figure, 112 B, is kept as `model_bandwidth_gbs`. See
docs/BUILD_AND_RUN.md section 6C.)

At AI = 3.5 FLOP/byte the kernel is deep in the memory-bound regime on
any GPU. On datacenter-class hardware (A100, this kernel's original
design target) the tiled kernel is expected to win by reducing
effective DRAM traffic. **On this repo's reference hardware (RTX 4050
Laptop, Ada Lovelace) that is not what happens**: tiled reads *more*
DRAM than naive at every configuration tested, and naive is already
near this GPU's practical DRAM-bandwidth ceiling. The measured
slowdown traces mainly to barrier stalls from halo-only warps and
extra warp instructions, not to bandwidth utilisation. See
`docs/CASE_STUDY.md` for the full counter-backed investigation,
including which candidate explanations hold up and which don't.

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
    --peak-bw 192 --peak-flops 12 \
    # (this card's real peak, not the A100 defaults shown in older
    # revisions of this command -- see docs/CASE_STUDY.md section 2)
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
| `CMAKE_CUDA_ARCHITECTURES` | `70;75;80;86;90` | Target SM arches. For a single local GPU, `-DCMAKE_CUDA_ARCHITECTURES=native` (CMake ≥ 3.24) builds only for it and is much faster |

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
