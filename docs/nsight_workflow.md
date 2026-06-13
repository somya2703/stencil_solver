# Nsight-Driven Optimisation Workflow

This document walks through the exact profiling workflow used to identify
the memory bottleneck in the naive kernel and verify the tiled kernel's
improvement. It mirrors what an NVIDIA DevTech engineer would do with a
customer's stencil code.

---

## Prerequisites

```bash
# Verify tools are available
ncu --version      # Nsight Compute CLI
nsys --version     # Nsight Systems CLI

# Build with line info (already set in CMake Release config)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DSTENCIL_ENABLE_NVTX=ON
cmake --build build --parallel
```

---

## Step 1 — Nsight Systems: find the hot spot

Before diving into kernel metrics, use Nsight Systems to get the
end-to-end picture: Is the solver compute-bound or communication-bound?
Are there unexpected CPU-GPU sync stalls?

```bash
./scripts/profile_nsight.sh systems ./build/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 --steps 50 --warmup 10
# → nsight-output/<timestamp>/stencil_tiled.nsys-rep
```

Open in `nsys-ui` and look for:

| What to check | Green (good) | Red (bad) |
|---|---|---|
| CUDA kernel occupancy in timeline | Dense, back-to-back kernels | Gaps between kernels |
| NVTX `launch_naive`/`launch_tiled` | Visible, annotated | Missing (NVTX not linked) |
| `cudaStreamSynchronize` stalls | None inside step loop | Any inside step loop |
| SM utilisation | > 80% | < 50% |
| Memory copy operations | Only at init/teardown | Inside step loop |

Expected result: kernels are back-to-back, no sync stalls. The NVTX
`step` region encapsulates the kernel launch cleanly.

---

## Step 2 — Nsight Compute: profile the naive kernel

```bash
./scripts/profile_nsight.sh compute ./build/bin/stencil_naive \
    --nx 512 --ny 512 --nz 512 --steps 1
# → nsight-output/<timestamp>/stencil_naive.ncu-rep
```

### Reading the Speed-of-Light section

Open the `.ncu-rep` in `ncu-ui`. The **Speed-of-Light** section shows
what percentage of theoretical peak is being achieved:

```
Memory Throughput:  ~45%  ← memory-bound (not compute-bound)
Compute Throughput: ~12%  ← confirms memory bottleneck
```

**If memory throughput < compute throughput → kernel is memory-bound.**
This is expected for any stencil with AI < ridge point.

### Key metrics to record (for the BENCHMARKS.md table)

```
# Run ncu in CLI mode to extract metrics as CSV
ncu --metrics \
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__hit_rate,\
    sm__warps_active.avg.pct_of_peak_sustained_active \
    --csv \
    ./build/bin/stencil_naive --nx 512 --steps 1
```

**What the naive kernel shows:**

- `dram__bytes_read.sum` ≈ `(6R+4) × sizeof(float) × N_interior`
  = 28 × 4 × 509³ ≈ **14.7 GB** per step for 512³ (R=4)
  → confirms every neighbour is fetched from DRAM

- `l1tex__hit_rate` ≈ 3–8%
  → X-direction loads hit L1 (contiguous), Y/Z do not (strided)

- Memory throughput ≈ 40–55% of peak
  → strided Y/Z loads are only partially coalesced (warp touches multiple
     cache lines per instruction for stride > 128 bytes)

---

## Step 3 — Profile the tiled kernel

```bash
./scripts/profile_nsight.sh compute ./build/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 --steps 1
# → nsight-output/<timestamp>/stencil_tiled.ncu-rep
```

**What the tiled kernel shows:**

- `dram__bytes_read.sum` ≈ `sizeof(float) × N_total`
  ≈ 4 × 512³ ≈ **0.54 GB** per step
  → ~27× fewer DRAM reads (6R+4 = 28 → 1 effective load/cell)

- `l1tex__hit_rate` ≈ 70–80%
  → XY plane served from SMEM; Z pencil from registers

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` ≈ 0
  → SMEM_X = 40 breaks the 32-bank conflict pattern

- Memory throughput % of peak: drops to ~20–30%
  → bottleneck shifts from DRAM bandwidth to SMEM latency/occupancy

### Side-by-side comparison (ncu-ui diff view)

1. Open `ncu-ui`
2. File → Open → select `stencil_naive.ncu-rep`
3. File → Add Baseline → select `stencil_tiled.ncu-rep`
4. The diff view highlights every metric where tiled improves on naive

---

## Step 4 — Use the compare script

The `profile_nsight.sh compare` mode runs both kernels and prints
a summary diff automatically:

```bash
./scripts/profile_nsight.sh compare ./build/bin/stencil_tiled \
    --nx 512 --steps 1
```

This profiles `stencil_naive` and `stencil_tiled` back-to-back and
writes both `.ncu-rep` files to the same timestamped directory.

---

## Step 5 — Occupancy analysis

For the tiled kernel, run the occupancy calculator:

```bash
ncu --set roofline ./build/bin/stencil_tiled --nx 512 --steps 1
```

Expected for Ampere (A100), fp32, R=4:

| Parameter | Value |
|---|---|
| Block size | 40 × 16 = 640 threads |
| Registers/thread | ~40 |
| SMEM/block | 2,560 B |
| Theoretical occupancy | ~50% |
| Achieved occupancy | ~45–55% |

The 640-thread block is larger than the 512-thread sweet spot for many
Ampere SMs, but the low SMEM footprint (2.5 KB) means many blocks can
run concurrently. Tuning `TILE_Y` from 8 to 4 reduces block size to
320 and can improve occupancy at the cost of fewer Y reuses.

---

## Step 6 — Nsight Systems: multi-GPU halo exchange

```bash
cmake -B build-mpi -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTENCIL_ENABLE_MPI=ON \
    -DSTENCIL_ENABLE_NCCL=ON \
    -DSTENCIL_ENABLE_NVTX=ON
cmake --build build-mpi --parallel

nsys profile \
    --trace=cuda,nvtx,mpi,ucx \
    --output nsight-output/multigpu \
    mpirun -n 4 ./build-mpi/bin/stencil_multigpu \
        --nx 256 --ny 256 --nz 1024 --steps 50
```

Open `nsight-output/multigpu.nsys-rep` in `nsys-ui`.

**What to look for:**

- NVTX `step` regions on all 4 ranks should be aligned (no load imbalance)
- NVTX `halo_exchange` region: NCCL send/recv should overlap with the
  *next* step's kernel launch (if you implement the overlap optimisation)
- MPI timeline (if enabled): `MPI_Sendrecv` calls only appear in the
  staged-MPI path, not the NCCL path

---

## Interpreting the roofline chart

The roofline model from `plot_results.py --peak-bw N --peak-flops N`
shows:

```
Attained GFLOP/s (Y)
        │   compute roof (horizontal line at peak TFLOP/s)
        │  /
        │ /
        │/ ← ridge point
        │  memory roof (diagonal: BW × AI)
        └─────────────── Arithmetic Intensity (X, log)
              ↑
         AI ≈ 0.5 FLOP/byte for this stencil
```

Both naive and tiled points appear at the **same X position** (same
arithmetic intensity — the kernel does the same math either way).
The tiled point is **higher on Y** because it achieves better bandwidth
utilisation, not because it changes the algorithm.

This is the correct interpretation to give in a performance report or
interview: *"The optimisation doesn't change the arithmetic intensity,
it improves how efficiently we use the available memory bandwidth."*

---

## Quick-reference commands

```bash
# Profile naive kernel, capture full metric set
./scripts/profile_nsight.sh compute ./build/bin/stencil_naive \
    --nx 512 --steps 1

# Profile tiled kernel
./scripts/profile_nsight.sh compute ./build/bin/stencil_tiled \
    --nx 512 --steps 1

# Both in one shot (for BENCHMARKS.md table)
./scripts/profile_nsight.sh compare ./build/bin/stencil_tiled \
    --nx 512 --steps 1

# End-to-end timeline
./scripts/profile_nsight.sh systems ./build/bin/stencil_tiled \
    --nx 512 --steps 50 --warmup 10

# Multi-GPU timeline (4 ranks)
nsys profile --trace=cuda,nvtx,mpi \
    mpirun -n 4 ./build-mpi/bin/stencil_multigpu \
        --nx 256 --ny 256 --nz 1024 --steps 20
```
