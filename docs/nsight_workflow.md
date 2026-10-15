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
Memory Throughput:  95.56%  ← nearly saturating DRAM bandwidth
Compute Throughput: 41.95%  ← confirms memory bottleneck
```

(Measured on this repo's reference hardware -- RTX 4050 Laptop GPU; see
docs/CASE_STUDY.md section 3.3. This kernel sits much closer to the
DRAM ceiling than a generic "~45%" example would suggest -- naive is
within ~4.5% of the practical bandwidth floor on this card.)

**If memory throughput < compute throughput → kernel is memory-bound.**
This is expected for any stencil with AI < ridge point.

### Key metrics to record (for the BENCHMARKS.md table)

```
# Run ncu in CLI mode to extract metrics as CSV
ncu --clock-control none -k kernel_naive --launch-count 1 --metrics \
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sector_hit_rate.pct,\
    sm__warps_active.avg.pct_of_peak_sustained_active \
    --csv \
    ./build/bin/stencil_naive --nx 512 --steps 1
```

**What the naive kernel actually shows, measured on the reference
hardware** (the `(6R+4) × sizeof(float) × N_interior` formula below is
the *theoretical* compulsory-traffic estimate assuming zero L2 reuse --
it is not what `dram__bytes_read.sum` reports):

- Formula estimate: 28 × 4 × 509³ ≈ **14.7 GB** per step for 512³ (R=4)
- `dram__bytes_read.sum` **measured: ~1.59 GB** per step -- about 9×
  lower than the formula, because this GPU's L2 cache absorbs most of
  the redundant neighbour reads before they reach DRAM

- `dram__throughput.avg.pct_of_peak_sustained_elapsed` **measured:
  95.7% of peak** (with `--clock-control none`) -- naive is close to saturating DRAM bandwidth
  overall, not because of poor coalescing (the uncoalesced-access rule
  found only ~6% excess sectors here) but simply because of the total
  compulsory byte volume at this GPU's ~192 GB/s peak

See docs/CASE_STUDY.md section 3.3 for the full derivation and the
caveats on every "GB/s" figure: since L1 the printed GB/s is compulsory
traffic over time (close to DRAM at large grids, L2-inflated at small
ones), and the old no-reuse figure is `model_bandwidth_gbs`.

---

## Step 3 — Profile the tiled kernel

```bash
./scripts/profile_nsight.sh compute ./build/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 --steps 1
# → nsight-output/<timestamp>/stencil_tiled.ncu-rep
```

**What the tiled kernel actually shows, measured on the reference
hardware** (TILE_Y=8, PENCIL_Z=12 -- this repo's current default; full
sweep and counter detail in docs/CASE_STUDY.md section 3.2-3.4):

- Formula estimate: 4 × 512³ ≈ **0.54 GB** per step -- the kernel's own
  design goal of ~27× fewer DRAM reads than naive's formula estimate
- `dram__bytes_read.sum` **measured: ~1.98 GB** per step -- *more* DRAM
  traffic than naive's measured 1.59 GB, the opposite of the design
  intent. Reads follow a two-term model (compulsory bytes + a re-read
  of the 2R halo planes at each Z-slab boundary) that fits every
  PENCIL_Z value tested within ~3%

- `l1tex__t_sector_hit_rate.pct` **measured: ~28.8%** (not 70-80%)

- `dram__throughput...` **measured: ~89.0% of peak** (not 20-30%) --
  still close to the DRAM ceiling, not shifted away from it

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` was not
  measured in this investigation, so the "SMEM_X=40 avoids bank
  conflicts" claim is unverified, not confirmed ≈0 -- don't cite it as
  measured. What *was* confirmed: tiled kernels are flagged for
  uncoalesced **global** memory access (~23% excess sectors, vs ~6%
  for naive), a separate issue from SMEM bank conflicts; a 40-wide
  block (not a multiple of 32) is the suspected cause, inferred from
  source, not independently isolated

- The actual measured cause of the slowdown is barrier stalls (33-51%
  of warp-active cycles, vs 0% for naive) and 1.6-2.4× more warp
  instructions per output cell, not a DRAM-vs-SMEM latency shift

### Side-by-side comparison (ncu-ui diff view)

1. Open `ncu-ui`
2. File → Open → select `stencil_naive.ncu-rep`
3. Open the Baselines panel (bottom of the window), select the naive report's tab and use the panel's add-baseline toolbar icon; then switch to `stencil_tiled.ncu-rep`
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

> ⚠️ The summary diff table this mode prints is documented elsewhere in
> this repo as broken (`ncu --csv` emits long-format CSV -- one row per
> metric, generic `"Metric Name"`/`"Metric Value"` columns -- but the
> script's parser looks metric names up as if they were column headers,
> so every lookup silently returns nothing; see BUILD_AND_RUN.md's
> Troubleshooting table). This has **not** been independently re-run and
> confirmed as part of the current investigation -- see checklist item
> M6. Until that's done, use Section 8C-8D of `BUILD_AND_RUN.md` for a
> validated manual `ncu`/`ncu-ui` workflow instead of this script's
> `compare` mode.

---

## Step 5 — Occupancy analysis

For the tiled kernel, run the occupancy calculator:

```bash
ncu --set roofline ./build/bin/stencil_tiled --nx 512 --steps 1
```

Measured on this repo's reference hardware (RTX 4050 Laptop GPU, Ada
Lovelace SM 8.9, 20 SMs), at TILE_Y=8/PENCIL_Z=12 -- the current default:

| Parameter | Value |
|---|---|
| Block size | 40 × 16 = 640 threads |
| Registers/thread | 48 |
| SMEM/block | 2,560 B |
| Theoretical occupancy | 83.3% |
| Achieved occupancy (`sm__warps_active`) | 82.8% |

**`TILE_Y=4` does not improve occupancy-limited performance here -- it
measures as the *slowest* valid configuration tested (0.58× of naive,
vs 0.71× at TILE_Y=8/PENCIL_Z=12).** A smaller `TILE_Y` shrinks a
fixed `2·STENCIL_RADIUS`-cell halo border's share of *useful* block
work faster than it shrinks the block itself, so the occupancy gain
doesn't translate to a speed gain. More generally, occupancy does not
order the tiled configurations on this GPU at all: the
lowest-occupancy sweep point (62.5%, at TILE_Y=16) is faster than
several higher-occupancy ones, and occupancy only becomes the
dominant factor once PENCIL_Z reaches 28 or more (where the register
count jumps and a 640-thread block drops from two resident blocks per
SM to one). See docs/CASE_STUDY.md section 3.2-3.4 for the full sweep
and the counter-level explanation -- don't re-derive the "shrink the
block for more occupancy" intuition from scratch, it doesn't hold here.

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
         AI ≈ 3.5 FLOP/byte for this stencil
```

Both naive and tiled points appear at the **same X position** (same
arithmetic intensity — the kernel does the same math either way). On
datacenter-class hardware (this kernel's original design target) the
tiled point is expected to sit **higher on Y**, from better bandwidth
utilisation, without changing the algorithm.

**On this repo's reference hardware the tiled point sits LOWER than
naive, not higher** -- naive attains ~625 GFLOP/s at 512³, tiled only
~449 GFLOP/s (0.71× of naive's throughput). The interview-soundbite
framing below is the textbook story for the kernel's intended
A100-class target; state plainly which hardware you're describing
before using it, since this repo's own reference measurements are the
counter-example, not confirmation:

*"The optimisation doesn't change the arithmetic intensity -- whether
it improves bandwidth utilisation, and whether that improvement
outweighs its own overhead, is hardware-dependent and must be
measured, not assumed."* See `docs/CASE_STUDY.md` for what actually
limits this kernel on Ada Lovelace: barrier stalls and increased DRAM
traffic, not an arithmetic-intensity effect.

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
