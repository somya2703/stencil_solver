# Performance Analysis: Tiled Kernel Regression on Ada-Class Laptop GPUs

**Subject:** `kernel_tiled` (shared-memory + register-pencil stencil kernel)
**Hardware under test:** NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, SM 8.9, 20 SMs, 6.1 GB, ~96 GB/s peak DRAM bandwidth)
**Status:** Root-caused and confirmed with hardware counters. Not resolved — see Recommendations.

---

## 1. Executive summary

The tiled kernel (`src/kernels/stencil_tiled.cu`) is designed to beat the
naive global-memory kernel (`src/kernels/stencil_naive.cu`) by staging data
in shared memory and a register pencil, cutting redundant DRAM traffic from
`(6R+4)` bytes/point down to roughly `1` byte/point (per the kernel's own
header comment, and the README's claim of a ~3.5× speedup on an A100).

**On this GPU, across every valid tuning configuration tested, the tiled
kernel is slower than naive — by 31–75% depending on tuning — and it moves
*more* DRAM traffic than naive, not less.** This directly contradicts the
kernel's own design assumption. The cause is fully identified: the register
pencil forces high per-thread register usage, which caps SM occupancy far
below naive's, and the resulting loss of warp-level parallelism outweighs
any bandwidth savings — savings which, on this GPU, mostly don't exist
anyway because the L2 cache already absorbs most of naive's "redundant"
reads before they reach DRAM.

This is not a bug in the sense of incorrect output — every correctness test
(`test_kernel_correctness`, `test_kernels`) passes on both kernels. It is an
**architecture-portability failure of a performance optimization**: a
technique tuned and validated for one class of GPU (A100/Ampere, 108 SMs,
thin L2 relative to compute) actively regresses on another class (Ada
laptop, 20 SMs, comparatively large L2) using the same source code and the
same tunable knobs the codebase already exposes for exactly this purpose.
This conclusion is supported by **four independent measurement methods**
(plain wall-clock timing, a systematic parameter sweep, isolated
single-launch Nsight Compute profiling, and an aggregate 50-step Nsight
Systems trace) that all agree on the magnitude of the regression, with the
Nsight Systems data additionally confirming that the entire slowdown is
intra-kernel — none of it comes from host-side scheduling overhead.

---

## 2. Background — what the tiled kernel claims, and why

### 2.1 The optimization's stated goal

From `src/kernels/stencil_tiled.cu`'s own header comment:

- Naive kernel: `(6R + 4) × sizeof(real_t)` bytes of DRAM traffic per
  interior point per step (assumes zero reuse across threads).
- Tiled kernel: reads the XY plane into shared memory once per tile
  (reused across `TILE_Y` rows) and keeps the Z-neighborhood in a
  per-thread register "pencil" of depth `PENCIL_Z + 2R`, reused across
  `PENCIL_Z` Z-steps without re-touching global memory.
- Claimed result: `dram__bytes_read.sum` drops from ~14.7 GB/step to
  ~0.54 GB/step at 512³ (a ~27× reduction), for a claimed ~3.5× wall-clock
  speedup on an A100 80GB.

### 2.2 The tunable knobs already built for this purpose

`CMakeLists.txt` exposes `STENCIL_TILE_Y` and `STENCIL_PENCIL_Z`
specifically because the defaults (`TILE_Y=8`, `PENCIL_Z=16` — a
640-thread block, 24-deep register pencil) were sized for A100-class
SM counts and register files. The patch that introduced them says
outright: *"a 20-SM laptop Ada part often does better with a smaller
pencil/block."* This is the exact GPU class under test here.

### 2.3 Baseline evidence that something is wrong

The repository's own `docs/BENCHMARKS.md`, generated on this same GPU
class, already showed the problem before this investigation began:

| Grid | Naive BW (GB/s) | Tiled BW (GB/s) | "Speedup" |
|---|---|---|---|
| 64³ | 1661.6 | 709.3 | 0.43× |
| 128³ | 1029.9 | 764.6 | 0.74× |
| 256³ | 1221.5 | 842.3 | 0.69× |
| 512³ | 1248.2 | 848.3 | 0.68× |

Every row shows tiled slower than naive. (Note: the report's printed
"Average speedup: 1.56×" headline number is a separate, unrelated bug —
see Appendix A.4 — and should be ignored; the per-row data above is the
correct signal.)

---

## 3. Problem statement

> On an Ada Lovelace laptop GPU (RTX 4050, 20 SMs), the shared-memory
> tiled stencil kernel runs consistently and reproducibly slower than
> the naive global-memory kernel, across the full valid range of
> `TILE_Y`/`PENCIL_Z` tuning parameters, contradicting the kernel's
> documented design intent and the ~3.5× speedup claimed for A100-class
> hardware.

---

## 4. How it shows up

This section documents every place the problem was independently
observed, in the order it was discovered, because the fact that four
different measurement paths agree is itself part of the evidence.

### 4.1 First observation — `stencil_tiled` console output

```
Grid        Naive ms   BW (GB/s)    Tiled ms   BW (GB/s)   Speedup
--------  ----------  ----------  ----------  ----------  --------
512         11.457 ms    1251.5       19.858 ms     722.1       0.58×
```

Config: default `TILE_Y=4, PENCIL_Z=8` (an initial, unvalidated tuning
guess). Tiled is 1.7× *slower* in wall-clock time than naive.

### 4.2 Second observation — parameter sweep

A 6-point sweep over `TILE_Y ∈ {8, 16}, PENCIL_Z ∈ {8, 16, 24}` (see
§6.1 for why `TILE_Y=4` and `TILE_Y=32` were excluded/failed):

| TILE_Y | PENCIL_Z | naive_ms | tiled_ms | speedup |
|---|---|---|---|---|
| 8 | 8 | 11.462 | 16.961 | 0.676× |
| 8 | 16 | 11.543 | 16.853 | 0.685× |
| 8 | 24 | 11.531 | 20.334 | 0.567× |
| 16 | 8 | 11.540 | 17.869 | 0.646× |
| 16 | 16 | 11.459 | 16.709 | 0.686× |
| **16** | **24** | **11.547** | **16.712** | **0.691×** (best found) |

**Every single configuration in the valid range underperforms.** The
best case found is still a 31% regression versus naive.

### 4.3 Third observation — isolated single-kernel-launch profiling

Because `stencil_tiled`'s binary always runs naive *then* tiled in one
process (see Appendix A.2 for why this complicated measurement),
`kernel_tiled` was isolated with:

```bash
ncu --launch-skip 1 --launch-count 1 \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,l1tex__hit_rate.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread \
    -o nsight-output/tiled_only \
    ./build-best/bin/stencil_tiled --nx 512 --steps 1 --warmup 0
```

at the best-found config (`TILE_Y=16, PENCIL_Z=24`). This is a clean,
single-launch, hardware-counter-level measurement — independent of both
the console-printed timer and the earlier sweep — and it agrees with
both:

| Metric | `kernel_naive` (baseline) | `kernel_tiled` | Delta |
|---|---|---|---|
| Duration | 11.47 ms | 19.25 ms | **+67.77%** |
| Compute Throughput | 66.86% | 43.84% | **−34.43%** |
| `sm__warps_active` (occupancy proxy) | — | 62.10% | **−26.09%** |
| Registers/thread | 40 | 62 | **+55.00%** |
| Block size (threads) | 32×4×2 = 256 | 40×24×1 = 960 | — |
| DRAM bytes read | (baseline) | 1.79 GB | **+13.02%** |
| DRAM bytes write | (baseline) | 517.25 MB | +0.02% (flat) |

Three independent methods — plain wall-clock timing in the binary
itself, a systematic parameter sweep, and isolated Nsight Compute
hardware-counter profiling — all converge on the same conclusion.

### 4.4 Fourth observation — Nsight Systems, aggregated across the full timed run

The measurements in §4.1–4.3 either used a single isolated kernel
launch or the binary's own aggregate timer. As a fourth, independent
check, Nsight Systems was used to capture the full 60-launch sequence
(10 warmup + 50 timed steps) for both kernels in one process, and to
separately verify that no host-side scheduling overhead was
contributing to the slowdown:

```bash
./scripts/profile_nsight.sh systems ./build-best/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 --steps 50 --warmup 10
```

`cuda_gpu_kern_sum` (GPU-side kernel execution time, aggregated over
all 60 instances of each kernel — not a single-launch snapshot):

| Kernel | Instances | Avg | Min | Max | StdDev |
|---|---|---|---|---|---|
| `kernel_naive` | 60 | 12.19 ms | 11.36 ms | 13.58 ms | 837 μs (~6.9% of mean) |
| `kernel_tiled` | 60 | 16.66 ms | 15.92 ms | 17.42 ms | 318 μs (~1.9% of mean) |

Ratio: 16.66 / 12.19 = **1.367× slower**, consistent with the
0.68–0.73× "speedup" figures seen in §4.1–4.3. This is the fourth
independent measurement method to agree on the same magnitude of
regression, aggregated over 50 timed steps rather than a single
launch — ruling out the possibility that the earlier single-launch
`ncu` measurement was a one-off anomaly.

`cuda_api_sum` was also inspected specifically to rule out host-side
causes:

- `cudaStreamSynchronize`: only **4 calls** total, all with durations
  matching the one-time initial-condition upload synchronization in
  `SolverBase::init()`, not the per-step loop.
- `cudaEventSynchronize`: only **2 calls** (one per solver instance),
  matching `GpuTimer::elapsed_ms()`'s single blocking wait at the end
  of each `run()` call — not per-step.
- `cudaLaunchKernel`: 120 calls (60 naive + 60 tiled) averaging **3.5 μs**
  each — negligible launch overhead.
- **Zero synchronization calls occur between the 50 timed kernel
  launches themselves** in either kernel's loop.

This confirms the timed loop launches kernels back-to-back
asynchronously with no CPU round-trip in between, for both kernels
equally — see §5.5 for why this matters.

---

## 5. What it signifies — the causal chain

The Nsight Compute data in §4.3 lets us state the mechanism precisely,
not just observe the symptom.

### 5.1 The DRAM-traffic assumption is falsified on this GPU

The tiled kernel's entire justification is reduced global memory
traffic. The measurement shows the opposite: **tiled reads 13% *more*
DRAM bytes than naive**, not the ~27× *fewer* bytes the kernel's own
header comment predicts for A100-class hardware.

This is explained by what naive's numbers already implied before this
investigation started: naive's reported bandwidth (1000+ GB/s) is
**far above this GPU's physical 96 GB/s DRAM peak.** That is only
possible if most of naive's "redundant" neighbor reads are being served
from the L2 cache rather than DRAM — i.e., the hardware's implicit
cache is *already doing the job* the tiled kernel's shared memory and
register pencil were manually built to do. Ada-class GPUs are known to
carry proportionally large L2 caches relative to their SM count
(a design trend distinct from the datacenter Ampere/A100 line this
kernel was originally tuned against), which is consistent with this
behavior.

Given that, tiling adds a real, new cost — each thread individually
streams its own Z-column of look-ahead values from global memory into
its register pencil, once per block, with no cross-thread sharing in
the Z direction — without removing a cost that was mostly already being
absorbed by hardware cache. Net effect: more real DRAM traffic, not
less.

### 5.2 Register pressure caps occupancy

Independent of the DRAM question, the kernel's register footprint is
the dominant issue:

- The register pencil holds `PENCIL_Z + 2·STENCIL_RADIUS` live
  `real_t` values per thread. At the best-found tuning
  (`PENCIL_Z=24, R=4`): `24 + 8 = 32` live values, before the compiler
  even accounts for the rest of the kernel's working state.
- Measured result: **62 registers/thread**, vs. naive's 40 — a 55%
  increase.
- Block size at this tuning is `SMEM_X × SMEM_Y × 1 = 40 × 24 × 1 = 960`
  threads. `960 × 62 = 59,520` registers demanded by a single block.
- Ada SMs have a fixed-size register file (independent of how many SMs
  the chip has in total). A block this register-hungry can leave room
  for **at most one resident block per SM** — there is no register
  budget left over for a second block to run concurrently, no matter
  how many warp schedulers or how much shared memory is available.
- Measured `sm__warps_active` (a direct occupancy proxy) confirms this:
  **62.10%, a 26% drop from naive.** Fewer warps in flight means less
  latency-hiding capacity, which shows up directly as the measured
  **34% drop in compute throughput.**

### 5.3 The full causal chain

```
Deep register pencil (PENCIL_Z + 2R live values/thread)
        │
        ▼
High register count/thread (62 vs. 40, +55%)
        │
        ▼
Register file exhausted per SM → fewer concurrent blocks
        │
        ▼
Lower achieved occupancy (sm__warps_active: 62.10%, −26%)
        │
        ▼
Less warp-level parallelism to hide memory/pipeline latency
        │
        ▼
Lower compute throughput (43.84%, −34%)
        │
        ▼
Higher wall-clock time (19.25ms vs. 11.47ms, +68%)
```

### 5.5 Ruling out host-side scheduling as a contributing cause

The causal chain in §5.3 is built entirely from per-kernel hardware
counters (registers, occupancy, throughput, DRAM bytes). Before
attributing the *entire* wall-clock regression to that chain, it was
necessary to rule out a distinct alternative explanation: that some of
the slowdown came from CPU-GPU scheduling gaps (sync stalls, launch
overhead, poor kernel-to-kernel overlap) rather than the kernels'
own execution efficiency.

The Nsight Systems data in §4.4 answers this directly. Across the
entire 50-step timed loop, for both kernels: no `cudaStreamSynchronize`
or `cudaEventSynchronize` calls occur between individual kernel
launches, and `cudaLaunchKernel` overhead is negligible (~3.5 μs vs.
~12–17 ms of kernel execution time — well under 0.1%). Kernels are
queued and executed back-to-back exactly as `SolverBase::run()`'s
design intends (a single warmup loop, then a single timed loop, with
only one synchronization at the very end).

**This means 100% of the measured 1.367× slowdown is attributable to
intra-kernel execution inefficiency — the register-pressure/occupancy
mechanism in §5.1–5.3 — and none of it to host-side scheduling
overhead.** This is a meaningful confirmation, not a redundant check:
it closes off the possibility that fixing a scheduling issue (e.g. an
unnecessary sync, or poor stream usage) could recover some or all of
the lost performance. The only path to recovering it is changing the
kernel's own register/occupancy profile, per the recommendations in
§8.

**Secondary observation (unplanned, noted for completeness):**
`kernel_naive`'s per-step timing variance (StdDev ≈ 837 μs, ~6.9% of
its mean) is proportionally about 3.6× larger than `kernel_tiled`'s
(StdDev ≈ 318 μs, ~1.9% of its mean), with naive's individual step
times ranging from 11.36–13.58 ms across the 50-step run. One plausible
explanation is that naive's higher-occupancy, more DRAM-request-heavy
execution pattern interacts more with laptop-class thermal/clock
(DVFS) variance than tiled's lower-occupancy, register-bound pattern —
but this was not independently investigated and should be treated as
an observation, not a confirmed finding.

### 5.4 Broader significance

This demonstrates a specific, general failure mode in
architecture-tuned GPU kernels: **a manual data-locality optimization
(shared memory + register blocking) that trades hardware-cache reliance
for explicit register/SMEM management is only a net win if the target
GPU's implicit cache is *worse* at the job than the hand-rolled
replacement.** On a GPU where L2 is proportionally large relative to
compute (as appears to be the case for this Ada-class laptop part
relative to the register file it also has to share), the hand-rolled
version can lose on both axes at once — it doesn't reduce real DRAM
traffic much (cache was already doing that), and it actively harms
occupancy (register cost that the naive kernel never pays).

The practical takeaway: **"tiling is faster" is not a hardware-portable
truth for this codebase.** It must be re-validated per GPU architecture
class, and the existing `TILE_Y`/`PENCIL_Z` knobs — while necessary —
are not sufficient to recover the intended speedup on this class of
hardware; every point in the valid tuning space tested here still
underperforms naive.

---

## 6. Supporting investigation detail

### 6.1 Why `TILE_Y=4` made things *worse*, not better

The initial hypothesis ("fewer SMs → shrink the tile") was tested first
and found to backfire, which is itself informative. Every block pays a
fixed `2·STENCIL_RADIUS = 8`-cell halo border in both X and Y regardless
of tile size (`SMEM_X = TILE_X + 2R`, `SMEM_Y = TILE_Y + 2R`, and only
the interior `[R, SMEM-R)` region of threads produces output). Shrinking
`TILE_Y` shrinks the *useful* fraction of each block faster than it
shrinks the block's overhead:

| TILE_Y | SMEM_Y | Y-interior fraction | Approx. total block efficiency (×80% fixed X-fraction) |
|---|---|---|---|
| 4 | 12 | 33% | 27% |
| 8 (orig. default) | 16 | 50% | 40% |
| 16 | 24 | 67% | 53% |
| 32 | 40 | 80% | 64% |

`TILE_Y=4` has the *worst* efficiency of any value tested — more of
every block is spent loading halo cells that never contribute an
output. This explains why the very first, unvalidated tuning attempt
(§4.1) regressed even further than the eventual best-found
configuration.

### 6.2 Why `TILE_Y=32` is not a valid configuration

At `TILE_Y=32`: `SMEM_X × SMEM_Y = 40 × 40 = 1600` threads/block —
over CUDA's hard 1024-thread-per-block limit on every current
architecture. This produces `invalid configuration argument` at launch,
which is CUDA correctly rejecting an illegal kernel launch, not a
tuning failure. Given `TILE_X=32` is fixed (required to equal the warp
size for coalesced X loads), the real ceiling is `TILE_Y ≤ 17`
(`40 × (TILE_Y+8) ≤ 1024`). `TILE_Y=16` — the best-found value in
§4.2 — sits right at the practical edge of the legal range.

### 6.3 Measurement methodology notes

Getting the isolated `kernel_tiled`-only profile (§4.3) required
working around two tooling issues, documented in full in Appendix A
since they are relevant to anyone else attempting to reproduce this
analysis:

- `stencil_tiled`'s binary always launches `kernel_naive` before
  `kernel_tiled` in the same process, so a naive `ncu --launch-count 1`
  invocation captures the *wrong* kernel.
- The repo's `scripts/profile_nsight.sh` automated diff-table feature
  silently produces all `—` (no data) due to a CSV-parsing bug,
  independent of the launch-order issue above.

The fix used throughout this analysis:
`--launch-skip 1 --launch-count 1` combined with running the binary
itself at `--steps 1 --warmup 0`, guaranteeing exactly two kernel
launches occur (one naive, one tiled) and that `ncu` lands on the
second one precisely.

---

## 7. Reproduction steps

```bash
# 1. Configure at a candidate tuning
sudo rm -rf build-best
cmake -B build-best -S . -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DSTENCIL_TILE_Y=16 -DSTENCIL_PENCIL_Z=24 \
    -DSTENCIL_ENABLE_NVTX=ON
cmake --build build-best --parallel $(nproc)

# 2. Confirm the wall-clock regression
./build-best/bin/stencil_tiled --nx 512 --ny 512 --nz 512 --steps 200 --warmup 20

# 3. Isolate kernel_tiled with Nsight Compute
ncu --launch-skip 1 --launch-count 1 \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
launch__registers_per_thread,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    -o nsight-output/tiled_only \
    ./build-best/bin/stencil_tiled --nx 512 --steps 1 --warmup 0

# 4. Also capture a naive-only baseline report for comparison
ncu --launch-count 1 -o nsight-output/naive_only \
    ./build-best/bin/stencil_naive --nx 512 --steps 1 --warmup 0

# 5. Open both in the GUI and add the naive report as a Baseline
#    (File > Open both .ncu-rep files as tabs; select the naive tab;
#    use the "Add Current Result as Baseline" icon above the empty
#    Baselines panel; switch back to the tiled tab to see diff bars)
ncu-ui nsight-output/naive_only.ncu-rep
```

Full parameter sweep (§4.2):

```bash
for ty in 8 16; do
  for pz in 8 16 24; do
    sudo rm -rf build-tune
    cmake -B build-tune -S . -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="89" \
        -DSTENCIL_TILE_Y=$ty -DSTENCIL_PENCIL_Z=$pz \
        -DSTENCIL_BUILD_TESTS=OFF -DSTENCIL_BUILD_BENCHMARKS=OFF > /dev/null 2>&1
    cmake --build build-tune --parallel $(nproc) --target stencil_tiled > /dev/null 2>&1
    ./build-tune/bin/stencil_tiled --nx 512 --ny 512 --nz 512 --steps 200 --warmup 20 \
        | grep -E "^  (naive|tiled)"
  done
done
```

Nsight Systems aggregate confirmation (§4.4 / §5.5):

```bash
./scripts/profile_nsight.sh systems ./build-best/bin/stencil_tiled \
    --nx 512 --ny 512 --nz 512 --steps 50 --warmup 10

# Inspect kernel-level aggregate timing and host-side API calls directly
# from the generated .sqlite without opening the GUI:
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
    nsight-output/<timestamp>/stencil_tiled.nsys-rep

# Or open the full timeline visually:
nsys-ui nsight-output/<timestamp>/stencil_tiled.nsys-rep
```

---

## 8. Recommendations / future work

1. **Do not present the tiled kernel as universally faster.** Any
   report or README claim of "~3.5× speedup" should be scoped
   explicitly to A100/Ampere-class datacenter GPUs, with this
   document (or a summary of it) linked as the counter-example for
   Ada-class laptop parts.
2. **A shallower register pencil is the most promising avenue**, since
   register pressure — not DRAM traffic — is the dominant, directly
   measured cause. Values of `PENCIL_Z` below the tested range (e.g.
   4) were not yet tried; the halo-overhead math in §6.1 applies to
   `TILE_Y`, not `PENCIL_Z`, so a smaller pencil doesn't carry the same
   efficiency penalty and is worth testing on its own.
3. **Consider an alternative Z-handling strategy for high-L2/low-SM
   GPUs** that relies more on the hardware cache and less on an
   explicit register pencil — effectively a partial rollback toward
   the naive kernel's access pattern, but keeping the XY shared-memory
   tile (which was not shown to be the source of the regression by
   itself; the register pencil was).
4. **Promote the sweep script in §7 into a first-class repo tool**
   (e.g. `scripts/tune_tiled.sh`), since manually finding
   architecture-appropriate `TILE_Y`/`PENCIL_Z` values is clearly
   necessary per-GPU and currently undocumented as a required step.
5. **Fix the two tooling bugs uncovered during this investigation**
   (Appendix A.2, A.3) so future contributors can reproduce this
   analysis without independently rediscovering them.
6. **Re-run this same analysis on an actual A100 or A6000** if one
   becomes available, to confirm the kernel does behave as designed
   there — this document only demonstrates the regression on
   Ada-class laptop silicon; it does not by itself prove the kernel is
   flawed in general, only that it is not portable as-is.

---

## Appendix A — Tooling issues discovered during this investigation

These are documented separately because they are artifacts of the
*measurement process*, not of the kernel's performance itself, but they
materially slowed down reaching the conclusion above and would do the
same to anyone else attempting to reproduce it.

### A.1 CMakeCache path collisions between Docker and host builds

Reusing the same `build/`-named directory across a container build
(mounted at `/workspace`) and a host build (at the actual repo path)
produces:
```
CMake Error: The current CMakeCache.txt directory ... is different
than the directory ... where CMakeCache.txt was created.
```
Compounded by container-created files being root-owned, so a plain
`rm -rf build` fails with `Permission denied` and leaves the stale
cache in place, silently reproducing the same error on the next
attempt. Fix: `sudo rm -rf build` (or the affected directory name),
and use distinct directory names per build context going forward
(`build`, `build-cpu`, `build-tune`, `build-best`, etc., never reused
across Docker/host).

### A.2 `--launch-count 1` captures the wrong kernel for multi-kernel binaries

`scripts/profile_nsight.sh`'s `run_compute()` always passes
`--launch-count 1` to bound profiling overhead. This is correct for a
binary that launches one kernel type, but `stencil_tiled`'s `main()`
(per its own doc comment) always runs `kernel_naive` first, then
`kernel_tiled`, in one process. `--launch-count 1` stops after the
*first* launch encountered — naive — so `kernel_tiled` is never
profiled at all when using the wrapper script on `stencil_tiled`
directly. Fix used throughout this document:
`--launch-skip 1 --launch-count 1`, combined with running the target
binary at `--steps 1 --warmup 0` so exactly two launches occur and the
skip count is unambiguous.

### A.3 `profile_nsight.sh`'s diff-table parser reads the wrong CSV shape

`ncu --csv --metrics a,b,c` emits **long-format** CSV — one row per
metric, with generic columns literally named `"Metric Name"` and
`"Metric Value"` — not one column per requested metric. The script's
`extract()` does:
```python
row = rows[-1]
return {k.strip(): v.strip() for k, v in row.items()}
```
then later looks up `n.get("dram__bytes_read.sum", "")` — a **column**
name that never exists in long-format CSV (`dram__bytes_read.sum` only
ever appears as a *value* under the `Metric Name` column). Every
lookup silently returns empty, so the printed diff table shows `—` for
every metric with no error raised. Fix: pivot the parsed rows into a
`{row["Metric Name"]: row["Metric Value"] for row in rows}` dict before
doing any `.get()` lookups.

### A.4 `plot_results.py`'s "Average speedup" line is inverted

`write_markdown_report()` computes the per-row table correctly
(`naive.time_ms / tiled.time_ms`, so >1× means tiled is faster) but
computes the separate summary-line average from the reciprocal
(`tiled.time_ms / naive.time_ms`). This is why a prior run of this repo
showed every row in `docs/BENCHMARKS.md` as sub-1× (tiled slower) while
the "Average speedup" line beneath the table still printed a
misleadingly positive 1.56×. Fix: use
`n_r["time_ms"] / t_r["time_ms"]` in both places.

### A.5 `docs/nsight_workflow.md`'s "File → Add Baseline" instruction is stale

Current Nsight Compute UI versions do not expose "Add Baseline" under
the File menu. Baselines are managed via the dedicated **Baselines**
panel (bottom of the window): open both reports as tabs, select the
tab you want as the reference, and use the "Add Current Result as
Baseline" icon in the small toolbar directly above the (initially
empty) Baselines panel.

### A.6 `nvtx3/nvtx3.hpp` is not present in a stock host CUDA toolkit install

`STENCIL_ENABLE_NVTX` defaults to `ON`, but the NVTX3 header is fetched
manually inside `Dockerfile.dev` from the upstream NVIDIA/NVTX GitHub
repo — it is not part of a standard CUDA toolkit install on the host.
Building locally without first fetching this header fails with
`fatal error: nvtx3/nvtx3.hpp: No such file or directory`. Fix: either
fetch the header manually (mirroring the Dockerfile step) or configure
with `-DSTENCIL_ENABLE_NVTX=OFF` for host builds that don't need
profiler range markers.

### A.7 Grid-size sweep trap in `main_naive.cpp`

Passing `--nx 256 --ny 256 --nz 256` (the compiled-in default) to
`stencil_naive` is indistinguishable from passing no size flags at
all, silently triggering a 3-size sweep (`128³, 256³, 512³`) instead of
the single run a user likely intended. Any size other than exactly
256/256/256 avoids this.

### A.8 Small-grid GB/s figures exceed physical peak bandwidth

`bandwidth_bytes_per_step()` assumes zero cache reuse between steps.
At small grid sizes (roughly ≤192³ on this GPU's cache size), the
entire per-step working set fits in L2, so most traffic is served from
cache rather than DRAM — the reported "GB/s" is a theoretical
worst-case figure, not a measured hardware counter, and can (and does,
per §2.3's table) exceed the GPU's actual 96 GB/s DRAM peak by an order
of magnitude. This is the same underlying phenomenon as this document's
main finding (§5.1) — L2 absorbing traffic the byte-counting formula
assumes goes to DRAM — just observed via a different code path
(the analytic formula in `types.hpp` vs. the hardware counter in
`ncu`).

### A.9 Unit test count

`tests/CMakeLists.txt`'s `unit`-labeled targets
(`test_fd_coeffs`, `test_grid`, `test_timer`, `test_solver_cpu`,
`test_solver`, `test_kernels`) total **59** test cases, matching the
repository's own initial-commit message ("59/59 tests passing"), not
43 as an earlier draft of the build guide stated.
