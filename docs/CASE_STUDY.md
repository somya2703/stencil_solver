# Naive vs. shared-memory tiled stencil kernels on an RTX 4050 Laptop GPU

A counter-backed performance case study of `stencil-solver` (8th-order 3-D wave stencil, fp32).

- **Repository state:** commit `7d97de8` (`main` = `origin/main`), tracked tree clean
- **Measurements taken:** 2026-09-20, single machine, single GPU
- **Reference documents:** the uploaded `BUILD_AND_RUN.md` (the "guide") and `PERFORMANCE_ANALYSIS.md` (the "analysis doc"). Other copies in the repo were ignored.

**Evidence tags used throughout**

| Tag | Meaning |
|---|---|
| **[V]** | Verified: measured directly, or read from a hardware counter |
| **[D]** | Derived: arithmetic on measured values |
| **[I]** | Inferred: consistent with the data but not directly tested |
| **[U]** | Unverified: from source reading only; nothing was run |

---

## 1. Summary

1. **The naive kernel beats the tiled kernel at every grid size and every tile configuration tested.** With the guide's Step 3 build (`TILE_Y=4`, `PENCIL_Z=8`) tiled runs at 0.42–0.68× of naive (0.58× at 512³). The best tiled configuration found is `TILE_Y=8, PENCIL_Z=12` at about **0.71×**, confirmed in 5/5 back-to-back repeats against `8/16` (16.12 ms vs 17.02 ms, +5.6%). **[V]**
2. **Naive is already at the practical DRAM limit.** It reads 1.59 GB and writes 0.517 GB per step, which is the compulsory traffic, and Nsight Compute reports 95.7% of peak DRAM throughput. The compulsory-traffic floor is 10.96 ms against 11.46 ms measured, so at most ~4.5% headroom exists. No kernel that only improves data reuse can meaningfully beat it, and the ~3.5× speedup described in the README was never available on this GPU. **[V/D]**
3. **The DRAM peak is ~192 GB/s, not the 96 GB/s the code prints.** The bandwidth formula omits the DDR ×2 factor. The code's "effective GB/s" is a model number (analytic bytes ÷ time) that tracks L2 sector traffic, not DRAM. **[V/D]** *(Superseded by L1, commit b67dca6: the printed `bandwidth_gbs` now uses compulsory traffic, 4 x interior x 4 B per step; the old figure is `model_bandwidth_gbs`. This finding describes the earlier output.)*
4. **Tiled kernels move more DRAM bytes than naive** (+12% to +38%), following `read ≈ 1.59 GB + 0.537 GB × 8/PENCIL_Z`. Three out-of-sample predictions of this model landed within 1.6%. **[V]**
5. **Tiled kernels lose mainly to barrier stalls and extra instructions.** Barrier stalls are 33–51% of warp cycles (naive: 0%), and warp-instruction counts are 1.6–2.4× naive's. In an `8/x` block, 10 of 20 warps contain no interior threads. Occupancy explains only the `PENCIL_Z ≥ 28` regime (41.7%) and does not order the configurations. **[V/I]**
6. **Unresolved:** the cause of the +16–18% step between `PENCIL_Z=16` and `20`, and of the +23–26% step between `28` and `32`. Most candidate explanations are excluded (§3.6).
7. **Corrections** to the analysis doc, the guide, and several repo/tooling issues are listed in §5–§6. **Section 9** (multi-GPU) and **Section 10** (CI) of the guide were **not evaluated**.

---

## 2. Environment and method

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop, Ada Lovelace, SM 8.9, 20 SMs, 6141 MiB |
| Driver / toolkit | 595.84 (reports CUDA 13.2) / nvcc 12.8.93 |
| Driver note | Kernel module 595.84 while the user-space packages were 595.99.02 (NVML mismatch seen during the §3.7 clock tests, so the later `bench_stream`, Nsight and compare runs used that mixed state; when it began is unknown). After a reboot the module is 595.99.02. `bench_stream` re-run: read 187.8, write 183.6, copy 174.5 GB/s best, within 0.5% of the earlier runs. One solver point was re-run on the new driver (`stencil_tiled`, 512^3, 200 steps, 20 warmup): naive 11.534 ms, tiled 15.947 ms, ratio 0.72x, within about 0.5% of the earlier runs. The other grid sizes and tile configurations were not re-run. |
| Host toolchain | GCC 13.3.0, CMake 3.28.3, Ninja 1.11.1, Python 3.11.10, Ubuntu (Legion 5 16IRX9) |
| Profilers | Nsight Compute 2025.1.1.0, Nsight Systems (bundled with CUDA 12.8) |
| Build | Release, `CMAKE_CUDA_ARCHITECTURES=89`, NVTX on; fresh `build/` (previous one moved to `build-prev/`) |
| Configurations | Named `TILE_Y/PENCIL_Z` (e.g. `8/12`), set with `-DSTENCIL_TILE_Y` and `-DSTENCIL_PENCIL_Z`. `4/8` is the guide's Step 3 config. |

**Protocol**

- **Predictions first.** Before most runs, an expected value was written down (§4). Failures are reported.
- **Ranking uses the benchmark binaries only** (200 timed steps, 20 warmup). Their run-to-run spread is under 1% (n = 5: sd 0.03–0.06 ms).
- **Nsight Compute is used for counter ratios, not for ranking.** Single-launch durations vary about ±5% between profiles, with a launch-order drift (five consecutive `8/16` launches: 17.00 → 15.62 ms). Profiles used `--clock-control none`.
- **Clock and power state** (`nvidia-smi` sampled during runs): the memory clock was 8001 MHz in all 209 active samples of the tiled run. SM clock averaged 1.86–1.89 GHz in sustained benchmarks and 2.58–2.62 GHz in single-launch `ncu` profiles. Peak power was 55.8 W and peak temperature 58 °C.
- **Peak DRAM bandwidth.** 8001 MHz × 2 (DDR) × 96-bit ÷ 8 = 192.1 GB/s. For the naive kernel, Nsight's `dram__throughput` percentage implies 192.2 GB/s, and an independent read-only kernel (`bench_stream`, §3.7) reaches about 188 GB/s (97.8%). **[D/V]**

---

## 3. Results

### 3.1 Naive vs tiled across grid sizes (guide Step 3 build, `4/8`)

| Grid | Naive ms | Tiled ms | Tiled ÷ naive |
|---|---|---|---|
| 64³ | 0.012 | 0.028 | 0.42× |
| 128³ | 0.187 | 0.276 | 0.68× |
| 192³ | 0.632 | 0.975 | 0.65× |
| 256³ | 1.401 | 2.353 | 0.60× |
| 320³ | 2.759 | 4.654 | 0.59× |
| 384³ | 4.799 | 8.102 | 0.59× |
| 512³ | 11.477 | 19.892 | 0.58× |

Mean ratio 0.59×. A separate `bench_naive` run reproduced naive at 512³ (11.462 ms) and 256³ (1.403 ms) within 1%. **[V]**

A cold-process, short-warmup run (`--warmup 10`) measures naive 256³ at ~1.57 ms, up to 12% slower than the 1.40 ms steady state reached with `--warmup 300`. Cold-run timings are therefore not comparable with the benchmarks. **[V]**

### 3.1b Naive vs tiled across grid sizes (best config, `8/12`)

Same protocol as §3.1 (`bench_tiled`, 200 steps, 20 warmup, one run per size), built with
`TILE_Y=8, PENCIL_Z=12`.

| Grid | Naive ms | Tiled ms | Tiled ÷ naive |
|---|---|---|---|
| 64³ | 0.009 | 0.019 | 0.49× |
| 128³ | 0.157 | 0.200 | 0.79× |
| 192³ | 0.582 | 0.740 | 0.79× |
| 256³ | 1.399 | 1.807 | 0.77× |
| 320³ | 2.758 | 3.528 | 0.78× |
| 384³ | 4.801 | 6.717 | 0.71× |
| 512³ | 11.463 | 15.992 | 0.72× |

Mean ratio about 0.72× (0.59× at `4/8`). Tiled is slower at every size; naive timings
match §3.1 within about 1% at the sizes where both were measured, and the 512³ ratio
agrees with the sweep in §3.2 (0.716×). Single runs, no confidence intervals. **[V]**

### 3.2 Tile-parameter sweep (512³, 200 steps; naive ≈ 11.45–11.56 ms throughout)

| Config | Regs | Tiled ms | Tiled ÷ naive |
|---|---|---|---|
| 4/8 | 40 | 19.907 | 0.58 |
| 8/8 | 47 | 17.000 | 0.675 |
| **8/12** | 48 | **16.027** | **0.716** |
| 8/16 | 48 | 16.842 / 16.941 | 0.686 / 0.676 |
| 8/20 | 48 | 19.998 | 0.573 |
| 8/24 | 48 | 20.315 / 20.480 | 0.568 / 0.559 |
| 8/28 | 60 | 24.856 | 0.461 |
| 8/32 | 62 | 30.483 | 0.376 |
| 16/8 | 52 | 17.867 | 0.647 |
| 16/16 | 58 | 16.674 | 0.691 |
| 16/24 | 62 | 16.706 | 0.687 |

**Repeat check, `8/12` vs `8/16`** (five alternating pairs): 8/12 = 16.066–16.219 ms (mean 16.120, sd 0.059); 8/16 = 16.988–17.063 ms (mean 17.018, sd 0.030). 8/12 won all five pairs by 5.2–5.9%. Against naive's 11.46 ms that is 0.711× and 0.673×. **[V]**

`8/12` was not in the analysis doc's sweep grid ({8, 16, 24}), which is why its best (16/24) missed it. Spills: none in any configuration (`LOCAL = 0`). **[V]**

### 3.3 Where naive sits: the DRAM roofline

| Quantity | Value | Tag |
|---|---|---|
| DRAM read / write per step (512³) | 1.59 GB / 0.517 GB | [V] |
| Compulsory traffic (1 × `p_cur` + interior `p_prev`, `vel2`, write `p_next`) | ≈ 1.56 GB read, 0.51 GB write | [D] |
| Naive achieved DRAM throughput | 183.9 GB/s = 95.68% of peak (Nsight) | [V] |
| Floor at peak bandwidth (2.107 GB ÷ 192.2 GB/s) | 10.96 ms | [D] |
| Naive measured | 11.46 ms, i.e. 4.5% above the floor | [V] |
| Analytic model bytes ("effective GB/s") | 14.3 GB/step → 1251 "GB/s" | [D] |
| Naive L2 sector traffic (rule output, 485.17 M sectors × 32 B) | ≈ 15.5 GB (may include stores) | [D] |

The model's byte count is ~92% of naive's L2 sector traffic. The code's "GB/s" and `BW Eff.` columns therefore describe L2 throughput, not DRAM. `BW Eff.` of 725–824% in `bench_scaling` (which divides by the 96 GB/s figure) has no physical meaning. Nsight's roofline-style rule also flags naive's headroom as small: the largest estimated speedup is 5.98% (uncoalesced access), then 4.44% (occupancy).

### 3.4 Why tiled loses: counters (512³, single Nsight launch each)

| Config | ms | Regs | Warps active % | DRAM read GB | DRAM % | Warp instr (B) | Issue-active % | Barrier stall % | Long-scoreboard % |
|---|---|---|---|---|---|---|---|---|---|
| naive | 11.46 | 40 | 85.3 | 1.59 | 95.7 | 0.686 | 36.5 | 0 | 53.6 |
| 4/8 | 19.54 | 40 | 89.7 | 2.19 | 82.4 | 1.678 | 41.9 | 51.4 | 24.9 |
| 8/12 | 16.73 | 48 | 82.8 | 1.98 | 89.0 | 1.263 | 39.5 | 47.0 | 22.6 |
| 8/16 | 16.16 | 48 | 81.6 | 1.88 | 79.5 | 1.266 | 40.0 | 44.6 | 27.7 |
| 8/20 | 19.77 | 48 | 82.4 | 1.82 | 68.6 | 1.313 | 34.8 | 45.7 | 31.5 |
| 8/24 | 20.16 | 48 | 82.6 | 1.79 | 66.0 | 1.367 | 35.3 | 44.9 | 30.5 |
| 8/28 | 23.85 | 60 | 41.4 | 1.76 | 49.9 | 1.394 | 28.7 | 39.8 | 28.6 |
| 8/32 | 30.01 | 62 | 41.5 | 1.73 | 39.0 | 1.386 | 22.5 | 43.6 | 33.4 |
| 16/24 | 15.68 | 62 | 62.0 | 1.78 | 76.5 | 1.124 | 35.3 | 33.5 | 24.3 |

DRAM writes are ≈ 0.517 GB in all rows. Percentages are single-launch values with roughly ±3 points of pass-to-pass noise on the stall reasons. **[V]**

**DRAM traffic follows a two-term model [V/D].** Reads ≈ 1.59 GB + 0.537 GB × 8/`PZ`, i.e. the compulsory reads plus a re-read of the `2R` halo planes at each Z-slab boundary.

| PZ | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|
| Model (GB) | 2.127 | 1.948 | 1.859 | 1.805 | 1.769 | 1.743 | 1.724 |
| Measured (GB) | 2.19 | 1.98 | 1.88 | 1.82 | 1.79 | 1.76 | 1.73 |

The model was fitted to five points. Values at PZ = 12, 20 and 28 were predicted before measuring (within 1.6%), and PZ = 32 was checked afterwards with the formula fixed. Traffic falls as PZ rises while time rises, so deep-PZ slowness is not a DRAM effect.

**Occupancy does not order the configurations [V].** Theoretical occupancy from block size and registers (my calculation), against measured speed:

| Theoretical occupancy | Configs | Tiled ÷ naive |
|---|---|---|
| ~94% | 4/8 | 0.58 |
| ~83% | 8/8, 8/12, 8/16, 8/20, 8/24 | 0.56–0.72 |
| 62.5% | 16/8, 16/16, 16/24 | 0.65–0.69 |
| 41.7% | 8/28, 8/32 | 0.46, 0.38 |

At `PZ ≥ 28` the register count jumps to 60–62, the block drops to one per SM, and warps-active falls to 41.4% (matching the arithmetic). That regime explains the `28` step, not the ranking elsewhere. At `4/8` occupancy (89.7%) exceeds naive's (85.3%) while tiled is 42% slower.

**Barrier stalls track block geometry [D/I].** Counting warps that contain interior threads in a `40 × (TILE_Y+8)` block:

| TILE_Y | Warps with compute work | Barrier stall % |
|---|---|---|
| 4 | 5 of 15 (33%) | 51.4 |
| 8 | 10 of 20 (50%) | 44.6–47.3 |
| 16 | 20 of 30 (67%) | 33.5 |

Halo-only warps run the loop, skip the compute and wait at the end-of-iteration barrier, which matches Nsight's rule text about diverging paths before a barrier. This explains the level across `TILE_Y`, not the dependence on `PENCIL_Z`.

**Tiling reduced L1→L2 traffic and still lost [V].** Total L2 sectors are 232.2 M (8/12) and 222.4 M (8/16), against 485.2 M for naive (2.1–2.2× fewer), while DRAM reads rose. Both tiled kernels are flagged for uncoalesced global access (23% excess sectors; loads use 22.4–23.6 of 32 bytes per sector, stores 25.4; naive: 6% excess). A 40-wide block, not a multiple of 32, with a −4 float offset is a plausible cause. **[I]**

**Instruction growth [V].** Warp instructions per warp-iteration rise steadily with PZ (119.5, 120.8, 123.2, 126.4, 127.9 for PZ = 12–28). The dead refill load (see §8) executes: 62.4 M global-load instructions at `4/8` against ~63 M predicted with it and ~47 M without, about 24–27% of the global-load instructions.

### 3.5 Stall attribution (Nsight Compute source page, SASS view)

| Kernel | Top stall-sampled instructions | Share |
|---|---|---|
| naive | one `FADD.FTZ` (first consumer of the stencil loads) | 53.5% |
| tiled 8/12 | `@P3 BRA`, then `FADD.FTZ` | 39.4%, 18.4% |
| tiled 8/16 | `BSSY`, then `FADD.FTZ` | 40.1%, 16.4% |
| tiled 8/20 | `BSSY`, then `FMUL.FTZ`, `FADD.FTZ` | 43.2%, 11.9%, 11.1% |

Stall samples attach to the waiting instruction, so a load's latency lands on its consumer. In each tiled report the top line sits 5–15 instructions after the second `BAR.SYNC`, and `BAR.SYNC` itself carries ~1%. This is consistent with the 44–47% barrier-stall metric, but the attribution is **[I]**: the CUDA-source CSV export has only `Line No, Source` columns, so source-line mapping was not obtained.

Rule-engine output (full-set reports, one launch each):

| | 8/12 | 8/16 | 8/20 |
|---|---|---|---|
| Barrier stall, cycles per warp (est. speedup) | 11.0 (14.0%) | 10.9 (20.3%) | 13.2 (34.3%) |
| Scheduler issue interval, cycles | 2.3 | 2.4 | 2.8 |
| Active / eligible warps per scheduler | 10.25 / 1.30 | 10.02 / 1.13 | 10.37 / 0.93 |
| L1TEX-scoreboard rule | not shown | not shown | 9.0 cycles (30.9%) |

The 8/12 and 8/16 outputs were cut at 40 lines, so a missing L1TEX rule there is uncertain. All three flag uncoalesced access (est. ≥ 21%), and non-fused FP32 (est. 6–7%, not a limiter).

### 3.6 The unexplained steps

`8/16 → 8/20`: **+18%** in the 200-step bench, +20.2% in elapsed cycles (40.57 M → 48.77 M), +16.3% in a back-to-back five-launch check.

| Candidate explanation | Status at 16→20 |
|---|---|
| Registers / occupancy | Excluded (48 regs; ~10 active warps/scheduler in both) |
| Instruction count | Excluded (+3.7%) |
| DRAM bytes | Excluded (−3%) |
| Register spills | Excluded (`LOCAL = 0`) |
| Prologue latency | Excluded (samples before the first barrier −5%: 129 k → 123 k) |
| Load-batching code shape | Excluded (identical loop shape at 16, 20, 24) |
| L1 capacity | Not supported (hit rate 28.8 → 28.1 → 27.0 → 26.1 over PZ 12/16/20/24: smooth) |
| Barrier stall % | Flat within noise (45–47%) |
| Where the added time lands | Inside the loop: +146 k samples, of which the end-of-iteration line +87 k and the `FADD`/`FMUL` chain +67 k |

The five-launch check (n = 5 each): time 16.42 → 19.09 ms (Welch t = 6.4); issue-active 37.3 → 34.0% (t = −3.5); eligible warps 0.988 → 0.878 (t = −3.7). The ranges barely separate (eligible 0.93 vs 0.92; issue 35.61 vs 35.51). **Issue-active and eligible warps are not independent explanations.** With instruction count fixed, both are the same fact as "took more cycles" (correlation of time with issue-active inside `8/16`: −0.94). The informative question is why warps stall, and that remains open.

`28 → 32` (+23–26%): registers 60 → 62, occupancy identical (41.5%), instructions flat (1.394 → 1.386 B), no spills.

**All-stall-reasons check** (`ncu --clock-control none`, `-k kernel_tiled`, 5 launches averaged per
PZ after 2 skipped, 512^3, `TILE_Y=8`; stall values are % of warp-active cycles; DRAM and SM
clocks not recorded):

| PZ | ms | inst (B) | issue-active % | barrier | long_sb | short_sb | math | not_selected | no_inst | branch_res | imc_miss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 15.35 | 1.263 | 43.0 | 51.8 | 23.6 | 2.2 | 5.8 | 8.5 | 0.65 | 0.87 | 0.00 |
| 16 | 15.63 | 1.266 | 41.8 | 48.7 | 28.7 | 0.1 | 4.1 | 7.3 | 0.62 | 1.31 | 0.00 |
| 20 | 18.88 | 1.313 | 35.5 | 49.5 | 31.3 | 1.9 | 3.7 | 5.7 | 0.40 | 0.99 | 0.00 |
| 24 | 19.32 | 1.367 | 36.1 | 48.4 | 30.2 | 3.6 | 3.7 | 6.4 | 0.43 | 0.97 | 0.00 |
| 28 | 23.92 | 1.394 | 28.9 | 41.5 | 28.2 | 2.7 | 5.8 | 7.0 | 0.65 | 1.55 | 0.00 |
| 32 | 30.04 | 1.386 | 22.4 | 44.4 | 33.0 | 3.9 | 2.5 | 5.0 | 0.45 | 1.15 | 0.00 |

- **16 -> 20:** no stall reason moves by 3 points. The largest is long_scoreboard (+2.6), then
  short_scoreboard (+1.7). The prediction that no_instruction or branch_resolving would rise
  by 3 points failed (both fell). **[V]**
- **Instruction cache excluded as a cause of the 16 -> 20 step:** imc_miss is 0.00 at every PZ and
  no_instruction stays at or below 0.65%. **[D]**
- Multiplying share by time, long_scoreboard cycles grow about 32% against about 23% for
  barrier and 21% for time; this assumes similar active warps, which was not checked for
  these launches. **[I]**
- **28 -> 32:** instructions and occupancy are flat, time rises about 26%, and long_scoreboard
  rises 4.75 points (barrier +2.9, math -3.3). The only mover above 3 points; mechanism
  untested. **[I]**

**Source-line attribution for 16 -> 20** (`ncu -i ... --page source --print-source cuda,sass --csv`,
reports `tiled_y8z16` and `tiled_y8z20`; samples summed from SASS rows, which equal the
line-row totals):

| Line | Source | PZ=16 | PZ=20 | Delta | Dominant stall |
|---|---|---|---|---|---|
| L210 | `+ dt2 * vel2[c] * (lap_x + lap_y + lap_z);` | 22,918 | 115,054 | +92,136 | long_sb |
| L225 | end-of-iteration `__syncthreads` / refill condition | 320,159 | 401,076 | +80,917 | barrier |
| L159 | `for (int pz = 0; pz < PENCIL_Z; ++pz)` | 5,931 | 71,107 | +65,176 | long_sb |
| L209 | `- p_prev[c]` | 129,861 | 106,466 | -23,395 | long_sb |
| L160 | `const int iz = iz0 + pz;` | 73,541 | 0 | -73,541 | long_sb |
| total | | 781,692 | 928,047 | +146,355 (+18.7%) | |

- The pre-fill and refill lines (L151, L152, L155) change by about 1-2k samples each. **[V]**
- Samples move between neighbouring lines (L209 to L210, L160 to L159), so single lines
  mislead: L209+L210 grows by about 69k, and L159+L160 is roughly flat. This is probably the
  compiler reordering instructions across source lines. **[I]**
- Reading: the added time sits in long-scoreboard waits on the loads feeding the
  time-integration expression (`p_prev`, `vel2`) and at the end-of-iteration barrier, not in
  the pencil refill. **[I]** Consistent with R12 (loads issued late); not tested by changing
  the kernel.

**Kernel edits tried afterwards** (`TILE_Y=8, PENCIL_Z=12`, 512^3, 5 alternating pairs of
`bench_tiled`, 200 steps, 20 warmup; baseline about 16.03 ms):

| Edit | Registers | Gate (85 tests) | Tiled ms (mean) | Change |
|---|---|---|---|---|
| Use `pencil[R]` instead of reloading `p_cur[c]` (commit ea0662e) | 48 -> 48 | pass | 16.032 | -0.02% (noise) |
| Prefetch `p_prev`/`vel2` before the first barrier (reverted) | 48 -> 48 | pass | 16.825 | +5.0% slower, all 5 pairs |

- The reload removal is correct and neutral. **[V]**
- Moving the `p_prev`/`vel2` loads ahead of the barrier did not help and cost about 5%, so
  this form of prefetch does not recover the PZ 16->20 step or the tiled slowdown, which
  weakens (but does not exclude) the late-load explanation: the edit also added a guarded
  branch, so load placement was not isolated. **[V]** for the timings, **[I]** for the reading.
  Why it is slower (extra guarded branch and live ranges for halo-only threads) is **[I]**;
  the SASS was not inspected.
- The PZ steps remain unexplained.

The mechanism behind both steps remains unresolved.

### 3.7 Independent bandwidth test (`tests/perf/bench_stream.cu`)

Standalone STREAM-style kernels (grid-stride, `float4`, 2 × 512 MiB buffers, about
21× the 24 MiB L2; 100 timed reps after 10 warmup; three separate process launches).
GB = 1e9 bytes, peak = 192.0 GB/s (derived).

| Kernel | Best GB/s | Median GB/s | Best % of peak |
|---|---|---|---|
| read (B bytes) | 187.7–187.8 | 187.6 | 97.8% |
| write (B bytes) | 184.1–184.3 | 183.1 | 95.9–96.0% |
| copy (2B bytes) | 174.1–174.3 | 169.9–172.4 | 90.6–90.8% |

Spread across the three launches is under 1% for read and write and about 1–2% for
copy. An earlier run with 50 reps and 5 warmup gave a copy median of 151 GB/s, so
short warmup matters here as well. **[V]**

- **Peak:** a read-only kernel reaches about 98% of the derived peak, so at least
  about 188 GB/s is achievable. This corroborates the derived 192.0 GB/s but does
  not measure the ceiling. **[V]**
- **Mixing cost:** copy (6.16 ms) is about 6.6% slower than read plus write run
  separately (2.86 + 2.92 = 5.78 ms). **[D]**
- **Floor for naive:** with 1.59 GB read and 0.517 GB written at the measured
  read and write rates, the additive floor is about 11.27 ms; naive's 11.46 ms is
  about 1.7% above it. Naive is about 75% reads, so the copy rate is not its bound.
  **[D]**

**Nsight cross-check** (`ncu --clock-control none`, one launch each, `--warmup 0`):

| Launch | DRAM read | DRAM write | Duration | bytes÷time | Reported `dram__throughput` |
|---|---|---|---|---|---|
| read | 545 MB | 1 MB | 2.92 ms | about 187 GB/s (97.5%) | 97.67% |
| write | 8.7 MB | 524 MB | 3.50 ms | about 152 GB/s (79.3%) | 90.69% |
| copy | 537 MB | 526 MB | 7.03 ms | about 151 GB/s (78.8%) | 90.14% |

DRAM byte counts match the expected 537 MB per buffer (write and copy are about
10 MB low, plausibly dirty lines still in L2 at kernel end, which is **[I]**).
The percentage agrees with bytes÷time÷192 for the read launch (within 1 point)
but is about 11 points higher for both write-containing launches; the implied
"peak" is about 168 GB/s in both. That cannot be a real limit, since the
steady-state write kernel reaches 184 GB/s, and the single ncu launches are also
slower than steady state (write 3.50 vs 2.92 ms, copy 7.03 vs 6.16 ms). The cause is a lowered DRAM clock during those launches (resolution below).

**Resolution.** Re-running with `dram__cycles_elapsed.avg.per_second` recorded
(`--clock-control none`, one launch each, cold = first launch, warm = after 5 warmups):

| Launch | DRAM clock | bytes÷time | Reported % | Implied peak |
|---|---|---|---|---|
| read, cold, `--cache-control all` | 7.99 GHz | 187.8 GB/s | 98.05 | 191.5 |
| read, cold, `none` | 7.70 | 169.2 | 91.61 | 184.7 |
| write, cold, `all` | 7.99 | 181.3 | 94.53 | 191.8 |
| write, cold, `none` | 6.94 | 146.8 | 88.10 | 166.7 |
| copy, cold, `all` | 7.99 | 172.9 | 90.20 | 191.7 |
| copy, cold, `none` | 7.84 | 162.5 | 86.34 | 188.2 |

All six warm launches ran at 7.99 GHz and at steady-state times. In every launch the
implied peak equals 192.0 × (clock ÷ 7.99 GHz) to within about 0.4 GB/s, so
`dram__throughput` % is normalised to the DRAM clock during the launch, not to a fixed
192.0 GB/s. At full clock it agrees with bytes÷time÷192 within about 0.4 points for
read, write and copy. The earlier 11-point gap (write 3.50 ms, copy 7.03 ms) corresponds
to a clock of about 7.0 GHz; the clock was not recorded for those launches, so that is
**[I]**. Dips appeared in the `--cache-control none` cold launches and in the earlier
default-cache-control launches, never in warm ones; why they occur only sometimes is
**[U]** (a memory-clock ramp after idle is plausible but untested). Use bytes÷time, or
record the DRAM clock alongside the percentage. **[V]** for the table, **[I]** for the
mechanism. Other cases where a DRAM-% gap was seen (§5.3, locked vs unlocked clocks) were
not re-examined with the clock recorded until the check below.

**Clock-recorded compare runs** (`scripts/profile_nsight.sh compare`, 512^3, one launch each,
`--steps 1 --warmup 0`; `CLOCK_CONTROL` = none or base):

| Kernel, mode | DRAM clock | Reported DRAM % | Implied GB/s |
|---|---|---|---|
| naive, none | 6.99 GHz | 96.4 | ~162 |
| naive, base | 6.99 | 95.8 | ~161 |
| tiled (8/12), none | 7.99 | 86.4 | ~166 |
| tiled (8/12), base | 7.99 | 74.2 | ~142 |

- Implied GB/s = reported % x 192.0 x (clock / 7.99 GHz). **[D]**
- The wrapper's launches can run at a lowered DRAM clock (6.99 GHz seen for naive at 512^3 and for
  both kernels at 255^3, in repeated runs and under both clock-control modes), so its DRAM %
  columns are not comparable between kernels unless the clock row matches. **[V]** for the
  clocks; the cause (a memory-clock ramp after a short or cold run) is **[U]**.
- Naive's implied time at the lowered clock is about 13 ms against 11.46 ms steady state. **[D]**
- Tiled's DRAM % drops from 86.4 to 74.2 under base SM clocks with the memory clock unchanged,
  while naive is unchanged: consistent with tiled being SM-bound, not tested directly. **[I]**
- The earlier locked-vs-unlocked DRAM % pair (79.5 vs ~89) came from runs without the clock
  recorded, so it is not re-derived here.

**Correction: the dip is intermittent.** Repeating the 512^3 unlocked compare three more times
gave DRAM clocks (naive / tiled) of 6.99 / 7.47, 7.99 / 6.99 and 7.99 / 7.99 GHz; an earlier
run had 7.99 / 7.99 and another 6.99 / 7.99. So the table above is one sample, not a steady
state: any kernel can land at a lowered clock in any launch, and the cause stays **[U]**.
`scripts/profile_nsight.sh compare` now prints a note when the two clocks differ by more
than 2% (seen firing on two of the three repeats). With matched clocks (7.99 / 7.99) the
compare table read naive 96.2% and tiled 85.9% (about 185 and 165 GB/s, **[D]**). **[V]**
for the clocks, **[I]** for the SM-bound reading of tiled's base-clock drop.

**Why the DRAM clock dips.** Single-launch `ncu` runs land at a lowered DRAM clock
intermittently (6.99 to 7.47 GHz against 7.99 GHz), for either kernel and under both
clock-control modes. Sampling `clocks.mem` every ~20 ms during a compare run showed the
memory clock leaving idle (405 MHz, P8) through 7001 MHz (P3) to 8001 MHz (P0), then
returning to 7001 MHz and back to 8001 MHz while the GPU stayed busy. 7001 MHz
corresponds to 168 GB/s, the implied peak seen in the lowered-clock launches. **[V]**
that the dips are real memory-clock states, not an accounting artifact. All six warm
`bench_stream` launches ran at 7.99 GHz, but running `bench_stream` (about 2 s of load)
immediately before the wrapper did not remove the dips: both kernels still read 6.99 GHz.
Whether ncu's startup or replay passes lower the clock, or the clock fluctuates under
short bursts, is **[U]**. The sampler is not synchronised with launches, so it cannot
say which launch ran at which clock. Practical rule: quote bytes/time, or the DRAM %
together with the clock row.

Reproduce: `./build/bin/bench/bench_stream --warmup 10 --reps 100`; cross-check
commands are in the header comment of `tests/perf/bench_stream.cu`.

---

## 4. Prediction ledger

Predictions written before the run, selected from this session (not exhaustive). **24 held, 15 failed.**

| # | Prediction | Outcome | |
|---|---|---|---|
| 1 | Step 3 banner: Tile Y 4, Pencil Z 8, archs 89, NVTX on | Matched | ✓ |
| 2 | Build takes 3–8 min (guide) | 4.3 s wall, 33 s CPU | ✗ |
| 3 | Naive 40 regs; tiled at 4/8 below the 62 seen at 16/24; no spills | 40 / 40 / `LOCAL 0` | ✓ |
| 4 | Static `LDG` at 4/8 ≈ 19, plus a surviving refill | 20 | ✓ |
| 5 | 59/59 unit, 15/15 GPU tests, GPU IDs 60–74 | All; last ID `#74` | ✓ |
| 6 | 5B: naive ≈ 11.5, tiled ≈ 19.9 ms, 0.58× | 11.467 / 19.907 / 0.58× | ✓ |
| 7 | 5A at 255³ ≈ 1.4 ms (±15%) | 1.572 ms (+12%, edge of tolerance) | ✓ |
| 8 | Alignment explains the 5A gap | 256³ cold 1.578 ms; steady state 1.40 with long warmup | ✗ |
| 9 | Memory clock at 8001 MHz in both timed loops | 209/209 active samples | ✓ |
| 10 | Tiled-phase SM clock ≥ 5% below naive-phase | 1.7% lower | ✗ |
| 11 | 6B: tiled slower at all sizes, 0.4–0.75× | 0.42–0.68× | ✓ |
| 12 | 6C `BW Eff.` ≈ 750% at 512³ | 749.2% | ✓ |
| 13 | Section 7 average 0.59× | 0.59× | ✓ |
| 14 | Roofline points sit above the roof | Yes (plot) | ✓ |
| 15 | `scaling.png` link broken; `plot_scaling` never saves | Confirmed | ✓ |
| 16 | `RmProfilingAdminOnly = 1`, `ncu` needs a fix | 0; `ncu` worked | ✗ |
| 17 | Sweep speedups within 0.03 of analysis doc §4.2 | Within 0.005 | ✓ |
| 18 | 8/24 worst by occupancy (assumed 62 regs); 4/8 best | 8/24 has 48 regs; 4/8 is slowest | ✗ |
| 19 | Regs 40/40/48/62; naive DRAM 90–97% | 95.68% | ✓ |
| 20 | Dead refill load executes (62.4 M vs ~63 M) | 62.4 M | ✓ |
| 21 | 8/24 stalls dominated by long-scoreboard | Barrier-dominated (44.9%) | ✗ |
| 22 | `nsys`: 120 launches, 4 stream syncs, 2 event syncs | Exact | ✓ |
| 23 | `nsys` naive mean ≈ 11.5 ms | 12.19 (median 11.84; includes cold launches) | ✗ |
| 24 | Registers 47–50 at every PZ (TILE_Y=8) | 60, 62 at PZ 28, 32 | ✗ |
| 25 | PZ 12/16/20 at 16.8–17.3 ms | 16.0 / 16.9 / 20.0 | ✗ |
| 26 | DRAM reads at 8/12, 8/20, 8/28: 1.95 / 1.81 / 1.74 GB | 1.98 / 1.82 / 1.76 | ✓ |
| 27 | Regs 48/48/60; max warps 83.3/83.3/41.7% | Exact | ✓ |
| 28 | Warp instr at PZ 20 ≈ 1.32 B | 1.3125 B | ✓ |
| 29 | Static `LDG` = PZ + 12 | 24 / 28 / 32 / 36 | ✓ |
| 30 | 8/12 beats 8/16 in ≥ 4/5 pairs by ≥ 3% | 5/5, 5.2–5.9% | ✓ |
| 31 | 8/32: 62 regs, 41.67%, ~1.41 B, issue < 28.7% | 62, 41.67%, 1.386 B, 22.45% | ✓ |
| 32 | 8D: top-3 stall lines are BARs plus the `STS` | Branch/`BSSY` lines and `FADD`/`FMUL` | ✗ |
| 33 | 8D: `BAR` opcodes ≥ 40% of samples | ~1% | ✗ |
| 34 | 8D: naive `LDG` ≥ 50% of samples | 24.3% (one `FADD` holds 53.5%) | ✗ |
| 35 | 8D: 8/20 pre-first-BAR share ≥ 3 points above 8/16 | −3.3 points | ✗ |
| 36 | 8D: rule engine flags barrier/occupancy stalls | Barrier rule fires | ✓ |
| 37 | L1 hit rate drops ≥ 1.5 points 16→20 | −1.07, smooth | ✗ |
| 38 | Five-launch: 8/20 eligible ≥ 0.1 lower; issue lower in most pairs | −0.110; 5/5 pairs | ✓ |
| 39 | 8/24 has ~8 more static `MOV` than 8/16 | Plain `MOV` 34 vs 39 (`IMAD.MOV` rose 59 → 95; see §5.3) | ✗ |

---

## 5. Corrections

### 5.1 To `PERFORMANCE_ANALYSIS.md` and repo docs

| Statement | Finding |
|---|---|
| Register/occupancy chain is "root-caused" for the tiled slowdown | Confirmed at 16/24 (62 regs, one 960-thread block/SM, 62.5% theoretical vs 61.96% measured). It does **not** generalize: see the occupancy table in §3.4. At 4/8 occupancy exceeds naive's yet tiled is 0.58×. |
| Best tiled config is 16/24 (~0.69×) | 8/12 is faster (~0.71×); the sweep grid skipped `PZ = 12` |
| Peak DRAM ≈ 96 GB/s | ≈ 192 GB/s; naive is at 95.7% of it |
| README / `nsight_workflow.md`: naive reads ~14.7 GB/step from DRAM; tiled ~0.54 GB (27× less); ~3.5× speedup | Not reproduced here: naive reads 1.59 GB and tiled reads more. These are A100 projections, but the DRAM-byte claim ("every neighbour fetched from DRAM") does not hold on this GPU. Memory-throughput expectation (~45% naive) was 95.6%. |
| Appendix A.4 (inverted average speedup) | Fixed in commit `7d97de8`; script now reports 0.59× |

### 5.2 To the guide (`BUILD_AND_RUN.md`)

| Location | Issue |
|---|---|
| Step 1 | Clone URL is a placeholder (`YOUR_USERNAME/stencil-solver`); the repo is `somya2703/stencil_solver` |
| Prerequisites | Expected `nvidia-smi` header shows `CUDA Version: 12.x`; current drivers report 13.x (nit) |
| Step 3 vs §8B | Step 3 prescribes `TILE_Y=4, PENCIL_Z=8` as the tuned setting; §8B says `TILE_Y=4` measures worse than the default. Measured here: 0.58×, the slowest of the valid configs. |
| Step 3 / §5B / §6B | Text attributing "tiled slower" to mistuned defaults is refuted at the guide's own recommended config |
| Step 4 | "3–8 minutes" build: 4.3 s wall observed |
| Section 4 | GPU test IDs `44–58` should be `60–74` with 59 unit tests; expected "30–60 sec" was 2.06 s |
| §5A/§5B/§6A/§6B/§6C | Expected outputs (82.4 ms naive; tiled 3.16× or 2.46–3.29× faster; `BW Eff.` 15–30%) contradict measurement (11.46 ms; 0.42–0.68×; 725–824%). The guide's error factor varies with grid size (27× at 64³, 7× at 512³), so the table is not a rescaled version of reality. |
| §5A | `--nx 256` alone triggers a 128/256/512 sweep. The guide's 255³ workaround measures the short-warmup regime (up to 12% slower than steady state). |
| §5A | Memory lines `66.30 / 265.19 MB` should be `66.33 / 265.30 MB` |
| §8A | `perf_event_paranoid` remedy was not needed (`ncu` worked, `RmProfilingAdminOnly = 0`) |
| §8D | The baseline-tooltip wording was not found in `ncu-ui` 2025.1.1; baselines are available via the Baselines panel |
| §8D "what to look for" | The occupancy-cap explanation is not supported at 8/12 (occupancy −3%), and its "GB/s exceeds 96 GB/s peak" premise rests on the wrong peak |

### 5.3 Interpretation revisions made during this investigation

Recorded so the trail is auditable:

- Early guide comparisons quoted an older inline copy of the guide; they were redone against the uploaded file (test count, expected outputs).
- The 255³ "alignment" hypothesis was wrong; warmup length explains the gap.
- The `RmProfilingAdminOnly` prediction was wrong, and my first grep used the wrong key.
- My SASS histogram split mnemonics at "." and folded `IMAD.MOV.U32` (register moves) into `IMAD`. The MOV-shift test was therefore not valid. Corrected counts: `IMAD.MOV` 55 / 59 / 79 / 95 for PZ 12 / 16 / 20 / 24, and the +66 `IMAD` growth from 16 to 24 splits into +36 moves and +30 other.
- A lead that "ncu's DRAM percent minus bytes÷time÷192 gap tracks slowness" was refuted by 8/12 (fast, gap 11.3 points) and 8/28 (slow, gap 0.2).
- "8/12 is best" was held as unconfirmed after one ncu launch flipped its sign, then confirmed by the 5×5 repeat.
- Issue-active and eligible-warps "signatures" at the 16→20 step were restatements of the slowdown, not causes (§3.6).
- The 8D stall predictions confused where samples land (§3.5).

---

## 6. Repository, tooling and reporting findings

| ID | Finding | Evidence | Tag |
|---|---|---|---|
| R1 | Peak-bandwidth formula lacks the DDR ×2 factor (`bench_naive`, `main_naive`, `bench_scaling`); prints 96 instead of ~192 GB/s | Nsight DRAM % vs bytes÷time | V |
| R2 | "Effective GB/s" and `BW Eff.` are model numbers (≈ L2 traffic), not DRAM; the roofline plot draws points 7–17× above the memory roof | §3.3; plot screenshot | V |
| R2 fix | L1 (b67dca6): bandwidth uses compulsory bytes, roofline AI is 3.5 FLOP/B; at 512^3 naive reads about 178 GB/s and `BW Eff.` for tiled about 67% | bench output | V |
| R3 | `format_perf_table(results, 0)` is called on a list of different grid sizes (`bench_tiled`, `main_tiled`, `main_naive` sweep), so the Speedup column compares every size against 64³ (0.06×, 0.02×, 0.00×) and rows carry no grid label | Program output | V |
| R4 | Benchmark JSON records GPU, CUDA version, radius and precision but not `TILE_Y`, `PENCIL_Z`, steps or warmup, so results cannot be attributed to a config | `head -12 results/bench_tiled.json` | V |
| R5 | `plot_scaling` never calls `savefig`; the report hardcodes a link to `plots/scaling.png`, which is never created | `grep savefig`; `ls docs/plots` | V |
| R6 | `ny`/`nz` are overwritten by `nx` in every main, so non-cubic grids cannot be run. Sweep trigger differs: `main_naive` sweeps only when all of nx/ny/nz are 256, the other binaries when `nx == 256`. | Output of `--nx 256` vs `--nx 256 --ny 256 --nz 257` | V |
| R7 | `clang-format` and `clang-tidy` have no leading dot, so the tools never auto-discover them | `ls -a`, `git ls-files` | V |
| R8 | Workflows sit in `infra/.github/workflows/`; GitHub only reads root `.github/workflows/` | `git ls-files` | V |
| R9 | `CMakeLists.txt.patch` is still tracked (an earlier commit claims leftovers were removed); `PERFORMANCE_ANALYSIS.md`, `tuning_results/` and the guide copy are untracked | `git status` | V |
| R10 | Root-owned `build-tuned/` and `docs/plots/` (a container wrote into the bind-mounted tree). `plot_results.py` failed with `PermissionError` until `docs/plots` was re-owned. | `find -user root`; traceback | V |
| R11 | `nvtx3/nvtx3.hpp` was found on CUDA 12.8; the missing-header problem (A.6) is toolkit-dependent | Clean configure and build | V |
| R12 | The kernel reloads `p_cur[c]` from global although `pencil[R]` holds the same value; `p_prev`/`vel2` loads are issued after the first barrier and consumed at the end of the iteration | Source; static LDG count | I |

---

## 7. Suspected issues in the multi-GPU path (not evaluated)

These come from reading a snapshot of `main_multigpu.cpp`, `nccl_exchange.cpp` and `merge_scaling.py`. **Nothing was run and none was checked against HEAD.** [U]

1. Every rank calls `cudaSetDevice(g_rank % g_nranks)`, which equals `rank`; ranks beyond the node's GPU count would request nonexistent devices, and several ranks cannot share one GPU.
2. Each rank builds its own Gaussian at the centre of its local slab, so N ranks produce N pulses instead of one global pulse.
3. No halo exchange occurs before the first step.
4. `ncclCommInitAll` is a single-process API; under `mpirun` it would create a private clique per process (`ncclGetUniqueId` + `ncclCommInitRank` is the multi-process route).
5. `merge_scaling.py`'s docstring claims the strong-scaling formula but the code computes `t1/tN` for both modes.
6. `run_scaling.sh` ends by suggesting `--scaling <directory>`, while `plot_results.py` needs a JSON file.

Each can be checked with a `grep` or a short run; see the reproduction appendix.

---

## 8. Hypotheses for future work (untested)

**Kernel-level candidates.** Nsight's rule engine gives upper bounds that are not additive.

| Idea | Basis |
|---|---|
| Set defaults to `TILE_Y=8, PENCIL_Z=12` | Measured best (§3.2) |
| Use `pencil[R]` instead of reloading `p_cur[c]` | R12 |
| Prefetch `p_prev`/`vel2` for the next iteration before the first barrier | R12; barrier and long-scoreboard stalls |
| Remove the dead refill load (`pencil[PZ+2R-1]`, never read) | 24–27% of global-load instructions; small effect expected |
| Make the block width a multiple of 32 and shrink halo-only warps | Uncoalescing rule (≥ 21%), barrier rule (14–34%), warp-geometry table |

**Beating the floor requires fewer bytes, not more reuse.** Ideas, none tested: skip the `vel2` read for homogeneous media (−0.51 GB/step, ≈ 8.3 ms floor at 192.2 GB/s, arithmetic only [D]), reduced-precision field storage, temporal blocking. Each changes the algorithm or accuracy and would need its own validation.

---

## 9. Limitations

- One GPU, one laptop chassis. Power and thermal behavior (mean 1.86–1.89 GHz sustained vs 2.6 GHz single-launch) may differ elsewhere.
- Toolchain differs from earlier data: CUDA 12.8 here, 12.4 in the repo's earlier benchmark JSON. Results matched the analysis doc within 0.5%, but the difference is not zero.
- The DRAM peak is derived from a clock and a bus width implied by the code's own printed value. A standalone read-only kernel reaches about 188 GB/s (about 98% of the derived 192.0 GB/s), which shows at least that much is achievable; it does not measure the true ceiling. `dram__throughput` % matches bytes÷time÷192 for read-dominated launches but sits about 11 points above it for write-only and copy launches (§3.7); that gap reflects a lowered DRAM clock during those launches, which the
percentage is normalised to.
- The DRAM-traffic model was fitted post hoc; only three of its points were predicted in advance.
- Repeat counts are small (n = 5) and only for `8/12` vs `8/16`. Other sweep cells are single runs (1% run-to-run spread, but no confidence intervals).
- Stall attribution stops at the SASS level. Source-line mapping was not obtained.
- The mechanism behind the 16→20 and 28→32 steps is unresolved.
- The barrier explanation rests on block geometry and rule-engine text (§3.4–§3.5). It was not tested by changing the kernel.

**Not evaluated:** Section 9 (multi-GPU/MPI/NCCL) and Section 10 (CI in Docker). The MPI and NCCL paths and all multi-GPU scaling claims are untested on this hardware.

---

## Appendix A. Reproduction

```bash
# Build (guide Step 3), then tests
mv build build-prev
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DSTENCIL_BUILD_TESTS=ON -DSTENCIL_BUILD_BENCHMARKS=ON \
      -DSTENCIL_ENABLE_NVTX=ON -DCMAKE_CUDA_ARCHITECTURES="89" -DSTENCIL_TILE_Y=4 -DSTENCIL_PENCIL_Z=8
cmake --build build --parallel $(nproc)
ctest --test-dir build --label-regex unit --parallel $(nproc)
ctest --test-dir build --label-regex gpu --timeout 300

# Benchmarks (200 steps, 20 warmup)
./build/bin/bench/bench_naive   --steps 200 --warmup 20 --output results/bench_naive.json
./build/bin/bench/bench_tiled   --steps 200 --warmup 20 --output results/bench_tiled.json
./build/bin/stencil_tiled --nx 512 --steps 200 --warmup 20

# Tile sweep for one config (repeat for each TILE_Y / PENCIL_Z)
cmake -B build-tune -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89" \
      -DSTENCIL_TILE_Y=8 -DSTENCIL_PENCIL_Z=12 -DSTENCIL_BUILD_TESTS=OFF -DSTENCIL_BUILD_BENCHMARKS=OFF
cmake --build build-tune --parallel $(nproc) --target stencil_tiled
cuobjdump --dump-resource-usage build-tune/bin/stencil_tiled | grep -A1 kernel_tiled

# Counters (512^3, one launch)
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_shared_ld.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
ncu --clock-control none -k kernel_tiled --launch-count 1 --metrics $M ./build/bin/stencil_tiled --nx 512 --steps 1 --warmup 0

# Full reports for the GUI / rule engine
ncu --set full --import-source yes --clock-control none -f -k kernel_tiled --launch-count 1 \
    -o nsight-output/8d/tiled_y8z16 ./build-t8z16/bin/stencil_tiled --nx 512 --steps 1 --warmup 0
ncu -i nsight-output/8d/tiled_y8z16.ncu-rep --page source --print-source sass --csv > tiled_y8z16.sass.csv
ncu -i nsight-output/8d/tiled_y8z16.ncu-rep --page details | grep -A4 -E "^\s+(OPT|WRN)"

# Clock/power log during a run
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory,pstate \
    --format=csv,noheader,nounits -lms 100 > gpu-clocks.csv &

# Checks for the unverified items in section 7 (run at HEAD)
grep -n "cudaSetDevice" src/main_multigpu.cpp
grep -n "init_gaussian" -A2 src/main_multigpu.cpp
grep -n "ncclCommInitAll\|ncclGetUniqueId" src/comm/nccl_exchange.cpp
grep -n "efficiency" scripts/merge_scaling.py
```

Stall-attribution script used for the SASS export (`/tmp/stall_top.py`): read the CSV, find the header row containing a `Source` column and the column whose name contains "stall" and "all", print the top-N lines by samples, aggregate by opcode, and report the share before the first `BAR`.

## Appendix B. Artifacts produced

`build-step4.log`, `test-unit.log`, `test-gpu.log`, `run-5a.log`, `run-5b.log`, `run-6a.log`, `run-6b.log`, `run-6c.log`, `gpu-clocks-sec5.csv`, `gpu-clocks-phases.csv`, `results/*.json`, `tuning_results/sweep.csv`, `tuning_results/sweep_y8.csv`, `nsight-output/ncu-*.log`, `nsight-output/nsys-run.log`, `nsight-output/8d/*.ncu-rep`, `nsight-output/8d/*.sass.csv`, `docs/plots/*.png`.
