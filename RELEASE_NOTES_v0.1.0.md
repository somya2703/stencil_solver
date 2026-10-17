## stencil-solver v0.1.0

A GPU 3D wave/heat simulator built to test a famous GPU speed-up trick (shared-memory tiling)
and document, with hardware counters, what actually happens.

### Headline result (RTX 4050 Laptop GPU, Ada SM 8.9, CUDA 12.8)

**The textbook optimisation does not win on this GPU.** At 512³ the naive kernel takes about
11.5 ms per step. The first tiled kernel (v1) took about 16.0 ms (0.72x of naive). After a
redesign (v2: 32 x 8 blocks where every thread owns an output column) it takes about 13.1 ms
(0.875x, sd 0.001 over 10 back-to-back pairs). Naive is already within about 4.5% of the memory-bandwidth floor, so there is very
little left for tiling to remove.

### Why (from Nsight Compute / Systems)

- Naive reads 1.59 GB per step and reaches about 96% of peak DRAM throughput.
- Tiled v1 read *more* DRAM (1.98 GB) and spent 33-51% of warp cycles waiting at barriers;
  v2 cut that to about 20% and recovered roughly half of the gap.
- Peak bandwidth was cross-checked with an independent STREAM-style kernel (about 188 GB/s
  read, 98% of the derived 192 GB/s).

### What is in the release

- Naive and tiled (v2) CUDA kernels, wave and heat solvers, 70 unit tests and 15 GPU tests
- `bench_naive`, `bench_tiled`, `bench_scaling`, `bench_stream`; plot and report scripts
- `stencil_slice` plus `scripts/make_slice_gif.py`: write the simulation out as an animation
- `docs/CASE_STUDY.md`: every claim tagged verified / derived / inferred / unverified, with a
  ledger of 39 written-down predictions (24 held, 15 failed)
- `BUILD_AND_RUN.md`: step-by-step guide with measured output for every command

### Limits (stated up front)

- One GPU, one laptop. The tiled design targeted A100-class hardware and has not been run there.
- The multi-GPU (MPI/NCCL) path is implemented but has **not** been run or validated.
- GPU tests, benchmarks and profiling run locally; CI covers the format check and the
  CPU-fallback build with unit tests.
