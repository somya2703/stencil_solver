/**
 * src/main_multigpu.cpp
 *
 * Multi-GPU stencil solver entry point.
 *
 * Parallelisation strategy: Z-slab domain decomposition.
 *   - MPI rank r owns global Z slice [r*Nz_local, (r+1)*Nz_local)
 *   - Each rank runs the tiled kernel on its local slab
 *   - After each step, Z-face halos are exchanged via IHaloExchange
 *     (NCCL for single-node, MPI_Sendrecv for multi-node)
 *
 * Launch:
 *   Single node (4 GPUs):
 *     mpirun -n 4 ./stencil_multigpu --nx 512 --ny 512 --nz 512 --steps 200
 *
 *   Multi-node (2 nodes × 4 GPUs):
 *     mpirun -n 8 --npernode 4 ./stencil_multigpu --nx 512 --ny 512 --nz 1024
 *
 * Scaling output (appended to --output JSON):
 *   Each rank writes its own timing; rank 0 aggregates and writes the
 *   ScalingResult JSON consumed by plot_results.py.
 *
 * Weak scaling:  keep Nz_local constant, increase nranks → expect ~constant ms/step
 * Strong scaling: keep total NZ constant, increase nranks → expect ~linear speedup
 */

#include "stencil/comm.hpp"
#include "stencil/config.hpp"
#include "stencil/grid.hpp"
#include "stencil/io.hpp"
#include "stencil/kernels.cuh"
#include "stencil/timer.hpp"
#include "stencil/types.hpp"

#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>
#endif

#ifdef ENABLE_MPI
#  include <mpi.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// ── MPI helpers ───────────────────────────────────────────────────────────────
static int  g_rank   = 0;
static int  g_nranks = 1;

static void mpi_init(int& argc, char**& argv) {
#ifdef ENABLE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);
#else
    (void)argc; (void)argv;
#endif
}

static void mpi_finalize() {
#ifdef ENABLE_MPI
    MPI_Finalize();
#endif
}

static double mpi_allreduce_max(double v) {
#ifdef ENABLE_MPI
    double result = 0.0;
    MPI_Allreduce(&v, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
#else
    return v;
#endif
}

static double mpi_allreduce_sum(double v) {
#ifdef ENABLE_MPI
    double result = 0.0;
    MPI_Allreduce(&v, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return result;
#else
    return v;
#endif
}

// ── Per-rank solver ───────────────────────────────────────────────────────────
/**
 * run_rank — execute the full time loop for this rank's Z-slab.
 *
 * Returns the wall time per step in milliseconds (measured at this rank).
 * Rank 0 aggregates across all ranks to report the max (critical path).
 */
static double run_rank(const stencil::Config& cfg,
                        stencil::IHaloExchange* comm) {
    using namespace stencil;

    // ── Build local slab dims ─────────────────────────────────────────────
    const GridDims global = cfg.make_grid();
    const GridDims local  = local_dims(global, g_rank, g_nranks);

#ifndef STENCIL_CPU_FALLBACK
    // Bind this rank to its GPU (assumes rank == device index)
    CUDA_CHECK(cudaSetDevice(g_rank % g_nranks));

    // Upload FD coefficients to this device's constant memory
    upload_fd_coefficients();

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
#else
    void* stream = nullptr;
    upload_fd_coefficients();
#endif

    // ── Allocate device slab ──────────────────────────────────────────────
    DeviceGrid grid(local);

    // Initial conditions: Gaussian pulse (same as single-GPU solver)
    HostGrid h_field(local);
    init_gaussian(h_field,
                  cfg.pulse_cx, cfg.pulse_cy, cfg.pulse_cz,
                  cfg.pulse_sigma);
    grid.upload_cur (h_field, stream);
    grid.upload_prev(h_field, stream);

    HostGrid h_vel(local);
    init_constant_velocity(h_vel, cfg.velocity);
    grid.upload_vel2(h_vel, stream);

#ifndef STENCIL_CPU_FALLBACK
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif

    // ── Warmup ────────────────────────────────────────────────────────────
    for (int i = 0; i < cfg.warmup_steps; ++i) {
        launch_slab(grid.cur(), grid.prev(), grid.vel2(), grid.next(),
                    local, stream);
        grid.rotate();
        if (g_nranks > 1)
            comm->exchange(grid.cur(), local, stream);
    }
#ifndef STENCIL_CPU_FALLBACK
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif

    // ── Timed loop ────────────────────────────────────────────────────────
    GpuTimer timer;
    timer.start(stream);

    for (int step = 0; step < cfg.steps; ++step) {
        NVTX_RANGE("step");

        launch_slab(grid.cur(), grid.prev(), grid.vel2(), grid.next(),
                    local, stream);
        grid.rotate();

        if (g_nranks > 1) {
            NVTX_RANGE("halo_exchange");
            comm->exchange(grid.cur(), local, stream);
        }
    }

    timer.stop(stream);
    const double ms_total = static_cast<double>(timer.elapsed_ms());

#ifndef STENCIL_CPU_FALLBACK
    CUDA_CHECK(cudaStreamDestroy(stream));
#endif

    return ms_total / static_cast<double>(cfg.steps);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    mpi_init(argc, argv);

    if (g_rank == 0)
        std::printf("=== stencil-solver: multi-GPU (%d rank%s) ===\n\n",
                    g_nranks, g_nranks > 1 ? "s" : "");

    stencil::Config cfg;
    try {
        cfg = stencil::parse_args(argc, argv);
    } catch (const std::exception& e) {
        if (g_rank == 0)
            std::fprintf(stderr,
                "Error: %s\n\n"
                "Usage: mpirun -n N ./stencil_multigpu "
                "[--nx N] [--ny N] [--nz N] [--steps N] "
                "[--warmup N] [--velocity F] [--output path.json]\n",
                e.what());
        mpi_finalize();
        return EXIT_FAILURE;
    }

    cfg.variant = stencil::KernelVariant::MultiGPU;

    // ── Print decomposition info (rank 0 only) ────────────────────────────
    if (g_rank == 0) {
        const stencil::GridDims global = cfg.make_grid();
        std::printf("Global grid : %zu × %zu × %zu\n", cfg.nx, cfg.ny, cfg.nz);
        std::printf("Ranks       : %d\n", g_nranks);
        std::printf("Local NZ    : %zu + %d halos/side\n",
                    stencil::local_nz(cfg.nz, 0, g_nranks), STENCIL_RADIUS);
        std::printf("Steps       : %d  Warmup: %d\n\n",
                    cfg.steps, cfg.warmup_steps);
        (void)global;
    }

    // ── Create halo exchange ──────────────────────────────────────────────
    std::unique_ptr<stencil::IHaloExchange> comm;
    try {
        comm = stencil::make_halo_exchange(g_rank, g_nranks, cfg);
        if (g_rank == 0)
            std::printf("Comm backend: %s\n\n", comm->name().c_str());
    } catch (const std::exception& e) {
        if (g_rank == 0)
            std::fprintf(stderr, "Comm init failed: %s\n", e.what());
        mpi_finalize();
        return EXIT_FAILURE;
    }

    // ── Run ───────────────────────────────────────────────────────────────
    double ms_per_step = 0.0;
    try {
        ms_per_step = run_rank(cfg, comm.get());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] FAILED: %s\n", g_rank, e.what());
        mpi_finalize();
        return EXIT_FAILURE;
    }

    // ── Aggregate across ranks ────────────────────────────────────────────
    // Report the max time across ranks (critical-path step time).
    const double ms_max = mpi_allreduce_max(ms_per_step);
    const double ms_sum = mpi_allreduce_sum(ms_per_step);

    if (g_rank == 0) {
        // Effective bandwidth: all ranks together process the full global grid
        const stencil::GridDims global = cfg.make_grid();
        const double bw_bytes = stencil::bandwidth_bytes_per_step(global);
        const double bw_gbs   = bw_bytes / (ms_max * 1e-3) / 1e9;
        const double gflops   = stencil::flops_per_step(global)
                              / (ms_max * 1e-3) / 1e9;

        std::printf("Results (%d GPU%s):\n", g_nranks, g_nranks > 1 ? "s" : "");
        std::printf("  Max ms/step  : %.3f ms\n",  ms_max);
        std::printf("  Mean ms/step : %.3f ms\n",  ms_sum / g_nranks);
        std::printf("  Eff. BW      : %.1f GB/s\n", bw_gbs);
        std::printf("  GFLOP/s      : %.1f\n",      gflops);

        // Write scaling JSON for run_scaling.sh + plot_results.py
        if (!cfg.output_path.empty()) {
            stencil::ScalingResult sr;
            sr.mode       = "multigpu";
            sr.ngpu       = g_nranks;
            sr.time_ms    = ms_max;
            // Efficiency = time_1gpu / (N * time_Ngpu), approximated here as
            // BW_achieved / (N * BW_single).  Caller scripts compute this properly.
            sr.efficiency = 1.0 / static_cast<double>(g_nranks);

            stencil::write_scaling_json(cfg.output_path, {sr});
        }
    }

    mpi_finalize();
    return EXIT_SUCCESS;
}
