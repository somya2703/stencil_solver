/**
 * tests/perf/bench_scaling.cpp
 *
 * Single-GPU scaling benchmark: measures throughput vs grid size
 * and computes memory-bandwidth efficiency against the device peak.
 * Multi-GPU scaling is handled by scripts/run_scaling.sh + stencil_multigpu.
 *
 * Also serves as a sanity check that roofline numbers are plausible
 * before running the full multi-GPU scaling experiments.
 *
 * Usage:
 *   ./bench/bench_scaling [options]
 *   ./bench/bench_scaling --output results/scaling.json
 *
 * Output JSON schema (for plot_results.py --scaling):
 * {
 *   "results": [
 *     { "mode": "single_gpu", "ngpu": 1, "time_ms": ..., "efficiency": ... }
 *   ]
 * }
 */

#include "stencil/config.hpp"
#include "stencil/io.hpp"
#include "stencil/solver.hpp"
#include "stencil/timer.hpp"
#include "stencil/types.hpp"

#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// Query peak memory bandwidth in GB/s for the given device.
static double peak_bandwidth_gbs(int device) {
#ifndef STENCIL_CPU_FALLBACK
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, device) == cudaSuccess) {
        return static_cast<double>(p.memoryClockRate) * 1e3
             * static_cast<double>(p.memoryBusWidth) / 8.0 / 1e9;
    }
#else
    (void)device;
#endif
    return 900.0;  // fallback: ~A100 peak
}

int main(int argc, char** argv) {
    std::printf("=== bench_scaling — single-GPU roofline ===\n\n");

    stencil::Config cfg;
    try {
        cfg = stencil::parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }

    cfg.variant      = stencil::KernelVariant::Tiled;
    cfg.warmup_steps = std::max(cfg.warmup_steps, 10);

    const double peak_bw = peak_bandwidth_gbs(cfg.device);
    std::printf("Peak BW   : %.1f GB/s\n", peak_bw);
    std::printf("Kernel    : tiled\n");
    std::printf("Radius    : %d  (%d-th order)\n\n",
                STENCIL_RADIUS, 2 * STENCIL_RADIUS);

    const std::vector<std::size_t> sizes =
        (cfg.nx != 256) ? std::vector<std::size_t>{cfg.nx}
                        : std::vector<std::size_t>{64, 96, 128, 192, 256, 384, 512};

    std::printf("%-8s  %10s  %10s  %10s\n",
                "Grid", "ms/step", "GB/s", "BW Eff.");
    std::printf("%-8s  %10s  %10s  %10s\n",
                "--------", "----------", "----------", "----------");

    std::vector<stencil::ScalingResult> scaling_results;

    for (std::size_t N : sizes) {
        stencil::Config run = cfg;
        run.nx = run.ny = run.nz = N;

        try {
            auto solver = stencil::make_solver(run);
            auto r = solver->run(run.steps, run.warmup_steps);

            const double efficiency = r.bandwidth_gbs / peak_bw;
            std::printf("%-8zu  %8.3f ms  %8.1f     %8.1f%%\n",
                        N, r.time_ms, r.bandwidth_gbs, efficiency * 100.0);
            std::fflush(stdout);

            stencil::ScalingResult sr;
            sr.mode       = "single_gpu";
            sr.ngpu       = 1;
            sr.time_ms    = r.time_ms;
            sr.efficiency = efficiency;
            scaling_results.push_back(sr);

        } catch (const std::exception& e) {
            std::fprintf(stderr, "  %zu³ FAILED: %s\n", N, e.what());
        }
    }

    stencil::write_scaling_json(cfg.output_path, scaling_results);
    return EXIT_SUCCESS;
}
