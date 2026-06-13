/**
 * tests/perf/bench_naive.cpp
 *
 * Benchmark for the naive global-memory stencil kernel.
 * Runs across a configurable range of grid sizes and outputs
 * timing + roofline metrics as JSON (consumed by plot_results.py).
 *
 * Usage:
 *   ./bench/bench_naive [options]
 *   ./bench/bench_naive --nx 512 --steps 200 --output results/naive.json
 *
 * Output columns:
 *   Grid  |  ms/step  |  GB/s  |  GFLOP/s
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
#include <vector>
#include <stdexcept>

static void print_device_banner(int device) {
#ifndef STENCIL_CPU_FALLBACK
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, device) == cudaSuccess) {
        std::printf("Device : %s\n", p.name);
        std::printf("Memory : %.1f GB   Peak BW: %.1f GB/s\n",
                    static_cast<double>(p.totalGlobalMem) / 1e9,
                    static_cast<double>(p.memoryClockRate) * 1e3
                    * static_cast<double>(p.memoryBusWidth) / 8.0 / 1e9);
    }
#else
    (void)device;
    std::printf("Device : CPU (OpenMP fallback)\n");
#endif
}

int main(int argc, char** argv) {
    std::printf("=== bench_naive — global-memory stencil ===\n\n");

    stencil::Config cfg;
    try {
        cfg = stencil::parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    cfg.variant = stencil::KernelVariant::Naive;

    print_device_banner(cfg.device);
    std::printf("\nRadius : %d  (%d-th order)\n", STENCIL_RADIUS, 2*STENCIL_RADIUS);
    std::printf("Steps  : %d  Warmup: %d\n\n", cfg.steps, cfg.warmup_steps);

    // Grid sizes to sweep. If user set a specific size, use only that.
    const std::vector<std::size_t> sizes =
        (cfg.nx != 256) ? std::vector<std::size_t>{cfg.nx}
                        : std::vector<std::size_t>{64, 128, 192, 256, 320, 384, 512};

    std::printf("%-8s  %10s  %10s  %12s\n",
                "Grid", "ms/step", "GB/s", "GFLOP/s");
    std::printf("%-8s  %10s  %10s  %12s\n",
                "--------", "----------", "----------", "------------");

    std::vector<stencil::PerfResult> results;
    results.reserve(sizes.size());

    for (std::size_t N : sizes) {
        stencil::Config run = cfg;
        run.nx = run.ny = run.nz = N;

        try {
            auto solver = stencil::make_solver(run);
            auto r      = solver->run(run.steps, run.warmup_steps);
            r.label     = "naive";

            std::printf("%-8zu  %8.3f ms  %8.1f     %10.1f\n",
                        N, r.time_ms, r.bandwidth_gbs, r.gflops);
            std::fflush(stdout);
            results.push_back(r);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  %zu³ FAILED: %s\n", N, e.what());
        }
    }

    stencil::write_results_json(cfg.output_path, results, cfg.device);
    return EXIT_SUCCESS;
}
