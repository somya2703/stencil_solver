/**
 * src/main_naive.cpp
 *
 * Entry point for the naive (global-memory) stencil solver.
 *
 * Usage:
 *   ./stencil_naive [options]
 *
 * Key flags:
 *   --nx N  --ny N  --nz N     Grid dimensions (default 256³)
 *   --steps N                  Timed time steps (default 100)
 *   --warmup N                 Un-timed warmup steps (default 10)
 *   --physics wave|heat        Physics model (default: wave)
 *   --velocity F               P-wave velocity in m/s (default: 1500)
 *   --device N                 CUDA device index (default: 0)
 *   --output path/to/out.json  Write benchmark results to JSON
 *   --verbose                  Print grid info and per-step output
 *
 * Example — reproduce the Nsight "before" profile:
 *   ./stencil_naive --nx 512 --ny 512 --nz 512 --steps 200 --warmup 20
 *
 * Example — write results for the benchmark report:
 *   ./stencil_naive --nx 256 --steps 500 --output results/naive_256.json
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

// ── Device info banner ────────────────────────────────────────────────────────
static void print_device_info(int device) {
#ifndef STENCIL_CPU_FALLBACK
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        std::printf("GPU [%d]: %s\n", device, prop.name);
        std::printf("  Compute: %d.%d   SMs: %d   Memory: %.1f GB\n",
                    prop.major, prop.minor,
                    prop.multiProcessorCount,
                    static_cast<double>(prop.totalGlobalMem) / 1e9);
        std::printf("  Peak bandwidth: %.1f GB/s\n",
                    static_cast<double>(prop.memoryClockRate) * 1e3
                    * static_cast<double>(prop.memoryBusWidth) / 8.0 / 1e9);
    }
#else
    (void)device;
    std::printf("CPU fallback mode (OpenMP)\n");
#endif
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::printf("=== stencil-solver: naive kernel ===\n\n");

    // Parse CLI
    stencil::Config cfg;
    try {
        cfg = stencil::parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n\nUsage: %s [--nx N] [--ny N] [--nz N] "
                     "[--steps N] [--warmup N] [--physics wave|heat] "
                     "[--velocity F] [--device N] [--output path] [--verbose]\n",
                     e.what(), argv[0]);
        return EXIT_FAILURE;
    }

    // Force naive variant — this binary always runs the global-memory kernel
    cfg.variant = stencil::KernelVariant::Naive;

    print_device_info(cfg.device);
    std::printf("\n");

    // ── Run across a range of grid sizes ────────────────────────────────────
    // If the user specified a grid size, run only that one.
    // Otherwise, run a small suite: 128³, 256³, 512³ — good for reports.
    std::vector<std::size_t> grid_sizes;
    if (cfg.nx != 256 || cfg.ny != 256 || cfg.nz != 256) {
        // User explicitly set a grid — honour it exactly
        grid_sizes = { cfg.nx };
    } else {
        grid_sizes = { 128, 256, 512 };
    }

    std::vector<stencil::PerfResult> results;
    results.reserve(grid_sizes.size());

    for (std::size_t N : grid_sizes) {
        stencil::Config run_cfg = cfg;
        run_cfg.nx = run_cfg.ny = run_cfg.nz = N;

        std::printf("Grid: %zu³   steps: %d   warmup: %d\n",
                    N, run_cfg.steps, run_cfg.warmup_steps);

        try {
            auto solver = stencil::make_solver(run_cfg);
            auto result = solver->run(run_cfg.steps, run_cfg.warmup_steps);

            stencil::print_perf(result);
            results.push_back(result);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  FAILED: %s\n", e.what());
            return EXIT_FAILURE;
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    if (results.size() > 1) {
        std::printf("\n── Summary ──────────────────────────────────────────\n");
        stencil::format_perf_table(results, 0);
    }

    // ── JSON output ───────────────────────────────────────────────────────────
    stencil::write_results_json(cfg.output_path, results, cfg.device);

    return EXIT_SUCCESS;
}
