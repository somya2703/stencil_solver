/**
 * tests/perf/bench_tiled.cpp
 *
 * Side-by-side benchmark of naive vs tiled kernels across grid sizes.
 * Produces a speedup table and writes both result sets to JSON so
 * plot_results.py can render the comparison chart.
 *
 * Usage:
 *   ./bench/bench_tiled [options]
 *   ./bench/bench_tiled --nx 512 --steps 200 \
 *       --output results/tiled.json
 *
 * The naive results are written to <output_path>.naive.json automatically.
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
#include <string>
#include <vector>

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

// Derive a sibling output path: "results/tiled.json" → "results/naive.json"
static std::string naive_path(const std::string& tiled_path) {
    if (tiled_path.empty()) return {};
    const auto pos = tiled_path.rfind('.');
    if (pos == std::string::npos) return tiled_path + ".naive";
    return tiled_path.substr(0, pos) + "_naive" + tiled_path.substr(pos);
}

int main(int argc, char** argv) {
    std::printf("=== bench_tiled — naive vs shared-memory tiled ===\n\n");

    stencil::Config cfg;
    try {
        cfg = stencil::parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }

    print_device_banner(cfg.device);
    std::printf("\nRadius : %d  (%d-th order)\n", STENCIL_RADIUS, 2*STENCIL_RADIUS);
    std::printf("Steps  : %d  Warmup: %d\n\n", cfg.steps, cfg.warmup_steps);

    const std::vector<std::size_t> sizes =
        (cfg.nx != 256) ? std::vector<std::size_t>{cfg.nx}
                        : std::vector<std::size_t>{64, 128, 192, 256, 320, 384, 512};

    // Header
    std::printf("%-8s  %10s  %10s  %10s  %10s  %8s\n",
                "Grid", "Naive ms", "Naive GB/s", "Tiled ms", "Tiled GB/s", "Speedup");
    std::printf("%-8s  %10s  %10s  %10s  %10s  %8s\n",
                "--------", "----------", "----------",
                "----------", "----------", "--------");

    std::vector<stencil::PerfResult> naive_results, tiled_results;

    for (std::size_t N : sizes) {
        stencil::Config run = cfg;
        run.nx = run.ny = run.nz = N;

        stencil::PerfResult r_naive, r_tiled;

        try {
            run.variant = stencil::KernelVariant::Naive;
            auto s = stencil::make_solver(run);
            r_naive = s->run(run.steps, run.warmup_steps);
            r_naive.label = "naive";
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  naive %zu³ FAILED: %s\n", N, e.what());
            continue;
        }

        try {
            run.variant = stencil::KernelVariant::Tiled;
            auto s = stencil::make_solver(run);
            r_tiled = s->run(run.steps, run.warmup_steps);
            r_tiled.label = "tiled";
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  tiled %zu³ FAILED: %s\n", N, e.what());
            continue;
        }

        const double speedup = (r_tiled.time_ms > 0)
                             ? r_naive.time_ms / r_tiled.time_ms : 0.0;

        std::printf("%-8zu  %8.3f ms  %8.1f     %8.3f ms  %8.1f     %6.2f×\n",
                    N,
                    r_naive.time_ms, r_naive.bandwidth_gbs,
                    r_tiled.time_ms, r_tiled.bandwidth_gbs,
                    speedup);
        std::fflush(stdout);

        naive_results.push_back(r_naive);
        tiled_results.push_back(r_tiled);
    }

    // Overall summary
    if (!naive_results.empty()) {
        std::printf("\n── Summary ──────────────────────────────────────────────\n");
        stencil::format_perf_table(naive_results, 0);
        std::printf("── Tiled ────────────────────────────────────────────────\n");
        stencil::format_perf_table(tiled_results, 0);
    }

    // JSON output — separate files for naive and tiled so plot_results.py
    // can ingest them with --naive and --tiled flags independently
    stencil::write_results_json(cfg.output_path,          tiled_results, cfg.device);
    stencil::write_results_json(naive_path(cfg.output_path), naive_results, cfg.device);

    return EXIT_SUCCESS;
}
