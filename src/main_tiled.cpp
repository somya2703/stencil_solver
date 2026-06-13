/**
 * src/main_tiled.cpp
 *
 * Entry point for the shared-memory tiled stencil solver.
 * Runs naive and tiled kernels back-to-back so the speedup is printed
 * in a single invocation — this is the primary Nsight profiling target
 * for the before/after comparison.
 *
 * Usage:
 *   ./stencil_tiled [options]   (same flags as stencil_naive)
 *
 * Nsight Compute — capture tiled kernel only:
 *   ncu --kernel-name kernel_tiled ./stencil_tiled --nx 512 --steps 1
 *
 * Nsight Systems — end-to-end timeline with NVTX regions:
 *   nsys profile ./stencil_tiled --nx 512 --steps 50 --warmup 10
 *
 * JSON output for benchmark report:
 *   ./stencil_tiled --nx 256 --steps 500 --output results/tiled_256.json
 *   (naive results written to results/tiled_256_naive.json automatically)
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

static void print_device_info(int device) {
#ifndef STENCIL_CPU_FALLBACK
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        std::printf("GPU [%d]: %s\n", device, prop.name);
        std::printf("  Compute : %d.%d    SMs: %d\n",
                    prop.major, prop.minor, prop.multiProcessorCount);
        std::printf("  Memory  : %.1f GB   Peak BW: %.1f GB/s\n",
                    static_cast<double>(prop.totalGlobalMem) / 1e9,
                    static_cast<double>(prop.memoryClockRate) * 1e3
                    * static_cast<double>(prop.memoryBusWidth) / 8.0 / 1e9);
        std::printf("  SMEM/block (tiled): %d B\n",
                    (TILE_X + 2*STENCIL_RADIUS) *
                    (TILE_Y + 2*STENCIL_RADIUS) * static_cast<int>(sizeof(real_t)));
    }
#else
    (void)device;
    std::printf("CPU fallback mode (OpenMP — tiled delegates to naive)\n");
#endif
}

// Derive naive output path from tiled path:
// "results/tiled.json" → "results/tiled_naive.json"
static std::string naive_json_path(const std::string& tiled_path) {
    if (tiled_path.empty()) return {};
    const auto dot = tiled_path.rfind('.');
    if (dot == std::string::npos) return tiled_path + "_naive";
    return tiled_path.substr(0, dot) + "_naive" + tiled_path.substr(dot);
}

int main(int argc, char** argv) {
    std::printf("=== stencil-solver: naive vs tiled comparison ===\n\n");

    stencil::Config cfg;
    try {
        cfg = stencil::parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "Error: %s\n\n"
            "Usage: %s [--nx N] [--ny N] [--nz N] [--steps N] [--warmup N]\n"
            "          [--physics wave|heat] [--velocity F] [--device N]\n"
            "          [--output path.json] [--verbose]\n",
            e.what(), argv[0]);
        return EXIT_FAILURE;
    }

    print_device_info(cfg.device);
    std::printf("\nStencil radius : %d  (%d-th order)\n",
                STENCIL_RADIUS, 2 * STENCIL_RADIUS);
    std::printf("Tile dims      : %d × %d  (SMEM block: %d × %d)\n",
                TILE_X, TILE_Y,
                TILE_X + 2*STENCIL_RADIUS, TILE_Y + 2*STENCIL_RADIUS);
    std::printf("Z pencil depth : %d\n\n", PENCIL_Z);

    // Grid-size sweep: if user sets explicit nx, run that one size only.
    // Otherwise run 128³, 256³, 512³ for the report table.
    const std::vector<std::size_t> sizes =
        (cfg.nx != 256) ? std::vector<std::size_t>{cfg.nx}
                        : std::vector<std::size_t>{128, 256, 512};

    std::printf("%-8s  %10s  %10s  %10s  %10s  %8s\n",
                "Grid", "Naive ms", "BW (GB/s)", "Tiled ms", "BW (GB/s)", "Speedup");
    std::printf("%-8s  %10s  %10s  %10s  %10s  %8s\n",
                "--------", "----------", "----------",
                "----------", "----------", "--------");

    std::vector<stencil::PerfResult> naive_results, tiled_results;

    for (const std::size_t N : sizes) {
        stencil::Config run = cfg;
        run.nx = run.ny = run.nz = N;

        stencil::PerfResult r_naive, r_tiled;

        // ── Naive ──────────────────────────────────────────────────────────────
        try {
            run.variant = stencil::KernelVariant::Naive;
            auto solver = stencil::make_solver(run);
            r_naive = solver->run(run.steps, run.warmup_steps);
            r_naive.label = "naive";
            naive_results.push_back(r_naive);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  naive %zu³ FAILED: %s\n", N, e.what());
            continue;
        }

        // ── Tiled ──────────────────────────────────────────────────────────────
        try {
            run.variant = stencil::KernelVariant::Tiled;
            auto solver = stencil::make_solver(run);
            r_tiled = solver->run(run.steps, run.warmup_steps);
            r_tiled.label = "tiled";
            tiled_results.push_back(r_tiled);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  tiled %zu³ FAILED: %s\n", N, e.what());
            continue;
        }

        const double speedup = (r_tiled.time_ms > 0.0)
                             ? r_naive.time_ms / r_tiled.time_ms : 0.0;

        std::printf("%-8zu  %8.3f ms  %8.1f     %8.3f ms  %8.1f     %6.2f×\n",
                    N,
                    r_naive.time_ms, r_naive.bandwidth_gbs,
                    r_tiled.time_ms, r_tiled.bandwidth_gbs,
                    speedup);
        std::fflush(stdout);
    }

    // ── Summary tables ─────────────────────────────────────────────────────────
    if (!naive_results.empty()) {
        std::printf("\n── Naive ────────────────────────────────────────────────\n");
        stencil::format_perf_table(naive_results, 0);
        std::printf("── Tiled ────────────────────────────────────────────────\n");
        stencil::format_perf_table(tiled_results, 0);
    }

    // ── JSON output ────────────────────────────────────────────────────────────
    // Tiled results → cfg.output_path
    // Naive results → derived _naive path (for plot_results.py --naive flag)
    stencil::write_results_json(cfg.output_path,              tiled_results, cfg.device);
    stencil::write_results_json(naive_json_path(cfg.output_path), naive_results, cfg.device);

    return EXIT_SUCCESS;
}
