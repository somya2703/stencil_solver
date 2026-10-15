/**
 * src/utils/timer.cpp
 *
 * Implementation of CpuTimer and any non-inline helpers declared in
 * include/stencil/timer.hpp.  GpuTimer is header-only (inline CUDA calls).
 *
 * This file is compiled by the host C++ compiler (not nvcc), so it must
 * not include cuda_runtime.h directly.
 */

#include "stencil/timer.hpp"

#include <cstdio>

namespace stencil {

// CpuTimer is fully inline in the header — nothing to define here.
// PerfResult, print_perf, print_speedup are also inline.
// This .cpp exists so CMake has a compilation unit for the stencil_utils
// target and so future non-inline timer helpers have a home.

/**
 * format_perf_table — print a per-grid-size comparison table to stdout.
 *
 * Every real call site passes a grid-size SWEEP of one kernel (several N,
 * one result per N) -- not several kernels at the same size -- so there is
 * no meaningful single "baseline" row to compute a speedup against.
 * (format_perf_table(results, 0) used to compare every row's time_ms
 * against row 0 regardless of grid_size, producing meaningless values
 * like 0.12x when row 0 was a much smaller grid than the row being
 * printed.) This prints a plain table instead: one row per result,
 * labelled by its grid_size, with no speedup column.
 *
 * @param results  vector of PerfResult, ordered as they should appear
 */
void format_perf_table(const std::vector<PerfResult>& results) {
    if (results.empty())
        return;

    std::printf("\n%-26s  %8s  %10s  %10s  %12s\n", "Kernel", "Grid", "ms/step", "GB/s", "GFLOP/s");
    std::printf("%-26s  %8s  %10s  %10s  %12s\n", std::string(26, '-').c_str(), "--------",
                "----------", "----------", "------------");

    for (const auto& r : results) {
        std::printf("  %-24s  %6zu³  %8.3f ms  %8.1f     %10.1f\n", r.label.c_str(), r.grid_size,
                    r.time_ms, r.bandwidth_gbs, r.gflops);
    }
    std::printf("\n");
}

}  // namespace stencil
