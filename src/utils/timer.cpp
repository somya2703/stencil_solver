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
 * format_perf_table — print a comparison table to stdout.
 *
 * @param results  vector of PerfResult, ordered as they should appear
 * @param baseline index of the baseline entry (speedup denominator)
 */
void format_perf_table(const std::vector<PerfResult>& results,
                       std::size_t baseline_idx) {
    if (results.empty()) return;

    std::printf("\n%-26s  %10s  %10s  %12s  %8s\n",
                "Kernel", "ms/step", "GB/s", "GFLOP/s", "Speedup");
    std::printf("%-26s  %10s  %10s  %12s  %8s\n",
                std::string(26, '-').c_str(),
                "----------", "----------", "------------", "--------");

    const double base_ms = results[baseline_idx].time_ms;
    for (const auto& r : results) {
        const double speedup = (base_ms > 0.0) ? (base_ms / r.time_ms) : 1.0;
        std::printf("  %-24s  %8.3f ms  %8.1f     %10.1f    %6.2f×\n",
                    r.label.c_str(),
                    r.time_ms,
                    r.bandwidth_gbs,
                    r.gflops,
                    speedup);
    }
    std::printf("\n");
}

}  // namespace stencil
