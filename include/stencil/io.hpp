#pragma once

/**
 * include/stencil/io.hpp
 *
 * Declarations for JSON result I/O helpers used by benchmarks.
 */

#include "stencil/timer.hpp"

#include <string>
#include <vector>

namespace stencil {

/// Single multi-GPU scaling data point.
struct ScalingResult {
    std::string mode;       ///< "weak" or "strong"
    int         ngpu = 1;
    double      time_ms    = 0.0;
    double      efficiency = 0.0;  ///< parallel efficiency [0,1]
};

/// Write benchmark results to a JSON file (no-op if path is empty).
void write_results_json(const std::string&             path,
                        const std::vector<PerfResult>& results,
                        int                            device = 0);

/// Write multi-GPU scaling results to a JSON file.
void write_scaling_json(const std::string&                path,
                        const std::vector<ScalingResult>& results);

}  // namespace stencil
