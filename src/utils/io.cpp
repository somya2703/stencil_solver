/**
 * src/utils/io.cpp
 *
 * JSON result serialisation for benchmark output.
 *
 * Writes a self-contained JSON file that plot_results.py can read.
 * We hand-roll the JSON to avoid a heavy dependency; the schema is simple.
 *
 * Schema:
 * {
 *   "gpu_name":       "NVIDIA A100 80GB PCIe",
 *   "cuda_version":   "12.4",
 *   "stencil_radius": 4,
 *   "precision":      "fp32",
 *   "results": [
 *     {
 *       "label":         "naive",
 *       "grid_size":     256,
 *       "time_ms":       3.14,
 *       "bandwidth_gbs": 412.5,
 *       "gflops":        198.3
 *     }, ...
 *   ]
 * }
 */

#include "stencil/io.hpp"
#include "stencil/timer.hpp"
#include "stencil/types.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>
#endif

namespace stencil {

// ── GPU device info ───────────────────────────────────────────────────────────
static std::string query_gpu_name(int device) {
#ifndef STENCIL_CPU_FALLBACK
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
        return std::string(prop.name);
#else
    (void)device;
#endif
    return "CPU (fallback)";
}

static std::string query_cuda_version() {
#ifndef STENCIL_CPU_FALLBACK
    int ver = 0;
    cudaRuntimeGetVersion(&ver);
    return std::to_string(ver / 1000) + "." + std::to_string((ver % 1000) / 10);
#else
    return "n/a";
#endif
}

// ── JSON helpers ──────────────────────────────────────────────────────────────
static std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else           { out += c; }
    }
    return out;
}

// ── Public API ────────────────────────────────────────────────────────────────
void write_results_json(const std::string&            path,
                        const std::vector<PerfResult>& results,
                        int                            device) {
    if (path.empty()) return;

    std::ostringstream js;
    js << "{\n";
    js << "  \"gpu_name\": \""      << escape_json(query_gpu_name(device)) << "\",\n";
    js << "  \"cuda_version\": \""  << escape_json(query_cuda_version())   << "\",\n";
    js << "  \"stencil_radius\": "  << STENCIL_RADIUS                      << ",\n";
    js << "  \"precision\": \""
       << (sizeof(real_t) == 8 ? "fp64" : "fp32")                         << "\",\n";
    js << "  \"results\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        js << "    {\n";
        js << "      \"label\": \""        << escape_json(r.label) << "\",\n";
        js << "      \"grid_size\": "      << r.grid_size           << ",\n";
        js << "      \"time_ms\": "        << r.time_ms             << ",\n";
        js << "      \"bandwidth_gbs\": "  << r.bandwidth_gbs       << ",\n";
        js << "      \"gflops\": "         << r.gflops              << "\n";
        js << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }

    js << "  ]\n}\n";

    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("io: cannot open output file: " + path);
    f << js.str();

    std::printf("Results written to: %s\n", path.c_str());
}

/**
 * write_scaling_json — specialised writer for multi-GPU scaling results.
 *
 * Schema extension adds "mode" (weak|strong), "ngpu", and "efficiency".
 */
void write_scaling_json(const std::string&                   path,
                        const std::vector<ScalingResult>&    results) {
    if (path.empty()) return;

    std::ostringstream js;
    js << "{\n";
    js << "  \"results\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        js << "    {\n";
        js << "      \"mode\": \""     << escape_json(r.mode) << "\",\n";
        js << "      \"ngpu\": "       << r.ngpu              << ",\n";
        js << "      \"time_ms\": "    << r.time_ms           << ",\n";
        js << "      \"efficiency\": " << r.efficiency        << "\n";
        js << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }

    js << "  ]\n}\n";

    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("io: cannot open scaling output: " + path);
    f << js.str();

    std::printf("Scaling results written to: %s\n", path.c_str());
}

}  // namespace stencil
