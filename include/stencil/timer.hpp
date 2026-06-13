#pragma once

/**
 * include/stencil/timer.hpp
 *
 * Two timer classes:
 *   GpuTimer  — CUDA-event-based, microsecond precision, stream-aware.
 *   CpuTimer  — std::chrono wall-clock, used in CPU-fallback mode and
 *               for host-side setup timing.
 *
 * PerfResult  — carries timing + throughput metrics for one benchmark run.
 * print_perf / print_speedup — formatted console output.
 */

#include "stencil/types.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>



namespace stencil {

// ── CPU wall-clock timer ──────────────────────────────────────────────────────
class CpuTimer {
public:
    void start() noexcept { t0_ = clock_t::now(); }
    void stop()  noexcept { t1_ = clock_t::now(); }

    /// Elapsed milliseconds between last start() and stop().
    double elapsed_ms() const noexcept {
        return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
    }

private:
    using clock_t = std::chrono::steady_clock;
    clock_t::time_point t0_, t1_;
};

// ── GPU event timer ───────────────────────────────────────────────────────────
#ifndef STENCIL_CPU_FALLBACK

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Non-copyable, non-movable (owning CUDA resources)
    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaEventRecord(start_, s));
    }
    void stop(cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, s));
    }
    /// Blocks until the stop event completes, then returns elapsed ms.
    float elapsed_ms() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventSynchronize(stop_));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

#else  // CPU fallback: GpuTimer is just a CpuTimer alias

class GpuTimer {
public:
    void  start(void* = nullptr) noexcept { t_.start(); }
    void  stop(void*  = nullptr) noexcept { t_.stop(); }
    float elapsed_ms() noexcept { return static_cast<float>(t_.elapsed_ms()); }
private:
    CpuTimer t_;
};

#endif  // STENCIL_CPU_FALLBACK

// ── Performance result ────────────────────────────────────────────────────────
struct PerfResult {
    std::string label;
    double      time_ms      = 0.0;  ///< mean wall time per step [ms]
    double      bandwidth_gbs = 0.0; ///< effective bandwidth [GB/s]
    double      gflops        = 0.0; ///< compute throughput [GFLOP/s]
    std::size_t grid_size     = 0;   ///< N where grid is N×N×N (for reporting)
};

// ── Console output ────────────────────────────────────────────────────────────
inline void print_perf(const PerfResult& r) {
    std::printf("  %-24s  %8.3f ms/step  %7.1f GB/s  %7.1f GFLOP/s\n",
                r.label.c_str(), r.time_ms, r.bandwidth_gbs, r.gflops);
}

inline void print_speedup(const PerfResult& base, const PerfResult& opt) {
    if (base.time_ms <= 0.0) return;
    std::printf("\n  Speedup  (%s → %s) : %.2f×\n",
                base.label.c_str(), opt.label.c_str(),
                base.time_ms / opt.time_ms);
    std::printf("  Bandwidth gain          : %+.1f%%\n",
                (opt.bandwidth_gbs / base.bandwidth_gbs - 1.0) * 100.0);
}


// ── Multi-result table (implemented in src/utils/timer.cpp) ──────────────────
void format_perf_table(const std::vector<PerfResult>& results,
                       std::size_t baseline_idx = 0);

}  // namespace stencil
