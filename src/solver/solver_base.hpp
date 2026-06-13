#pragma once

/**
 * src/solver/solver_base.hpp  (internal — not installed)
 *
 * SolverBase — CRTP-free abstract base that implements the boilerplate
 * shared by WavePropSolver and HeatDiffusionSolver:
 *   - GPU resource ownership (DeviceGrid, cudaStream_t)
 *   - init()      — allocates grid, uploads ICs and velocity field
 *   - run()       — timed loop with warmup, returns PerfResult
 *   - download()  — device → host copy
 *   - step()      — public single-step, delegates to step_impl()
 *
 * Subclasses only need to implement:
 *   void        step_impl()  — call the right kernel
 *   std::string name() const — e.g. "wave/naive"
 */

#include "stencil/config.hpp"
#include "stencil/grid.hpp"
#include "stencil/kernels.cuh"
#include "stencil/solver.hpp"
#include "stencil/timer.hpp"
#include "stencil/types.hpp"

#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>
#endif

#include <memory>
#include <string>

namespace stencil {

class SolverBase : public ISolver {
public:
    ~SolverBase() override {
#ifndef STENCIL_CPU_FALLBACK
        if (stream_) cudaStreamDestroy(stream_);
#endif
    }

    // ISolver interface
    void       init(const Config& cfg)     override;
    void       step()                      override { step_impl(); }
    PerfResult run(int n_steps, int warmup = 0) override;
    void       download(HostGrid& out) const override;
    const GridDims& dims() const           override { return grid_->dims(); }

protected:
    SolverBase() = default;

    /// Subclass hook: advance one step using the appropriate kernel.
    virtual void step_impl() = 0;

    // State accessible to subclasses
    Config                    cfg_;
    std::unique_ptr<DeviceGrid> grid_;

#ifndef STENCIL_CPU_FALLBACK
    cudaStream_t stream_ = nullptr;
#else
    void*        stream_ = nullptr;   // unused in CPU path, kept for uniform code
#endif
};

}  // namespace stencil
