/**
 * src/solver/solver_base.cpp
 *
 * SolverBase method bodies + make_solver() factory.
 *
 * All the shared logic that every physics solver needs lives here:
 * GPU allocation, initial condition upload, timed run loop, and
 * device-to-host download.  Subclasses only supply step_impl() and name().
 */

#include "solver_base.hpp"

#include "stencil/grid.hpp"
#include "stencil/io.hpp"
#include "stencil/kernels.cuh"
#include "stencil/types.hpp"

#include <cstdio>
#include <stdexcept>

namespace stencil {

// ── Forward declarations from the concrete solvers ───────────────────────────
std::unique_ptr<ISolver> make_wave_solver(const Config& cfg);
std::unique_ptr<ISolver> make_heat_solver(const Config& cfg);

// ── SolverBase::init ──────────────────────────────────────────────────────────
void SolverBase::init(const Config& cfg) {
    cfg_ = cfg;

#ifndef STENCIL_CPU_FALLBACK
    CUDA_CHECK(cudaSetDevice(cfg.device));
    CUDA_CHECK(cudaStreamCreate(&stream_));
#endif
    // Upload FD coefficients (to __constant__ memory on GPU, plain array on CPU).
    // Must be called before the first kernel launch.
    upload_fd_coefficients();

    grid_ = std::make_unique<DeviceGrid>(cfg.make_grid());

    // ── Initial pressure field: Gaussian pulse ────────────────────────────────
    // Both p_cur and p_prev start identical so that the first-step
    // finite-difference time derivative (p_cur - p_prev)/dt is zero —
    // i.e. a quiescent medium with a pressure disturbance, no initial velocity.
    HostGrid h_field(grid_->dims());
    init_gaussian(h_field,
                  cfg.pulse_cx, cfg.pulse_cy, cfg.pulse_cz,
                  cfg.pulse_sigma);
    grid_->upload_cur (h_field, stream_);
    grid_->upload_prev(h_field, stream_);

    // ── Velocity-squared field ────────────────────────────────────────────────
    HostGrid h_vel(grid_->dims());
    init_constant_velocity(h_vel, cfg.velocity);
    grid_->upload_vel2(h_vel, stream_);

#ifndef STENCIL_CPU_FALLBACK
    CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif

    if (cfg.verbose) {
        print_grid_info(grid_->dims(), name().c_str());
    }
}

// ── SolverBase::run ───────────────────────────────────────────────────────────
PerfResult SolverBase::run(int n_steps, int warmup) {
    NVTX_PUSH(name().c_str());

    // Warmup: execute steps without timing.
    // Fills GPU caches, triggers JIT compilation of any lazy PTX, and
    // ensures the GPU is at steady-state clock frequency before measurement.
    for (int i = 0; i < warmup; ++i) {
        step_impl();
    }

#ifndef STENCIL_CPU_FALLBACK
    CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif

    // Timed region — CUDA events give μs-precision independent of CPU timing.
    GpuTimer timer;
    timer.start(stream_);

    for (int i = 0; i < n_steps; ++i) {
        step_impl();
    }

    timer.stop(stream_);

    // elapsed_ms() blocks until all work in the stream is complete.
    const double total_ms    = static_cast<double>(timer.elapsed_ms());
    const double ms_per_step = total_ms / static_cast<double>(n_steps);

    // Roofline metrics
    const double bw_bytes = bandwidth_bytes_per_step(grid_->dims());
    const double flops    = flops_per_step(grid_->dims());

    NVTX_POP();

    PerfResult r;
    r.label         = name();
    r.time_ms       = ms_per_step;
    r.bandwidth_gbs = bw_bytes / (ms_per_step * 1e-3) / 1e9;
    r.gflops        = flops    / (ms_per_step * 1e-3) / 1e9;
    r.grid_size     = grid_->dims().nx;
    return r;
}

// ── SolverBase::download ──────────────────────────────────────────────────────
void SolverBase::download(HostGrid& out) const {
    grid_->download_cur(out, stream_);
#ifndef STENCIL_CPU_FALLBACK
    CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif
}

// ── Factory ───────────────────────────────────────────────────────────────────
std::unique_ptr<ISolver> make_solver(const Config& cfg) {
    std::unique_ptr<ISolver> solver;
    switch (cfg.physics) {
        case Physics::WavePropagation:
            solver = make_wave_solver(cfg);
            break;
        case Physics::HeatDiffusion:
            solver = make_heat_solver(cfg);
            break;
        default:
            throw std::invalid_argument("make_solver: unknown Physics enum value");
    }
    solver->init(cfg);
    return solver;
}

}  // namespace stencil
