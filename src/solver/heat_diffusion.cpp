/**
 * src/solver/heat_diffusion.cpp
 *
 * HeatDiffusionSolver — parabolic heat equation solver.
 *
 * Physics:
 *   ∂T/∂t = α ∇²T
 *
 * where α is the thermal diffusivity [m²/s].
 *
 * Discretised with an explicit forward-Euler time scheme:
 *   T_next = T_cur + dt · α · ∇²T_cur
 *
 * The same high-order FD Laplacian used by the wave solver is reused here,
 * which means this solver exercises exactly the same memory access pattern.
 * This is useful for:
 *   - Validating the Laplacian against the analytical solution (see tests)
 *   - Benchmarking the kernel on a numerically stable problem
 *     (heat equation is unconditionally stable for dt ≤ dx²/6α, easier
 *      to satisfy than the wave CFL)
 *
 * Stability (von Neumann):
 *   dt ≤ dx² / (2·d·α)   where d=3 (spatial dimensions)
 *   Config::make_grid() sets dt via the wave CFL, which is typically
 *   stricter — so the heat solver inherits a safe dt automatically.
 *
 * Implementation note:
 *   The wave equation kernel (p_next = 2p - p_prev + dt²·v²·∇²p) does
 *   not directly implement forward-Euler for heat.  We therefore store
 *   the diffusivity α in the vel2 field (as α rather than v²), and we
 *   set p_prev = p_cur so that the wave formula reduces to:
 *
 *     T_next = 2T - T + dt² · α · ∇²T = T + dt² · α · ∇²T
 *
 *   We then set the effective "dt²" = dt (not dt²) inside the kernel by
 *   passing a modified GridDims with dt_eff = sqrt(dt).  This avoids
 *   adding a separate kernel path for heat while reusing all the
 *   optimised Laplacian code.
 *
 *   A dedicated heat kernel (forward-Euler, single time level) will be
 *   added in a future optimisation step.
 */

#include "solver_base.hpp"
#include "stencil/kernels.cuh"

#include <cmath>
#include <memory>
#include <string>

namespace stencil {

// ── Concrete solver ───────────────────────────────────────────────────────────
class HeatDiffusionSolver final : public SolverBase {
public:
    HeatDiffusionSolver() = default;

    std::string name() const override {
        switch (cfg_.variant) {
            case KernelVariant::Naive:    return "heat/naive";
            case KernelVariant::Tiled:    return "heat/tiled";
            case KernelVariant::MultiGPU: return "heat/multigpu";
            default:                       return "heat/unknown";
        }
    }

    // Override init to set up the diffusivity field and adjusted time step.
    void init(const Config& cfg) override {
        // Let SolverBase handle GPU allocation, stream, and IC upload.
        SolverBase::init(cfg);

        // Re-upload vel2 as diffusivity α (not v²).
        // SolverBase::init() already set vel2 = v² — we overwrite here.
        HostGrid h_diff(grid_->dims());
        std::fill(h_diff.data.begin(), h_diff.data.end(),
                  static_cast<real_t>(cfg.diffusivity));
        grid_->upload_vel2(h_diff, stream_);

        // Modify the grid's dt so that dt_eff² == dt_physical.
        // After this, the wave kernel computes:
        //   T + dt_eff² · α · ∇²T = T + dt · α · ∇²T  ✓
        heat_dims_       = grid_->dims();
        heat_dims_.dt    = std::sqrt(static_cast<double>(cfg_.make_grid().dt));

#ifndef STENCIL_CPU_FALLBACK
        CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif
    }

protected:
    /**
     * step_impl — one explicit forward-Euler step for the heat equation.
     *
     * We pass heat_dims_ (with adjusted dt) to the kernel so the
     * time-integration term works correctly.  The ring buffer rotation
     * is the same as the wave solver — p_prev is irrelevant since
     * p_prev == p_cur is enforced below each step to keep the formula valid.
     */
    void step_impl() override {
        launch(cfg_.variant,
               grid_->cur(),
               grid_->prev(),
               grid_->vel2(),
               grid_->next(),
               heat_dims_,     // adjusted dt
               stream_);
        grid_->rotate();

        // Keep p_prev == p_cur so the wave-equation formula stays as
        // forward Euler: T_next = 2T_cur - T_cur + dt_eff²·α·∇²T_cur
        //                       =   T_cur + dt·α·∇²T_cur
        // We do this by copying cur → prev after rotation.
        // (cur and prev now refer to the new time level and the slot we
        // just wrote into respectively, thanks to the ring rotation.)
#ifndef STENCIL_CPU_FALLBACK
        CUDA_CHECK(cudaMemcpyAsync(
            grid_->prev(), grid_->cur(),
            grid_->dims().size() * sizeof(real_t),
            cudaMemcpyDeviceToDevice, stream_));
#else
        std::copy(grid_->cur(),
                  grid_->cur() + grid_->dims().size(),
                  grid_->prev());
#endif
    }

private:
    GridDims heat_dims_{};   // dims with dt_eff = sqrt(dt_physical)
};

// ── Factory ───────────────────────────────────────────────────────────────────
std::unique_ptr<ISolver> make_heat_solver(const Config& cfg) {
    (void)cfg;
    return std::make_unique<HeatDiffusionSolver>();
}

}  // namespace stencil
