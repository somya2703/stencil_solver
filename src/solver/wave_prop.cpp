/**
 * src/solver/wave_prop.cpp
 *
 * WavePropSolver — acoustic wave equation solver.
 *
 * Physics:
 *   ∂²p/∂t² = v²(x) ∇²p
 *
 * Discretised with the standard second-order leapfrog time scheme:
 *   p_next = 2·p_cur - p_prev + dt² · v² · ∇²p_cur
 *
 * where ∇²p is approximated by the high-order (2R-th order) FD Laplacian
 * defined in stencil/types.hpp.
 *
 * This is the canonical kernel in seismic RTM and FWI.  The v² field
 * can represent:
 *   - A constant homogeneous model (unit tests, benchmarks)
 *   - A two-layer model (reflection tests)
 *   - A realistic velocity cube read from a SEG-Y file (future step)
 *
 * Stability (CFL condition):
 *   dt ≤ CFL_LIMIT · min(dx,dy,dz) / v_max
 *   Config::make_grid() enforces this automatically.
 */

#include "solver_base.hpp"
#include "stencil/kernels.cuh"

#include <string>
#include <memory>

namespace stencil {

// ── Concrete solver ───────────────────────────────────────────────────────────
class WavePropSolver final : public SolverBase {
public:
    WavePropSolver() = default;

    std::string name() const override {
        switch (cfg_.variant) {
            case KernelVariant::Naive:    return "wave/naive";
            case KernelVariant::Tiled:    return "wave/tiled";
            case KernelVariant::MultiGPU: return "wave/multigpu";
            default:                       return "wave/unknown";
        }
    }

protected:
    /**
     * step_impl — one leapfrog time step.
     *
     * Kernel reads  (p_cur, p_prev, vel2) → writes p_next.
     * After the kernel, the ring buffer rotates:
     *   old p_next becomes new p_cur
     *   old p_cur  becomes new p_prev
     *   old p_prev becomes the next write target
     *
     * No device synchronisation here — the stream ensures ordering,
     * and the next step's kernel launch is the implicit dependency.
     * This keeps the GPU pipeline full across steps.
     */
    void step_impl() override {
        launch(cfg_.variant,
               grid_->cur(),   // p at time t
               grid_->prev(),  // p at time t-dt
               grid_->vel2(),  // v²(x)
               grid_->next(),  // output: p at time t+dt
               grid_->dims(),
               stream_);
        grid_->rotate();
    }
};

// ── Factory ───────────────────────────────────────────────────────────────────
std::unique_ptr<ISolver> make_wave_solver(const Config& cfg) {
    (void)cfg;  // init() is called by make_solver() after construction
    return std::make_unique<WavePropSolver>();
}

}  // namespace stencil
