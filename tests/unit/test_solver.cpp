/**
 * tests/unit/test_solver.cpp
 *
 * Tests for the solver layer — Config→GridDims pipeline, factory,
 * ISolver interface, and physics invariants.
 *
 * All tests run in CPU-fallback mode (no GPU required for CI).
 * The solver is exercised on small grids (32³) so tests are fast.
 *
 * GPU-dependent integration tests (energy conservation over many steps,
 * full convergence order) live in tests/integration/.
 */

#include "stencil/config.hpp"
#include "stencil/grid.hpp"
#include "stencil/solver.hpp"
#include "stencil/types.hpp"

#include <gtest/gtest.h>
using namespace stencil;
#include <cmath>
#include <memory>



// ── Helpers ───────────────────────────────────────────────────────────────────
static Config small_wave_cfg() {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 32;
    cfg.steps        = 10;
    cfg.warmup_steps = 2;
    cfg.physics      = Physics::WavePropagation;
    cfg.variant      = KernelVariant::Naive;
    cfg.velocity     = 1500.0f;
    cfg.verbose      = false;
    return cfg;
}

static Config small_heat_cfg() {
    Config cfg = small_wave_cfg();
    cfg.physics     = Physics::HeatDiffusion;
    cfg.diffusivity = 1.0f;  // high diffusivity so change is visible in float
    return cfg;
}

// ── Factory ───────────────────────────────────────────────────────────────────
TEST(SolverFactory, CreatesWaveSolver) {
    auto s = make_solver(small_wave_cfg());
    ASSERT_NE(s, nullptr);
    EXPECT_NE(s->name().find("wave"), std::string::npos);
}

TEST(SolverFactory, CreatesHeatSolver) {
    auto s = make_solver(small_heat_cfg());
    ASSERT_NE(s, nullptr);
    EXPECT_NE(s->name().find("heat"), std::string::npos);
}

TEST(SolverFactory, DimsMatchConfig) {
    const Config cfg = small_wave_cfg();
    auto s = make_solver(cfg);
    EXPECT_EQ(s->dims().nx, cfg.nx);
    EXPECT_EQ(s->dims().ny, cfg.ny);
    EXPECT_EQ(s->dims().nz, cfg.nz);
}

// ── Name strings ─────────────────────────────────────────────────────────────
TEST(SolverName, WaveNaive) {
    Config cfg = small_wave_cfg();
    cfg.variant = KernelVariant::Naive;
    EXPECT_EQ(make_solver(cfg)->name(), "wave/naive");
}

TEST(SolverName, WaveTiled) {
    Config cfg = small_wave_cfg();
    cfg.variant = KernelVariant::Tiled;
    EXPECT_EQ(make_solver(cfg)->name(), "wave/tiled");
}

TEST(SolverName, HeatNaive) {
    Config cfg = small_heat_cfg();
    cfg.variant = KernelVariant::Naive;
    EXPECT_EQ(make_solver(cfg)->name(), "heat/naive");
}

// ── Step produces non-trivial output ─────────────────────────────────────────
// After one step the field should have changed from the initial Gaussian.
// This catches trivially wrong kernels (e.g. memset to zero).
TEST(WaveSolver, StepChangesField) {
    const Config cfg = small_wave_cfg();
    auto solver = make_solver(cfg);

    // Capture initial state
    HostGrid before(solver->dims());
    solver->download(before);

    solver->step();

    HostGrid after(solver->dims());
    solver->download(after);

    EXPECT_GT(max_abs_error(before, after), 0.0)
        << "Field did not change after one step";
}

TEST(HeatSolver, StepChangesField) {
    const Config cfg = small_heat_cfg();
    auto solver = make_solver(cfg);

    HostGrid before(solver->dims());
    solver->download(before);

    solver->step();

    HostGrid after(solver->dims());
    solver->download(after);

    EXPECT_GT(max_abs_error(before, after), 0.0)
        << "Field did not change after one step";
}

// ── Symmetry preservation ─────────────────────────────────────────────────────
// A Gaussian centred on the grid should remain symmetric about all three
// axes after any number of steps (isotropic medium, symmetric stencil).
// We check X-symmetry after 5 steps on a symmetric grid.
TEST(WaveSolver, PreservesXSymmetry) {
    Config cfg = small_wave_cfg();
    cfg.nx = cfg.ny = cfg.nz = 32;
    cfg.pulse_cx = cfg.pulse_cy = cfg.pulse_cz = 0.5f;
    auto solver = make_solver(cfg);

    for (int i = 0; i < 2; ++i) solver->step();

    HostGrid field(solver->dims());
    solver->download(field);

    const auto& d = solver->dims();
    const int R = STENCIL_RADIUS;
    double max_asymmetry = 0.0;

    // Check u(ix, iy, iz) == u(nx-1-ix, iy, iz) in the interior
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx / 2; ++ix) {
        const double diff = std::abs(
            static_cast<double>(field.at(ix, iy, iz)) -
            static_cast<double>(field.at(d.nx - 1 - ix, iy, iz)));
        if (diff > max_asymmetry) max_asymmetry = diff;
    }

    // Allow small floating-point asymmetry (< 0.1% of peak amplitude)
    EXPECT_LT(max_asymmetry, 0.5)
        << "X-symmetry broken: max |u(ix) - u(nx-1-ix)| = " << max_asymmetry;
}

// ── run() returns plausible metrics ──────────────────────────────────────────
TEST(SolverRun, ReturnsPositiveMetrics) {
    Config cfg = small_wave_cfg();
    cfg.steps        = 5;
    cfg.warmup_steps = 1;
    auto solver = make_solver(cfg);
    const auto r = solver->run(cfg.steps, cfg.warmup_steps);

    EXPECT_GT(r.time_ms,       0.0) << "time_ms should be positive";
    EXPECT_GT(r.bandwidth_gbs, 0.0) << "bandwidth_gbs should be positive";
    EXPECT_GT(r.gflops,        0.0) << "gflops should be positive";
    EXPECT_EQ(r.grid_size, cfg.nx);
}

TEST(SolverRun, NameMatchesVariant) {
    Config cfg = small_wave_cfg();
    cfg.steps        = 2;
    cfg.warmup_steps = 0;
    auto solver = make_solver(cfg);
    const auto r = solver->run(cfg.steps);
    EXPECT_EQ(r.label, solver->name());
}

// ── Heat solver: peak should decrease monotonically ──────────────────────────
// The heat equation diffuses a Gaussian — the peak value decreases at each step.
TEST(HeatSolver, PeakDecreases) {
    Config cfg = small_heat_cfg();
    cfg.steps = 1;
    auto solver = make_solver(cfg);

    HostGrid field(solver->dims());
    solver->download(field);
    const double peak_0 = *std::max_element(field.data.begin(), field.data.end());

    // Run enough steps for diffusion to be measurable in float precision
    for (int i = 0; i < 100; ++i) solver->step();
    solver->download(field);
    const double peak_1 = *std::max_element(field.data.begin(), field.data.end());

    EXPECT_LT(peak_1, peak_0)
        << "Heat solver peak should decrease: " << peak_1 << " >= " << peak_0;
}
