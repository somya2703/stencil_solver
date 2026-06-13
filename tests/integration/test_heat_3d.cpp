/**
 * tests/integration/test_heat_3d.cpp
 *
 * Integration tests for the heat diffusion solver.
 * Label: "gpu" — require a CUDA device.
 *
 * Tests:
 *   1. Peak decreases monotonically (diffusion always smooths)
 *   2. Total "heat" (integral of T) is conserved (no sources/sinks)
 *   3. Solution stays non-negative (Gaussian IC → always ≥ 0)
 *   4. Analytical convergence — Gaussian diffuses as σ²(t) = σ²(0) + 2αt
 *   5. Spatial symmetry preserved
 */

#include "stencil/config.hpp"
#include "stencil/grid.hpp"
#include "stencil/solver.hpp"
#include "stencil/types.hpp"

#include <gtest/gtest.h>
using namespace stencil;
#include <algorithm>
#include <cmath>
#include <numeric>



class HeatSolver3DTest : public ::testing::Test {
protected:
    static Config make_cfg(std::size_t N = 48) {
        Config cfg;
        cfg.nx = cfg.ny = cfg.nz = N;
        cfg.steps        = 20;
        cfg.warmup_steps = 0;
        cfg.physics      = Physics::HeatDiffusion;
        cfg.variant      = KernelVariant::Naive;
        cfg.diffusivity  = 1.0e-3f;
        cfg.velocity     = 500.0f;   // used only for dt = CFL * dx / v
        cfg.dx = cfg.dy  = cfg.dz = 1.0f;
        cfg.pulse_cx = cfg.pulse_cy = cfg.pulse_cz = 0.5f;
        cfg.pulse_sigma = 5.0f;
        cfg.verbose      = false;
        return cfg;
    }

    static double total_heat(const HostGrid& g) {
        double s = 0.0;
        for (auto v : g.data) s += static_cast<double>(v);
        return s;
    }

    static double peak_value(const HostGrid& g) {
        return static_cast<double>(
            *std::max_element(g.data.begin(), g.data.end()));
    }
};

// ── 1. Peak decreases monotonically ──────────────────────────────────────────
TEST_F(HeatSolver3DTest, PeakDecreaseMonotonically) {
    Config cfg = make_cfg();
    auto solver = make_solver(cfg);

    HostGrid f(solver->dims());
    solver->download(f);
    double prev_peak = peak_value(f);

    for (int i = 0; i < 10; ++i) {
        solver->step();
        solver->download(f);
        const double curr_peak = peak_value(f);
        EXPECT_LE(curr_peak, prev_peak + 1e-5)
            << "Peak increased at step " << i
            << ": " << prev_peak << " -> " << curr_peak;
        prev_peak = curr_peak;
    }
}

// ── 2. Total heat is conserved ────────────────────────────────────────────────
// With homogeneous Dirichlet BCs the field decays, but in an ideal forward-
// Euler step the interior sum should decrease only due to flux out of the
// domain — not spontaneously gain energy.
// We check the sum never *increases* (no energy creation).
TEST_F(HeatSolver3DTest, TotalHeatNeverIncreases) {
    Config cfg = make_cfg();
    auto solver = make_solver(cfg);

    HostGrid f(solver->dims());
    solver->download(f);
    double prev_sum = total_heat(f);

    for (int i = 0; i < cfg.steps; ++i) {
        solver->step();
        solver->download(f);
        const double curr_sum = total_heat(f);
        EXPECT_LE(curr_sum, prev_sum + 1e-3)
            << "Total heat increased at step " << i;
        prev_sum = curr_sum;
    }
}

// ── 3. Solution stays non-negative ───────────────────────────────────────────
// A Gaussian IC with α > 0 should never produce negative temperatures.
TEST_F(HeatSolver3DTest, SolutionNonNegative) {
    Config cfg = make_cfg();
    cfg.steps = 30;
    auto solver = make_solver(cfg);
    solver->run(cfg.steps);

    HostGrid f(solver->dims());
    solver->download(f);

    const auto& d = solver->dims();
    const int R = STENCIL_RADIUS;
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx - R; ++ix) {
        EXPECT_GE(static_cast<double>(f.at(ix, iy, iz)), -1e-6)
            << "Negative temperature at ("
            << ix << "," << iy << "," << iz << ")";
    }
}

// ── 4. Gaussian width grows as σ²(t) = σ²(0) + 2α·t ─────────────────────────
// The analytical solution for heat diffusion of a 3D Gaussian is another
// Gaussian with σ²(t) = σ²(0) + 2αt.
// We estimate σ² numerically and compare to the analytical prediction.
TEST_F(HeatSolver3DTest, GaussianWidthGrowsAnalytically) {
    Config cfg = make_cfg(64);
    cfg.steps       = 40;
    cfg.diffusivity = 1.0e-2f;
    cfg.pulse_sigma = 6.0f;
    auto solver = make_solver(cfg);

    const GridDims& d = solver->dims();
    const real_t alpha = cfg.diffusivity;
    const real_t dt    = d.dt;

    // Measure σ² at t=0
    HostGrid f(d);
    solver->download(f);

    auto measure_sigma2 = [&](const HostGrid& g) -> double {
        double total = 0.0, weighted = 0.0;
        const double cx = d.nx * 0.5, cy = d.ny * 0.5, cz = d.nz * 0.5;
        for (std::size_t iz = 0; iz < d.nz; ++iz)
        for (std::size_t iy = 0; iy < d.ny; ++iy)
        for (std::size_t ix = 0; ix < d.nx; ++ix) {
            const double v = static_cast<double>(g.at(ix, iy, iz));
            const double r2 = (ix-cx)*(ix-cx) + (iy-cy)*(iy-cy) + (iz-cz)*(iz-cz);
            total    += v;
            weighted += v * r2;
        }
        return (total > 1e-12) ? weighted / total : 0.0;
    };

    const double sigma2_0 = measure_sigma2(f);

    solver->run(cfg.steps);
    solver->download(f);
    const double sigma2_t = measure_sigma2(f);

    // Analytical: σ²(t) = σ²(0) + 2α·(steps·dt)
    // Factor of 3 because variance in r² = variance_x + variance_y + variance_z
    const double t        = static_cast<double>(cfg.steps) * static_cast<double>(dt);
    const double expected = sigma2_0 + 6.0 * static_cast<double>(alpha) * t;

    // Allow 20% error (finite-domain effects, discrete approximation)
    EXPECT_NEAR(sigma2_t, expected, expected * 0.20)
        << "Gaussian spread mismatch: measured=" << sigma2_t
        << " expected=" << expected;
}

// ── 5. Spatial symmetry preserved ────────────────────────────────────────────
TEST_F(HeatSolver3DTest, SymmetryPreserved) {
    Config cfg = make_cfg(48);
    cfg.steps = 10;
    auto solver = make_solver(cfg);
    solver->run(cfg.steps);

    HostGrid f(solver->dims());
    solver->download(f);
    const auto& d = solver->dims();
    const int R = STENCIL_RADIUS;

    double max_asym = 0.0;
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx / 2; ++ix) {
        max_asym = std::max(max_asym, std::abs(
            static_cast<double>(f.at(ix, iy, iz)) -
            static_cast<double>(f.at(d.nx - 1 - ix, iy, iz))));
    }
    EXPECT_LT(max_asym, 1e-5) << "Symmetry broken: " << max_asym;
}
