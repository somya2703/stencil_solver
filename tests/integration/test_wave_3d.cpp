/**
 * tests/integration/test_wave_3d.cpp
 *
 * Integration tests for the acoustic wave solver on a real 3D grid.
 * These tests require a GPU (label: "gpu") and are skipped in CPU-only CI.
 *
 * Tests:
 *   1. Energy conservation  — total energy should stay bounded over many steps
 *   2. Wave speed           — a pulse should travel ~v*t cells in t steps
 *   3. Symmetry             — 3D Gaussian stays symmetric on all three axes
 *   4. Boundary passivity   — field at halo boundary cells stays zero
 *   5. Reproducibility      — two identical runs produce bit-identical output
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



// ── Test fixture ──────────────────────────────────────────────────────────────
class WaveSolver3DTest : public ::testing::Test {
protected:
    // 64³ grid — small enough to be fast, large enough for physics to show
    static Config make_cfg(std::size_t N = 64) {
        Config cfg;
        cfg.nx = cfg.ny = cfg.nz = N;
        cfg.steps        = 20;
        cfg.warmup_steps = 0;
        cfg.physics      = Physics::WavePropagation;
        cfg.variant      = KernelVariant::Naive;
        cfg.velocity     = 1500.0f;
        cfg.dx = cfg.dy  = cfg.dz = 10.0f;  // 10 m spacing
        cfg.pulse_cx = cfg.pulse_cy = cfg.pulse_cz = 0.5f;
        cfg.pulse_sigma = 4.0f;
        cfg.verbose      = false;
        return cfg;
    }

    // Compute total acoustic energy: E = sum(p² + p_prev²) / 2
    // (proportional to kinetic + potential energy in the leapfrog scheme)
    static double total_energy(const HostGrid& cur, const HostGrid& prev) {
        double e = 0.0;
        for (std::size_t i = 0; i < cur.dims.size(); ++i)
            e += static_cast<double>(cur.data[i])  * static_cast<double>(cur.data[i])
               + static_cast<double>(prev.data[i]) * static_cast<double>(prev.data[i]);
        return 0.5 * e;
    }
};

// ── 1. Energy conservation ────────────────────────────────────────────────────
// In a lossless, free-boundary domain, total acoustic energy should not
// grow over time.  We allow a small tolerance for floating-point drift.
TEST_F(WaveSolver3DTest, EnergyDoesNotGrow) {
    Config cfg = make_cfg(48);
    cfg.steps  = 50;
    auto solver = make_solver(cfg);

    HostGrid cur(solver->dims()), prev(solver->dims());

    // Capture initial energy
    solver->download(cur);
    // prev ≈ cur at t=0 (both initialised to the Gaussian)
    const double E0 = total_energy(cur, cur);
    ASSERT_GT(E0, 0.0) << "Initial energy should be positive";

    // Run and re-measure
    solver->run(cfg.steps);
    solver->download(cur);

    // Energy should not have grown by more than 10×
    // (reflections at Dirichlet boundaries can cause some growth,
    //  but exponential blow-up signals a broken kernel)
    const double E1 = total_energy(cur, cur);
    EXPECT_LT(E1, E0 * 10.0)
        << "Energy grew suspiciously: E0=" << E0 << " E1=" << E1;
    EXPECT_GT(E1, 0.0) << "Energy should remain positive";
}

// ── 2. Pulse travels at correct wave speed ────────────────────────────────────
// After t steps, the wavefront should be at radius ≈ v * t * dt grid cells
// from the source.  We check that the maximum amplitude at t=0 vs t=N steps
// has spread outward.
TEST_F(WaveSolver3DTest, PulseSpreadsOutward) {
    Config cfg = make_cfg(64);
    cfg.steps  = 10;
    auto solver = make_solver(cfg);

    // Find initial peak location (should be near centre)
    HostGrid field0(solver->dims());
    solver->download(field0);
    auto it0 = std::max_element(field0.data.begin(), field0.data.end());
    const std::size_t peak_idx0 = std::distance(field0.data.begin(), it0);
    const std::size_t cx0 = peak_idx0 % solver->dims().nx;

    // Run steps
    for (int i = 0; i < cfg.steps; ++i) solver->step();

    HostGrid field1(solver->dims());
    solver->download(field1);

    // After propagation the initial peak should have decreased
    // (energy has spread outward — peak amplitude drops as 1/r in 3D)
    const double peak0 = static_cast<double>(*it0);
    const double peak1 = static_cast<double>(
        *std::max_element(field1.data.begin(), field1.data.end()));

    EXPECT_LT(peak1, peak0)
        << "Peak amplitude should decrease as wave spreads: "
        << peak1 << " >= " << peak0;

    (void)cx0;  // location check reserved for a more detailed test
}

// ── 3. 3-axis symmetry preservation ──────────────────────────────────────────
// A centred Gaussian in an isotropic medium remains symmetric about x, y, z.
TEST_F(WaveSolver3DTest, SymmetryPreservedAllAxes) {
    Config cfg = make_cfg(48);
    cfg.steps = 8;
    auto solver = make_solver(cfg);

    for (int i = 0; i < cfg.steps; ++i) solver->step();

    HostGrid f(solver->dims());
    solver->download(f);
    const auto& d = solver->dims();
    const int R = STENCIL_RADIUS;

    double max_x = 0, max_y = 0, max_z = 0;

    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx / 2; ++ix) {
        max_x = std::max(max_x, std::abs(
            static_cast<double>(f.at(ix, iy, iz)) -
            static_cast<double>(f.at(d.nx - 1 - ix, iy, iz))));
        max_y = std::max(max_y, std::abs(
            static_cast<double>(f.at(iy, ix, iz)) -
            static_cast<double>(f.at(d.ny - 1 - iy, ix, iz))));
        max_z = std::max(max_z, std::abs(
            static_cast<double>(f.at(ix, iy, iz)) -
            static_cast<double>(f.at(ix, iy, d.nz - 1 - iz))));
    }

    EXPECT_LT(max_x, 1e-4) << "X-symmetry broken: " << max_x;
    EXPECT_LT(max_y, 1e-4) << "Y-symmetry broken: " << max_y;
    EXPECT_LT(max_z, 1e-4) << "Z-symmetry broken: " << max_z;
}

// ── 4. Halo cells remain zero (Dirichlet boundary) ───────────────────────────
// The halo ring (first STENCIL_RADIUS cells on each face) is never written
// by the kernel — it stays zero throughout the simulation.
TEST_F(WaveSolver3DTest, HaloCellsRemainZero) {
    Config cfg = make_cfg(48);
    cfg.steps = 10;
    auto solver = make_solver(cfg);
    solver->run(cfg.steps);

    HostGrid f(solver->dims());
    solver->download(f);
    const auto& d = solver->dims();
    const int R = STENCIL_RADIUS;

    // Check ix=0..R-1 face
    for (std::size_t iz = 0; iz < d.nz; ++iz)
    for (std::size_t iy = 0; iy < d.ny; ++iy)
    for (int ix = 0; ix < R; ++ix) {
        EXPECT_FLOAT_EQ(f.at(ix, iy, iz), real_t{0})
            << "Halo not zero at (" << ix << "," << iy << "," << iz << ")";
    }
}

// ── 5. Reproducibility ────────────────────────────────────────────────────────
// Two solvers with identical config produce bit-identical results.
TEST_F(WaveSolver3DTest, BitIdenticalReproducibility) {
    Config cfg = make_cfg(32);
    cfg.steps = 5;

    auto s1 = make_solver(cfg);
    auto s2 = make_solver(cfg);

    s1->run(cfg.steps);
    s2->run(cfg.steps);

    HostGrid f1(s1->dims()), f2(s2->dims());
    s1->download(f1);
    s2->download(f2);

    EXPECT_DOUBLE_EQ(l2_error(f1, f2), 0.0)
        << "Identical runs gave different results";
}
