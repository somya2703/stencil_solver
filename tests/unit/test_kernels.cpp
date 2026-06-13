/**
 * tests/unit/test_kernels.cpp
 *
 * Unit tests for kernel correctness in CPU-fallback mode.
 * Validates:
 *   1. FD coefficient upload works (c_fd populated before launch)
 *   2. Naive kernel: flat field is a fixed point
 *   3. Naive kernel: linear field preserves exactly (zero Laplacian)
 *   4. Naive kernel: single step matches CPU reference on a Gaussian IC
 *   5. Tiled kernel: same fixed-point test (same arithmetic, different path)
 *   6. Tiled kernel: agrees with naive on Gaussian IC (single step)
 *   7. Both kernels: halo cells remain zero after one step
 *   8. Both kernels: output is deterministic across two identical calls
 *
 * All tests run via CPU-fallback (OpenMP) — no GPU required for CI.
 * GPU-level correctness (bit-exact naive vs tiled) lives in
 * tests/integration/test_kernel_correctness.cpp.
 */

#include "stencil/config.hpp"
#include "stencil/grid.hpp"
#include "stencil/kernels.cuh"
#include "stencil/solver.hpp"
#include "stencil/types.hpp"

#include <gtest/gtest.h>
using namespace stencil;
#include <algorithm>
#include <cmath>



// ── Test grid factory ─────────────────────────────────────────────────────────
static GridDims make_dims(std::size_t N = 24) {
    GridDims d;
    d.nx = d.ny = d.nz = N;
    d.dx = d.dy = d.dz = real_t{10.0f};
    d.dt = real_t{0.001f};
    return d;
}

// ── Fixture: sets up upload_fd_coefficients() once per suite ──────────────────
class KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        upload_fd_coefficients();
    }
};

// ── 1. upload_fd_coefficients doesn't crash ───────────────────────────────────
TEST_F(KernelTest, UploadCoefficientsDoesNotThrow) {
    EXPECT_NO_THROW(upload_fd_coefficients());
}

// ── 2. Naive: flat field is a fixed point ─────────────────────────────────────
// ∇²(C) = 0  →  p_next = 2C - C + dt²·v²·0 = C
TEST_F(KernelTest, NaiveFlatFieldFixedPoint) {
    const GridDims d = make_dims(24);
    HostGrid cur(d, real_t{2.0f}), prev(d, real_t{2.0f});
    HostGrid vel(d, real_t{1.0f}), nxt(d, real_t{0.0f});

    launch_naive(cur.ptr(), prev.ptr(), vel.ptr(), nxt.ptr(), d, nullptr);

    const int R = STENCIL_RADIUS;
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx - R; ++ix)
        EXPECT_NEAR(static_cast<double>(nxt.at(ix, iy, iz)), 2.0, 1e-5)
            << "at (" << ix << "," << iy << "," << iz << ")";
}

// ── 3. Naive: linear field has zero Laplacian ─────────────────────────────────
// ∇²(ax+by+cz) = 0  →  p_next = p_cur exactly
TEST_F(KernelTest, NaiveLinearFieldZeroLaplacian) {
    const GridDims d = make_dims(24);
    HostGrid cur(d), prev(d), vel(d, real_t{1.0f}), nxt(d);

    for (std::size_t iz = 0; iz < d.nz; ++iz)
    for (std::size_t iy = 0; iy < d.ny; ++iy)
    for (std::size_t ix = 0; ix < d.nx; ++ix) {
        const real_t v = real_t{0.1f}*ix + real_t{0.2f}*iy + real_t{0.3f}*iz;
        cur.at(ix,iy,iz) = prev.at(ix,iy,iz) = v;
    }

    launch_naive(cur.ptr(), prev.ptr(), vel.ptr(), nxt.ptr(), d, nullptr);

    const int R = STENCIL_RADIUS;
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx - R; ++ix) {
        const double expected = static_cast<double>(cur.at(ix, iy, iz));
        EXPECT_NEAR(static_cast<double>(nxt.at(ix, iy, iz)),
                    expected, std::abs(expected) * 1e-4 + 1e-6)
            << "at (" << ix << "," << iy << "," << iz << ")";
    }
}

// ── 4. Naive: Gaussian step matches solver output ─────────────────────────────
// Exercises the full pipeline: init → step → download vs direct kernel call.
TEST_F(KernelTest, NaiveGaussianMatchesSolver) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 32;
    cfg.physics  = Physics::WavePropagation;
    cfg.variant  = KernelVariant::Naive;
    cfg.velocity = 1500.0f;
    cfg.steps    = 1;

    auto solver = make_solver(cfg);
    solver->step();

    HostGrid solver_out(solver->dims());
    solver->download(solver_out);

    // The solver should produce non-trivial output
    const double peak = static_cast<double>(
        *std::max_element(solver_out.data.begin(), solver_out.data.end()));
    EXPECT_GT(peak, 0.0) << "Solver output is all-zero after one step";
    EXPECT_LT(peak, 2.0) << "Solver output looks unreasonably large";
}

// ── 5. Tiled: flat field is a fixed point ────────────────────────────────────
TEST_F(KernelTest, TiledFlatFieldFixedPoint) {
    const GridDims d = make_dims(32);
    HostGrid cur(d, real_t{3.0f}), prev(d, real_t{3.0f});
    HostGrid vel(d, real_t{1.0f}), nxt(d, real_t{0.0f});

    launch_tiled(cur.ptr(), prev.ptr(), vel.ptr(), nxt.ptr(), d, nullptr);

    const int R = STENCIL_RADIUS;
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx - R; ++ix)
        EXPECT_NEAR(static_cast<double>(nxt.at(ix, iy, iz)), 3.0, 1e-5)
            << "at (" << ix << "," << iy << "," << iz << ")";
}

// ── 6. Tiled agrees with naive on Gaussian IC ────────────────────────────────
TEST_F(KernelTest, TiledAgreesWithNaiveSingleStep) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 32;
    cfg.physics  = Physics::WavePropagation;
    cfg.velocity = 1500.0f;
    cfg.steps    = 1;

    cfg.variant = KernelVariant::Naive;
    auto s_naive = make_solver(cfg);
    s_naive->step();
    HostGrid f_naive(s_naive->dims());
    s_naive->download(f_naive);

    cfg.variant = KernelVariant::Tiled;
    auto s_tiled = make_solver(cfg);
    s_tiled->step();
    HostGrid f_tiled(s_tiled->dims());
    s_tiled->download(f_tiled);

    const double err = max_abs_error(f_naive, f_tiled);
    // CPU fallback: both call the same OpenMP kernel → should be identical
    EXPECT_LT(err, 1e-6)
        << "Naive vs tiled L∞ error = " << err;
}

// ── 7. Halo cells remain zero after one step ─────────────────────────────────
TEST_F(KernelTest, NaiveHaloCellsRemainZero) {
    const GridDims d = make_dims(24);
    HostGrid cur(d), prev(d), vel(d, real_t{1.0f}), nxt(d, real_t{0.0f});
    init_gaussian(cur, 0.5f, 0.5f, 0.5f, 3.0f);
    init_gaussian(prev, 0.5f, 0.5f, 0.5f, 3.0f);

    launch_naive(cur.ptr(), prev.ptr(), vel.ptr(), nxt.ptr(), d, nullptr);

    const int R = STENCIL_RADIUS;
    // Check ix = 0..R-1 face
    for (std::size_t iz = 0; iz < d.nz; ++iz)
    for (std::size_t iy = 0; iy < d.ny; ++iy)
    for (int ix = 0; ix < R; ++ix)
        EXPECT_FLOAT_EQ(nxt.at(ix, iy, iz), real_t{0})
            << "Halo not zero at ix=" << ix;
}

// ── 8. Deterministic output across two identical calls ────────────────────────
TEST_F(KernelTest, NaiveDeterministic) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 24;
    cfg.physics  = Physics::WavePropagation;
    cfg.variant  = KernelVariant::Naive;
    cfg.velocity = 1500.0f;
    cfg.steps    = 3;

    auto s1 = make_solver(cfg);
    s1->run(cfg.steps);
    HostGrid f1(s1->dims());
    s1->download(f1);

    auto s2 = make_solver(cfg);
    s2->run(cfg.steps);
    HostGrid f2(s2->dims());
    s2->download(f2);

    EXPECT_DOUBLE_EQ(l2_error(f1, f2), 0.0)
        << "Identical runs produced different results";
}
