/**
 * tests/integration/test_kernel_correctness.cpp
 *
 * Cross-validates the naive and tiled kernels against each other and
 * against a reference CPU implementation.
 * Label: "gpu" — requires a CUDA device.
 *
 * Tests:
 *   1. Single-step agreement: naive vs CPU reference (L∞ < ε)
 *   2. Single-step agreement: naive vs tiled (should be bit-identical)
 *   3. Multi-step drift:      naive vs tiled stays < ε after N steps
 *   4. Stencil coefficient correctness: flat field gives zero Laplacian
 *   5. Linear field gives zero Laplacian (∇²(ax+by+cz) = 0)
 */

#include "stencil/config.hpp"
#include "stencil/grid.hpp"
#include "stencil/solver.hpp"
#include "stencil/types.hpp"

#include <gtest/gtest.h>
using namespace stencil;
#include <cmath>



// ── Helpers ───────────────────────────────────────────────────────────────────
static Config correctness_cfg(std::size_t N = 32) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = N;
    cfg.steps        = 1;
    cfg.warmup_steps = 0;
    cfg.physics      = Physics::WavePropagation;
    cfg.velocity     = 1500.0f;
    cfg.dx = cfg.dy  = cfg.dz = 10.0f;
    cfg.pulse_cx = cfg.pulse_cy = cfg.pulse_cz = 0.5f;
    cfg.pulse_sigma = 4.0f;
    cfg.verbose      = false;
    return cfg;
}

// Reference CPU stencil: single step of the wave equation using
// the same FD coefficients, run without any GPU involvement.
// Used as ground truth for the GPU naive kernel.
static void cpu_reference_step(
    const HostGrid& cur, const HostGrid& prev,
    const HostGrid& vel2, HostGrid& next)
{
    const auto& d = cur.dims;
    const int R  = STENCIL_RADIUS;
    const int NX = static_cast<int>(d.nx);
    const int NY = static_cast<int>(d.ny);
    const int NZ = static_cast<int>(d.nz);

    FDCoeffs<STENCIL_RADIUS> fd;
    const real_t inv_dx2 = real_t{1} / (d.dx * d.dx);
    const real_t inv_dy2 = real_t{1} / (d.dy * d.dy);
    const real_t inv_dz2 = real_t{1} / (d.dz * d.dz);
    const real_t dt2     = d.dt * d.dt;

    for (int iz = R; iz < NZ - R; ++iz)
    for (int iy = R; iy < NY - R; ++iy)
    for (int ix = R; ix < NX - R; ++ix) {
        const std::size_t c  = d.idx(ix, iy, iz);
        const std::size_t sy = d.nx;
        const std::size_t sz = d.nx * d.ny;

        real_t lap_x = fd.c0 * cur.data[c] * inv_dx2;
        real_t lap_y = fd.c0 * cur.data[c] * inv_dy2;
        real_t lap_z = fd.c0 * cur.data[c] * inv_dz2;
        for (int r = 1; r <= R; ++r) {
            lap_x += fd.c[r-1] * (cur.data[c+r]      + cur.data[c-r])      * inv_dx2;
            lap_y += fd.c[r-1] * (cur.data[c+r*sy]   + cur.data[c-r*sy])   * inv_dy2;
            lap_z += fd.c[r-1] * (cur.data[c+r*sz]   + cur.data[c-r*sz])   * inv_dz2;
        }
        next.data[c] = real_t{2} * cur.data[c]
                     - prev.data[c]
                     + dt2 * vel2.data[c] * (lap_x + lap_y + lap_z);
    }
}

// ── 1. Naive kernel matches CPU reference ─────────────────────────────────────
TEST(KernelCorrectness, NaiveMatchesCPUReference) {
    Config cfg = correctness_cfg(32);
    cfg.variant = KernelVariant::Naive;

    // Run GPU solver one step
    auto solver = make_solver(cfg);
    solver->step();
    HostGrid gpu_out(solver->dims());
    solver->download(gpu_out);

    // Reconstruct IC and run CPU reference
    const GridDims d = cfg.make_grid();
    HostGrid h_cur(d), h_prev(d), h_vel(d), cpu_out(d);
    init_gaussian(h_cur,  cfg.pulse_cx, cfg.pulse_cy, cfg.pulse_cz, cfg.pulse_sigma);
    init_gaussian(h_prev, cfg.pulse_cx, cfg.pulse_cy, cfg.pulse_cz, cfg.pulse_sigma);
    init_constant_velocity(h_vel, cfg.velocity);
    cpu_reference_step(h_cur, h_prev, h_vel, cpu_out);

    // Compare interior only (halos are zero by construction)
    const int R = STENCIL_RADIUS;
    double max_err = 0.0;
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx - R; ++ix) {
        max_err = std::max(max_err, std::abs(
            static_cast<double>(gpu_out.at(ix, iy, iz)) -
            static_cast<double>(cpu_out.at(ix, iy, iz))));
    }

    // FP32 single-step: expect L∞ < 1e-4 (relative to peak ~1)
    EXPECT_LT(max_err, 1e-4)
        << "Naive GPU vs CPU reference L∞ error = " << max_err;
}

// ── 2. Naive vs tiled: single step bit-identical ─────────────────────────────
// Both kernels use the same coefficients and arithmetic — results should
// be identical up to floating-point rounding (L∞ < machine epsilon * peak).
TEST(KernelCorrectness, NaiveAndTiledAgreeSingleStep) {
    Config cfg = correctness_cfg(32);

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
    // Allow small ULP differences from shared-memory reduction order
    EXPECT_LT(err, 1e-5)
        << "Naive vs Tiled L∞ error after 1 step = " << err;
}

// ── 3. Naive vs tiled: multi-step drift stays small ──────────────────────────
TEST(KernelCorrectness, NaiveAndTiledAgreeMultiStep) {
    Config cfg = correctness_cfg(32);
    const int N = 20;

    cfg.variant = KernelVariant::Naive;
    auto s_naive = make_solver(cfg);

    cfg.variant = KernelVariant::Tiled;
    auto s_tiled = make_solver(cfg);

    for (int i = 0; i < N; ++i) {
        s_naive->step();
        s_tiled->step();
    }

    HostGrid f_naive(s_naive->dims()), f_tiled(s_tiled->dims());
    s_naive->download(f_naive);
    s_tiled->download(f_tiled);

    const double err = l2_error(f_naive, f_tiled);
    EXPECT_LT(err, 1e-4)
        << "Naive vs Tiled L2 error after " << N << " steps = " << err;
}

// ── 4. Flat field gives zero output (zero Laplacian) ─────────────────────────
// ∇²(constant) = 0, so p_next = 2*C - C + 0 = C for any constant C.
// We verify that a uniform field is an exact fixed point.
TEST(KernelCorrectness, FlatFieldIsFixedPoint) {
    Config cfg = correctness_cfg(24);
    // Override init: use constant IC instead of Gaussian.
    // We do this by running with near-zero pulse sigma so the field
    // is essentially uniform far from centre.
    // (A proper constant IC would need a custom init path — tested via
    //  the CPU reference instead.)

    const GridDims d = cfg.make_grid();
    HostGrid h_cur(d, real_t{1.0f});
    HostGrid h_prev(d, real_t{1.0f});
    HostGrid h_vel(d, real_t{1.0f});
    HostGrid h_next(d, real_t{0.0f});

    // CPU reference: flat field → next = 2·1 - 1 + dt²·1·0 = 1
    cpu_reference_step(h_cur, h_prev, h_vel, h_next);

    const int R = STENCIL_RADIUS;
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx - R; ++ix) {
        EXPECT_NEAR(static_cast<double>(h_next.at(ix, iy, iz)), 1.0, 1e-5)
            << "Flat field not preserved at ("
            << ix << "," << iy << "," << iz << ")";
    }
}

// ── 5. Linear field gives zero Laplacian ─────────────────────────────────────
// ∇²(ax + by + cz) = 0 exactly.
// p_next = 2·p_cur - p_prev + dt²·v²·0 = p_cur  (since p_cur == p_prev at t=0)
TEST(KernelCorrectness, LinearFieldZeroLaplacian) {
    Config cfg = correctness_cfg(24);
    const GridDims d = cfg.make_grid();

    HostGrid h_cur(d), h_prev(d), h_vel(d, real_t{1.0f}), h_next(d);
    for (std::size_t iz = 0; iz < d.nz; ++iz)
    for (std::size_t iy = 0; iy < d.ny; ++iy)
    for (std::size_t ix = 0; ix < d.nx; ++ix) {
        const real_t v = real_t{0.1f} * ix
                       + real_t{0.2f} * iy
                       + real_t{0.3f} * iz;
        h_cur.at(ix, iy, iz) = h_prev.at(ix, iy, iz) = v;
    }

    cpu_reference_step(h_cur, h_prev, h_vel, h_next);

    const int R = STENCIL_RADIUS;
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx - R; ++ix) {
        const double expected = static_cast<double>(h_cur.at(ix, iy, iz));
        EXPECT_NEAR(static_cast<double>(h_next.at(ix, iy, iz)),
                    expected, std::abs(expected) * 1e-4 + 1e-6)
            << "Linear field Laplacian not zero at ("
            << ix << "," << iy << "," << iz << ")";
    }
}
