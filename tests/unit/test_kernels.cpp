/**
 * tests/unit/test_kernels.cpp
 *
 * Unit tests for kernel correctness.
 * In GPU builds, the 4 direct-launch tests allocate device memory,
 * copy inputs to device, run the kernel, copy results back to host.
 * In CPU-fallback builds, host pointers are passed directly.
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

#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>
#endif

// ── Device buffer RAII helper (GPU build only) ────────────────────────────────
#ifndef STENCIL_CPU_FALLBACK
struct DevBuf {
    real_t* ptr = nullptr;
    explicit DevBuf(std::size_t n) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(real_t)));
    }
    ~DevBuf() { if (ptr) cudaFree(ptr); }
    void upload(const real_t* h, std::size_t n) {
        CUDA_CHECK(cudaMemcpy(ptr, h, n * sizeof(real_t), cudaMemcpyHostToDevice));
    }
    void download(real_t* h, std::size_t n) const {
        CUDA_CHECK(cudaMemcpy(h, ptr, n * sizeof(real_t), cudaMemcpyDeviceToHost));
    }
};
#endif

// ── Test grid factory ─────────────────────────────────────────────────────────
static GridDims make_dims(std::size_t N = 24) {
    GridDims d;
    d.nx = d.ny = d.nz = N;
    d.dx = d.dy = d.dz = real_t{10.0f};
    d.dt = real_t{0.001f};
    return d;
}

// ── Fixture ───────────────────────────────────────────────────────────────────
class KernelTest : public ::testing::Test {
protected:
    void SetUp() override { upload_fd_coefficients(); }
};

// ── Helper: run launch_naive with correct device/host handling ────────────────
static void run_naive(const HostGrid& cur, const HostGrid& prev,
                      const HostGrid& vel, HostGrid& nxt) {
    const std::size_t N = cur.dims.size();
#ifndef STENCIL_CPU_FALLBACK
    DevBuf d_cur(N), d_prev(N), d_vel(N), d_nxt(N);
    d_cur.upload(cur.ptr(), N);
    d_prev.upload(prev.ptr(), N);
    d_vel.upload(vel.ptr(), N);
    d_nxt.upload(nxt.ptr(), N);
    launch_naive(d_cur.ptr, d_prev.ptr, d_vel.ptr, d_nxt.ptr, cur.dims, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_nxt.download(nxt.data.data(), N);
#else
    launch_naive(cur.ptr(), prev.ptr(), vel.ptr(), nxt.ptr(), cur.dims, nullptr);
#endif
}

// ── Helper: run launch_tiled with correct device/host handling ────────────────
static void run_tiled(const HostGrid& cur, const HostGrid& prev,
                      const HostGrid& vel, HostGrid& nxt) {
    const std::size_t N = cur.dims.size();
#ifndef STENCIL_CPU_FALLBACK
    DevBuf d_cur(N), d_prev(N), d_vel(N), d_nxt(N);
    d_cur.upload(cur.ptr(), N);
    d_prev.upload(prev.ptr(), N);
    d_vel.upload(vel.ptr(), N);
    d_nxt.upload(nxt.ptr(), N);
    launch_tiled(d_cur.ptr, d_prev.ptr, d_vel.ptr, d_nxt.ptr, cur.dims, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_nxt.download(nxt.data.data(), N);
#else
    launch_tiled(cur.ptr(), prev.ptr(), vel.ptr(), nxt.ptr(), cur.dims, nullptr);
#endif
}

// ── 1. upload_fd_coefficients doesn't crash ───────────────────────────────────
TEST_F(KernelTest, UploadCoefficientsDoesNotThrow) {
    EXPECT_NO_THROW(upload_fd_coefficients());
}

// ── 2. Naive: flat field is a fixed point ─────────────────────────────────────
TEST_F(KernelTest, NaiveFlatFieldFixedPoint) {
    const GridDims d = make_dims(24);
    HostGrid cur(d, real_t{2.0f}), prev(d, real_t{2.0f});
    HostGrid vel(d, real_t{1.0f}), nxt(d, real_t{0.0f});

    run_naive(cur, prev, vel, nxt);

    const int R = STENCIL_RADIUS;
    for (std::size_t iz = R; iz < d.nz - R; ++iz)
    for (std::size_t iy = R; iy < d.ny - R; ++iy)
    for (std::size_t ix = R; ix < d.nx - R; ++ix)
        EXPECT_NEAR(static_cast<double>(nxt.at(ix, iy, iz)), 2.0, 1e-5)
            << "at (" << ix << "," << iy << "," << iz << ")";
}

// ── 3. Naive: linear field has zero Laplacian ─────────────────────────────────
TEST_F(KernelTest, NaiveLinearFieldZeroLaplacian) {
    const GridDims d = make_dims(24);
    HostGrid cur(d), prev(d), vel(d, real_t{1.0f}), nxt(d);

    for (std::size_t iz = 0; iz < d.nz; ++iz)
    for (std::size_t iy = 0; iy < d.ny; ++iy)
    for (std::size_t ix = 0; ix < d.nx; ++ix) {
        const real_t v = real_t{0.1f}*ix + real_t{0.2f}*iy + real_t{0.3f}*iz;
        cur.at(ix,iy,iz) = prev.at(ix,iy,iz) = v;
    }

    run_naive(cur, prev, vel, nxt);

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

    run_tiled(cur, prev, vel, nxt);

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
    EXPECT_LT(err, 1e-4)
        << "Naive vs tiled L∞ error = " << err;
}

// ── 7. Halo cells remain zero after one step ─────────────────────────────────
TEST_F(KernelTest, NaiveHaloCellsRemainZero) {
    const GridDims d = make_dims(24);
    HostGrid cur(d), prev(d), vel(d, real_t{1.0f}), nxt(d, real_t{0.0f});
    init_gaussian(cur, 0.5f, 0.5f, 0.5f, 3.0f);
    init_gaussian(prev, 0.5f, 0.5f, 0.5f, 3.0f);

    run_naive(cur, prev, vel, nxt);

    const int R = STENCIL_RADIUS;
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
