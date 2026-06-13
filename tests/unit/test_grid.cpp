/**
 * tests/unit/test_grid.cpp
 *
 * Tests for GridDims, HostGrid, and grid utility functions.
 * All CPU-only — no GPU required.
 */

#include "stencil/grid.hpp"
#include "stencil/types.hpp"
#include <gtest/gtest.h>
using namespace stencil;
#include <cmath>



// ── GridDims ──────────────────────────────────────────────────────────────────
TEST(GridDims, SizeAndIndex) {
    GridDims g;
    g.nx = 10; g.ny = 20; g.nz = 30;
    EXPECT_EQ(g.size(), 10u * 20u * 30u);
    EXPECT_EQ(g.idx(0, 0, 0), 0u);
    EXPECT_EQ(g.idx(1, 0, 0), 1u);
    EXPECT_EQ(g.idx(0, 1, 0), 10u);
    EXPECT_EQ(g.idx(0, 0, 1), 200u);
    EXPECT_EQ(g.idx(9, 19, 29), g.size() - 1);
}

TEST(GridDims, InteriorSize) {
    GridDims g;
    g.nx = 64; g.ny = 64; g.nz = 64;
    const int R = STENCIL_RADIUS;
    EXPECT_EQ(g.interior_nx(), 64u - 2u * R);
    EXPECT_EQ(g.interior_size(),
              (64u - 2u*R) * (64u - 2u*R) * (64u - 2u*R));
}

// ── HostGrid ──────────────────────────────────────────────────────────────────
TEST(HostGrid, ConstructAndFill) {
    GridDims g; g.nx = 4; g.ny = 4; g.nz = 4;
    HostGrid h(g, real_t{3.14f});
    EXPECT_EQ(h.data.size(), 64u);
    for (auto v : h.data)
        EXPECT_FLOAT_EQ(v, real_t{3.14f});
}

TEST(HostGrid, AtAccessor) {
    GridDims g; g.nx = 4; g.ny = 5; g.nz = 6;
    HostGrid h(g);
    h.at(1, 2, 3) = real_t{99};
    EXPECT_FLOAT_EQ(h.data[g.idx(1, 2, 3)], real_t{99});
}

// ── Initialisation helpers ────────────────────────────────────────────────────
TEST(GridInit, GaussianPeak) {
    GridDims g; g.nx = 32; g.ny = 32; g.nz = 32;
    HostGrid h(g);
    init_gaussian(h, 0.5f, 0.5f, 0.5f, 3.0f);

    // Peak should be near the centre
    const real_t centre = h.at(16, 16, 16);
    EXPECT_NEAR(static_cast<double>(centre), 1.0, 0.05);

    // Corner should be close to zero
    EXPECT_LT(static_cast<double>(h.at(0, 0, 0)), 0.01);
}

TEST(GridInit, ConstantVelocity) {
    GridDims g; g.nx = 8; g.ny = 8; g.nz = 8;
    HostGrid h(g);
    init_constant_velocity(h, 1500.0f);
    for (auto v : h.data)
        EXPECT_FLOAT_EQ(v, 1500.0f * 1500.0f);
}

TEST(GridInit, LayeredVelocity) {
    GridDims g; g.nx = 4; g.ny = 4; g.nz = 8;
    HostGrid h(g);
    init_layered_velocity(h, 1000.0f, 2000.0f);
    // Top half: v=1000 → v²=1e6
    EXPECT_FLOAT_EQ(h.at(0, 0, 0),  1000.0f * 1000.0f);
    // Bottom half: v=2000 → v²=4e6
    EXPECT_FLOAT_EQ(h.at(0, 0, 7), 2000.0f * 2000.0f);
}

// ── Validation ────────────────────────────────────────────────────────────────
TEST(GridValidation, ValidGrid) {
    GridDims g;
    g.nx = 64; g.ny = 64; g.nz = 64;
    g.dx = 10.0f; g.dy = 10.0f; g.dz = 10.0f;
    g.dt = 0.001f;
    EXPECT_NO_THROW(validate_grid(g, 1500.0f));
}

TEST(GridValidation, TooSmall) {
    GridDims g;
    g.nx = 2; g.ny = 64; g.nz = 64;  // nx too small for stencil radius
    g.dx = 10.0f; g.dy = 10.0f; g.dz = 10.0f; g.dt = 0.001f;
    EXPECT_THROW(validate_grid(g), std::invalid_argument);
}

TEST(GridValidation, CFLViolation) {
    GridDims g;
    g.nx = 64; g.ny = 64; g.nz = 64;
    g.dx = 10.0f; g.dy = 10.0f; g.dz = 10.0f;
    g.dt = 1.0f;  // huge dt → CFL >> 1
    EXPECT_THROW(validate_grid(g, 1500.0f), std::invalid_argument);
}

// ── Error norms ───────────────────────────────────────────────────────────────
TEST(GridError, L2ErrorIdentical) {
    GridDims g; g.nx = 8; g.ny = 8; g.nz = 8;
    HostGrid a(g, real_t{1}), b(g, real_t{1});
    EXPECT_DOUBLE_EQ(l2_error(a, b), 0.0);
}

TEST(GridError, L2ErrorKnown) {
    GridDims g; g.nx = 4; g.ny = 4; g.nz = 4;
    HostGrid a(g, real_t{0}), b(g, real_t{1});
    // Every element differs by 1, so RMS = 1
    EXPECT_NEAR(l2_error(a, b), 1.0, 1e-6);
}

TEST(GridError, MaxAbsError) {
    GridDims g; g.nx = 4; g.ny = 4; g.nz = 4;
    HostGrid a(g, real_t{0}), b(g, real_t{0});
    b.at(2, 2, 2) = real_t{5};
    EXPECT_NEAR(max_abs_error(a, b), 5.0, 1e-6);
}
