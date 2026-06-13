/**
 * tests/unit/test_solver_cpu.cpp
 *
 * Tests for Config parsing and the GridDims::make_grid() helper.
 * CPU-only; validates the configuration layer used by all solvers.
 */

#include "stencil/config.hpp"
#include "stencil/grid.hpp"
#include "stencil/types.hpp"
#include <gtest/gtest.h>
using namespace stencil;
#include <cstring>



// ── Config defaults ───────────────────────────────────────────────────────────
TEST(Config, DefaultValues) {
    Config cfg;
    EXPECT_EQ(cfg.nx, 256u);
    EXPECT_EQ(cfg.ny, 256u);
    EXPECT_EQ(cfg.nz, 256u);
    EXPECT_EQ(cfg.steps, 100);
    EXPECT_EQ(cfg.physics, Physics::WavePropagation);
    EXPECT_EQ(cfg.variant, KernelVariant::Naive);
}

// ── CLI parsing ───────────────────────────────────────────────────────────────
TEST(Config, ParseGridSize) {
    const char* argv[] = {"prog", "--nx", "512", "--ny", "128", "--nz", "64"};
    int argc = 7;
    Config cfg = parse_args(argc, const_cast<char**>(argv));
    EXPECT_EQ(cfg.nx, 512u);
    EXPECT_EQ(cfg.ny, 128u);
    EXPECT_EQ(cfg.nz, 64u);
}

TEST(Config, ParseSteps) {
    const char* argv[] = {"prog", "--steps", "200", "--warmup", "5"};
    int argc = 5;
    Config cfg = parse_args(argc, const_cast<char**>(argv));
    EXPECT_EQ(cfg.steps,        200);
    EXPECT_EQ(cfg.warmup_steps, 5);
}

TEST(Config, ParsePhysicsHeat) {
    const char* argv[] = {"prog", "--physics", "heat"};
    int argc = 3;
    Config cfg = parse_args(argc, const_cast<char**>(argv));
    EXPECT_EQ(cfg.physics, Physics::HeatDiffusion);
}

TEST(Config, ParseKernelTiled) {
    const char* argv[] = {"prog", "--kernel", "tiled"};
    int argc = 3;
    Config cfg = parse_args(argc, const_cast<char**>(argv));
    EXPECT_EQ(cfg.variant, KernelVariant::Tiled);
}

TEST(Config, ParseVerbose) {
    const char* argv[] = {"prog", "--verbose"};
    int argc = 2;
    Config cfg = parse_args(argc, const_cast<char**>(argv));
    EXPECT_TRUE(cfg.verbose);
}

TEST(Config, ParseOutputPath) {
    const char* argv[] = {"prog", "--output", "results/out.json"};
    int argc = 3;
    Config cfg = parse_args(argc, const_cast<char**>(argv));
    EXPECT_EQ(cfg.output_path, "results/out.json");
}

TEST(Config, UnknownFlagThrows) {
    const char* argv[] = {"prog", "--unknown"};
    int argc = 2;
    EXPECT_THROW(parse_args(argc, const_cast<char**>(argv)), std::runtime_error);
}

TEST(Config, UnknownPhysicsThrows) {
    const char* argv[] = {"prog", "--physics", "quantum"};
    int argc = 3;
    EXPECT_THROW(parse_args(argc, const_cast<char**>(argv)), std::runtime_error);
}

// ── make_grid ─────────────────────────────────────────────────────────────────
TEST(Config, MakeGridDimensions) {
    Config cfg;
    cfg.nx = 64; cfg.ny = 128; cfg.nz = 32;
    cfg.dx = 5.0f; cfg.dy = 5.0f; cfg.dz = 5.0f;
    const GridDims g = cfg.make_grid();
    EXPECT_EQ(g.nx, 64u);
    EXPECT_EQ(g.ny, 128u);
    EXPECT_EQ(g.nz, 32u);
    EXPECT_FLOAT_EQ(g.dx, 5.0f);
}

TEST(Config, MakeGridCFLStable) {
    Config cfg;
    cfg.nx = 64; cfg.ny = 64; cfg.nz = 64;
    cfg.dx = 10.0f; cfg.dy = 10.0f; cfg.dz = 10.0f;
    cfg.velocity = 1500.0f;
    const GridDims g = cfg.make_grid();
    // dt should be CFL_LIMIT * dx / v
    const real_t expected_dt = GridDims::CFL_LIMIT * cfg.dx / cfg.velocity;
    EXPECT_FLOAT_EQ(g.dt, expected_dt);
    // Should pass CFL validation
    EXPECT_NO_THROW(validate_grid(g, cfg.velocity));
}
