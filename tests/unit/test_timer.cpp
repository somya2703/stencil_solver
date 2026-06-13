/**
 * tests/unit/test_timer.cpp
 *
 * Tests for CpuTimer and PerfResult helpers.
 * GPU timer is not tested here (requires a device); it is exercised
 * in integration tests.
 */

#include "stencil/timer.hpp"
#include <gtest/gtest.h>
using namespace stencil;
#include <chrono>
#include <thread>



TEST(CpuTimer, ElapsedPositive) {
    CpuTimer t;
    t.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    t.stop();
    EXPECT_GT(t.elapsed_ms(), 5.0);   // at least 5ms
    EXPECT_LT(t.elapsed_ms(), 500.0); // not unreasonably long
}

TEST(CpuTimer, ElapsedZero) {
    CpuTimer t;
    t.start();
    t.stop();
    // Immediate start/stop should be very short
    EXPECT_GE(t.elapsed_ms(), 0.0);
    EXPECT_LT(t.elapsed_ms(), 5.0);
}

TEST(PerfResult, DefaultValues) {
    PerfResult r;
    EXPECT_EQ(r.label, "");
    EXPECT_DOUBLE_EQ(r.time_ms,       0.0);
    EXPECT_DOUBLE_EQ(r.bandwidth_gbs, 0.0);
    EXPECT_DOUBLE_EQ(r.gflops,        0.0);
}

TEST(BandwidthFlops, ScalesWithGrid) {
    GridDims g1; g1.nx = 64;  g1.ny = 64;  g1.nz = 64;
    GridDims g2; g2.nx = 128; g2.ny = 128; g2.nz = 128;

    // Doubling each dimension should 8× the bandwidth and FLOP counts
    const double bw1 = bandwidth_bytes_per_step(g1);
    const double bw2 = bandwidth_bytes_per_step(g2);
    const double fl1 = flops_per_step(g1);
    const double fl2 = flops_per_step(g2);

    EXPECT_NEAR(bw2 / bw1, 8.0, 2.5);
    EXPECT_NEAR(fl2 / fl1, 8.0, 2.5);
}

TEST(BandwidthFlops, PositiveValues) {
    GridDims g; g.nx = 32; g.ny = 32; g.nz = 32;
    EXPECT_GT(bandwidth_bytes_per_step(g), 0.0);
    EXPECT_GT(flops_per_step(g),           0.0);
}
