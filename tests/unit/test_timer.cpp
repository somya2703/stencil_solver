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

TEST(FormatPerfTable, LabelsEachRowByGridSize) {
    std::vector<PerfResult> results;
    PerfResult a; a.label = "naive"; a.time_ms = 1.403;  a.bandwidth_gbs = 1217.4; a.gflops = 608.7; a.grid_size = 256;
    PerfResult b; b.label = "naive"; b.time_ms = 11.462; b.bandwidth_gbs = 1250.9; b.gflops = 625.5; b.grid_size = 512;
    results.push_back(a);
    results.push_back(b);

    testing::internal::CaptureStdout();
    format_perf_table(results);
    const std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("256"), std::string::npos)
        << "256-cubed row's grid size should appear in the table";
    EXPECT_NE(out.find("512"), std::string::npos)
        << "512-cubed row's grid size should appear in the table";
}

TEST(FormatPerfTable, HasGridColumnNoSpeedupColumn) {
    std::vector<PerfResult> results;
    PerfResult a; a.label = "tiled"; a.time_ms = 2.0; a.bandwidth_gbs = 500.0; a.gflops = 250.0; a.grid_size = 128;
    results.push_back(a);

    testing::internal::CaptureStdout();
    format_perf_table(results);
    const std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("Grid"), std::string::npos)
        << "table header should have a Grid column";
    EXPECT_EQ(out.find("Speedup"), std::string::npos)
        << "the meaningless cross-grid-size Speedup column must not return";
}

TEST(FormatPerfTable, EmptyResultsProducesNoOutput) {
    std::vector<PerfResult> results;
    testing::internal::CaptureStdout();
    format_perf_table(results);
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_TRUE(out.empty());
}

TEST(FormatPerfTable, DoesNotCompareDifferentSizedRows) {
    // Regression guard for the original bug: format_perf_table(results, 0)
    // used to divide every row's time_ms by row 0's time_ms regardless of
    // grid_size, so a 512-cubed row measured against a much smaller
    // baseline row printed a nonsensical "speedup" like 0.12x. There is
    // no speedup column any more (see HasGridColumnNoSpeedupColumn), but
    // this test also checks each row's own numbers are preserved
    // untouched -- i.e. nothing is silently computed relative to another
    // row.
    std::vector<PerfResult> results;
    PerfResult a; a.label = "naive"; a.time_ms = 1.403;  a.bandwidth_gbs = 1217.4; a.gflops = 608.7; a.grid_size = 256;
    PerfResult b; b.label = "naive"; b.time_ms = 11.462; b.bandwidth_gbs = 1250.9; b.gflops = 625.5; b.grid_size = 512;
    results.push_back(a);
    results.push_back(b);

    testing::internal::CaptureStdout();
    format_perf_table(results);
    const std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("1.403"), std::string::npos);
    EXPECT_NE(out.find("11.462"), std::string::npos);
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
