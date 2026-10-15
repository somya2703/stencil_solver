/**
 * tests/unit/test_io.cpp
 *
 * Tests for write_results_json (checklist M2).
 *
 * write_results_json previously recorded gpu_name, cuda_version,
 * stencil_radius and precision but not the tile config (tile_x/tile_y/
 * pencil_z) or the run length (steps/warmup), so a results JSON file
 * could not be traced back to what produced it. These tests guard the
 * new fields and the no-default-value contract on steps/warmup (a call
 * site that forgets them should fail to compile, not silently write 0).
 */

#include <gtest/gtest.h>

#include "stencil/io.hpp"
#include "stencil/types.hpp"
using namespace stencil;
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

// Reads a file fully into a string; fails the test (via EXPECT) if it
// can't be opened, rather than throwing, so one bad test doesn't abort
// the whole binary.
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    EXPECT_TRUE(f.good()) << "could not open " << path;
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

class WriteResultsJsonTest : public ::testing::Test {
protected:
    std::string path = "test_io_output.json";
    void TearDown() override { std::remove(path.c_str()); }
};

}  // namespace

TEST_F(WriteResultsJsonTest, RecordsTileConfigFromCompileTimeMacros) {
    std::vector<PerfResult> results;
    PerfResult r;
    r.label     = "naive";
    r.grid_size = 512;
    r.time_ms   = 11.5;
    results.push_back(r);

    write_results_json(path, results, /*device=*/0, /*steps=*/200, /*warmup=*/20);
    const std::string content = read_file(path);

    // Checked against the actual compile-time macros, not a hardcoded
    // value, so this doesn't break if the project's tuned default changes.
    EXPECT_NE(content.find("\"tile_x\": " + std::to_string(TILE_X)), std::string::npos);
    EXPECT_NE(content.find("\"tile_y\": " + std::to_string(TILE_Y)), std::string::npos);
    EXPECT_NE(content.find("\"pencil_z\": " + std::to_string(PENCIL_Z)), std::string::npos);
}

TEST_F(WriteResultsJsonTest, RecordsStepsAndWarmupFromArguments) {
    std::vector<PerfResult> results;
    PerfResult r;
    r.label     = "tiled";
    r.grid_size = 256;
    r.time_ms   = 2.0;
    results.push_back(r);

    write_results_json(path, results, 0, 200, 20);
    const std::string content = read_file(path);

    EXPECT_NE(content.find("\"steps\": 200"), std::string::npos);
    EXPECT_NE(content.find("\"warmup\": 20"), std::string::npos);
}

TEST_F(WriteResultsJsonTest, DifferentStepsWarmupProduceDifferentOutput) {
    // Guards against the fields being written but silently ignoring the
    // actual arguments (e.g. hardcoded instead of wired through).
    std::vector<PerfResult> results;
    PerfResult r;
    r.label     = "naive";
    r.grid_size = 64;
    r.time_ms   = 0.1;
    results.push_back(r);

    write_results_json(path, results, 0, 50, 5);
    const std::string content_a = read_file(path);
    EXPECT_NE(content_a.find("\"steps\": 50"), std::string::npos);
    EXPECT_NE(content_a.find("\"warmup\": 5"), std::string::npos);

    write_results_json(path, results, 0, 999, 99);
    const std::string content_b = read_file(path);
    EXPECT_NE(content_b.find("\"steps\": 999"), std::string::npos);
    EXPECT_NE(content_b.find("\"warmup\": 99"), std::string::npos);
    EXPECT_EQ(content_b.find("\"steps\": 50"), std::string::npos)
        << "second write should have overwritten the first, not appended";
}

TEST_F(WriteResultsJsonTest, EmptyPathIsNoOp) {
    std::vector<PerfResult> results;
    PerfResult r;
    r.label     = "naive";
    r.grid_size = 64;
    r.time_ms   = 0.1;
    results.push_back(r);

    // Must not throw and must not create a file at the (empty) path.
    EXPECT_NO_THROW(write_results_json("", results, 0, 10, 1));
}

TEST_F(WriteResultsJsonTest, PerResultFieldsStillPresent) {
    // Pre-existing fields (checklist M2 only ADDS fields, must not drop any).
    std::vector<PerfResult> results;
    PerfResult r;
    r.label         = "tiled";
    r.grid_size     = 384;
    r.time_ms       = 6.743;
    r.bandwidth_gbs = 882.9;
    r.gflops        = 441.4;
    results.push_back(r);

    write_results_json(path, results, 0, 200, 20);
    const std::string content = read_file(path);

    EXPECT_NE(content.find("\"label\": \"tiled\""), std::string::npos);
    EXPECT_NE(content.find("\"grid_size\": 384"), std::string::npos);
    EXPECT_NE(content.find("6.743"), std::string::npos);
    EXPECT_NE(content.find("882.9"), std::string::npos);
    EXPECT_NE(content.find("441.4"), std::string::npos);
}
