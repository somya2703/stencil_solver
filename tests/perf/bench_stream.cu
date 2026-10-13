/**
 * tests/perf/bench_stream.cu
 *
 * Standalone STREAM-style DRAM bandwidth test (copy / read / write).
 * Independent of the solver kernels; gives the ~192 GB/s peak an
 * independent measurement instead of a derived one.
 *
 * Usage:
 *   bench_stream [--mib N] [--reps N] [--warmup N] [--device N]
 *                [--only copy|read|write] [--peak GBS] [--output file.json]
 *
 * Bytes counted: copy = 2*B, read = B, write = B (B = bytes per buffer).
 * GB = 1e9 bytes, to match the 192.1 GB/s peak (decimal).
 *
 * ncu cross-check (single launch, only the chosen kernel runs):
 *   ncu --clock-control none -k regex:stream_copy --launch-count 1 \
 *       --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
 *dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
 *       ./bench/bench_stream --only copy --warmup 0 --reps 1
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef STENCIL_CPU_FALLBACK

int main() {
    std::printf("bench_stream: skipped (built with STENCIL_CPU_FALLBACK)\n");
    return EXIT_SUCCESS;
}

#else

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define SB_CHECK(call)                                                        \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(e_));                   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ── Kernels (grid-stride, float4) ────────────────────────────────────────────
__global__ void stream_init(float4* p, size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < n; i += stride) {
        const float b = static_cast<float>(i & 0xFFFFF);
        p[i] = make_float4(b, b + 1.0f, b + 2.0f, b + 3.0f);
    }
}

__global__ void stream_copy(const float4* __restrict__ src,
                            float4* __restrict__ dst, size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < n; i += stride)
        dst[i] = src[i];
}

__global__ void stream_read(const float4* __restrict__ src, float* sink,
                            size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float acc = 0.0f;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < n; i += stride) {
        const float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // Practically never true; keeps the loads from being optimised away.
    if (acc == -1.2345f) sink[0] = acc;
}

__global__ void stream_write(float4* __restrict__ dst, size_t n, float c) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const float4 v = make_float4(c, c, c, c);
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < n; i += stride)
        dst[i] = v;
}

__global__ void stream_verify(const float4* a, const float4* b, size_t n,
                              unsigned long long* bad) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int local = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < n; i += stride) {
        const float4 x = a[i], y = b[i];
        local += (x.x != y.x) + (x.y != y.y) + (x.z != y.z) + (x.w != y.w);
    }
    if (local) atomicAdd(bad, static_cast<unsigned long long>(local));
}

// ── Helpers ──────────────────────────────────────────────────────────────────
struct Result {
    std::string name;
    double bytes = 0, best_ms = 0, med_ms = 0;
    double best_gbs() const { return bytes / (best_ms * 1e-3) / 1e9; }
    double med_gbs() const { return bytes / (med_ms * 1e-3) / 1e9; }
};

template <typename Launch>
static Result time_kernel(const char* name, double bytes, int warmup, int reps,
                          Launch launch) {
    cudaEvent_t e0, e1;
    SB_CHECK(cudaEventCreate(&e0));
    SB_CHECK(cudaEventCreate(&e1));
    for (int i = 0; i < warmup; ++i) launch();
    SB_CHECK(cudaGetLastError());
    SB_CHECK(cudaDeviceSynchronize());

    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        SB_CHECK(cudaEventRecord(e0));
        launch();
        SB_CHECK(cudaEventRecord(e1));
        SB_CHECK(cudaEventSynchronize(e1));
        float t = 0.f;
        SB_CHECK(cudaEventElapsedTime(&t, e0, e1));
        ms.push_back(static_cast<double>(t));
    }
    SB_CHECK(cudaGetLastError());
    SB_CHECK(cudaEventDestroy(e0));
    SB_CHECK(cudaEventDestroy(e1));

    std::sort(ms.begin(), ms.end());
    Result r;
    r.name = name;
    r.bytes = bytes;
    r.best_ms = ms.front();
    r.med_ms = ms[ms.size() / 2];
    return r;
}

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else o += c;
    }
    return o;
}

int main(int argc, char** argv) {
    size_t mib = 512;
    int reps = 50, warmup = 5, device = 0;
    double peak_override = 0.0;
    std::string only, output;

    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* f) { return std::strcmp(argv[i], f) == 0; };
        auto val = [&]() -> const char* {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", argv[i - 1]);
                std::exit(EXIT_FAILURE);
            }
            return argv[i];
        };
        if (eq("--mib")) mib = std::strtoull(val(), nullptr, 10);
        else if (eq("--reps")) reps = std::atoi(val());
        else if (eq("--warmup")) warmup = std::atoi(val());
        else if (eq("--device")) device = std::atoi(val());
        else if (eq("--peak")) peak_override = std::atof(val());
        else if (eq("--only")) only = val();
        else if (eq("--output")) output = val();
        else {
            std::fprintf(stderr, "unknown flag: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }
    if (reps < 1 || mib < 1) {
        std::fprintf(stderr, "--reps and --mib must be >= 1\n");
        return EXIT_FAILURE;
    }
    if (!only.empty() && only != "copy" && only != "read" && only != "write") {
        std::fprintf(stderr, "--only must be copy|read|write\n");
        return EXIT_FAILURE;
    }

    SB_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    SB_CHECK(cudaGetDeviceProperties(&prop, device));

    // Same derivation as the code's banner, WITH the DDR x2 factor.
    const double derived_peak = static_cast<double>(prop.memoryClockRate) * 1e3
                              * 2.0 * static_cast<double>(prop.memoryBusWidth)
                              / 8.0 / 1e9;
    const double peak = peak_override > 0.0 ? peak_override : derived_peak;

    const size_t bytes_per_buf = mib * 1024ull * 1024ull;
    const size_t n = bytes_per_buf / sizeof(float4);
    const double B = static_cast<double>(n * sizeof(float4));

    std::printf("=== bench_stream: DRAM bandwidth (STREAM-style) ===\n\n");
    std::printf("Device     : %s (SM %d.%d, %d SMs)\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount);
    std::printf("L2 cache   : %.1f MiB\n",
                static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0));
    std::printf("Peak (derived, mem clk x2 x bus/8): %.1f GB/s%s\n", derived_peak,
                peak_override > 0.0 ? "  [overridden by --peak]" : "");
    std::printf("Buffer     : 2 x %.0f MiB   reps=%d warmup=%d\n\n",
                B / (1024.0 * 1024.0), reps, warmup);
    if (B < 8.0 * static_cast<double>(prop.l2CacheSize))
        std::printf("WARNING: buffer < 8 x L2; results may include L2 hits.\n\n");

    float4 *a = nullptr, *b = nullptr;
    float* sink = nullptr;
    unsigned long long* bad = nullptr;
    SB_CHECK(cudaMalloc(&a, n * sizeof(float4)));
    SB_CHECK(cudaMalloc(&b, n * sizeof(float4)));
    SB_CHECK(cudaMalloc(&sink, sizeof(float)));
    SB_CHECK(cudaMalloc(&bad, sizeof(unsigned long long)));
    SB_CHECK(cudaMemset(sink, 0, sizeof(float)));
    SB_CHECK(cudaMemset(bad, 0, sizeof(unsigned long long)));

    const int threads = 256;
    int blocks_per_sm = 0;
    SB_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, stream_copy, threads, 0));
    const int blocks = std::max(1, blocks_per_sm) * prop.multiProcessorCount;

    stream_init<<<blocks, threads>>>(a, n);
    stream_init<<<blocks, threads>>>(b, n);
    SB_CHECK(cudaGetLastError());
    SB_CHECK(cudaDeviceSynchronize());

    std::vector<Result> results;
    const bool all = only.empty();

    if (all || only == "copy")
        results.push_back(time_kernel("copy", 2.0 * B, warmup, reps,
                                      [&] { stream_copy<<<blocks, threads>>>(a, b, n); }));
    if (all || only == "read")
        results.push_back(time_kernel("read", B, warmup, reps,
                                      [&] { stream_read<<<blocks, threads>>>(a, sink, n); }));
    if (all || only == "write")
        results.push_back(time_kernel("write", B, warmup, reps,
                                      [&] { stream_write<<<blocks, threads>>>(b, n, 1.0f); }));

    // Sanity: a fresh copy must reproduce a exactly (guards against a no-op kernel).
    stream_copy<<<blocks, threads>>>(a, b, n);
    stream_verify<<<blocks, threads>>>(a, b, n, bad);
    SB_CHECK(cudaGetLastError());
    unsigned long long h_bad = 0;
    SB_CHECK(cudaMemcpy(&h_bad, bad, sizeof(h_bad), cudaMemcpyDeviceToHost));

    std::printf("%-6s  %9s  %9s  %10s  %10s  %9s  %9s\n", "Kernel", "best ms",
                "med ms", "best GB/s", "med GB/s", "best %pk", "med %pk");
    for (const auto& r : results)
        std::printf("%-6s  %9.3f  %9.3f  %10.1f  %10.1f  %8.1f%%  %8.1f%%\n",
                    r.name.c_str(), r.best_ms, r.med_ms, r.best_gbs(),
                    r.med_gbs(), 100.0 * r.best_gbs() / peak,
                    100.0 * r.med_gbs() / peak);
    std::printf("\nVerify copy: %s (%llu mismatches)\n",
                h_bad == 0 ? "OK" : "FAILED", h_bad);

    if (!output.empty()) {
        FILE* f = std::fopen(output.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "cannot open %s\n", output.c_str());
        } else {
            std::fprintf(f, "{\n  \"gpu_name\": \"%s\",\n", json_escape(prop.name).c_str());
            std::fprintf(f, "  \"peak_gbs\": %.3f,\n  \"buffer_mib\": %zu,\n", peak, mib);
            std::fprintf(f, "  \"reps\": %d,\n  \"warmup\": %d,\n  \"results\": [\n", reps, warmup);
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& r = results[i];
                std::fprintf(f,
                    "    {\"label\": \"%s\", \"best_ms\": %.4f, \"median_ms\": %.4f, "
                    "\"best_gbs\": %.2f, \"median_gbs\": %.2f}%s\n",
                    r.name.c_str(), r.best_ms, r.med_ms, r.best_gbs(), r.med_gbs(),
                    i + 1 < results.size() ? "," : "");
            }
            std::fprintf(f, "  ]\n}\n");
            std::fclose(f);
            std::printf("Results written to: %s\n", output.c_str());
        }
    }

    cudaFree(a); cudaFree(b); cudaFree(sink); cudaFree(bad);
    return h_bad == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#endif  // STENCIL_CPU_FALLBACK