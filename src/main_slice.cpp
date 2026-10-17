/**
 * src/main_slice.cpp
 *
 * Visualisation helper: runs the solver and writes the middle XY plane of the
 * field every N steps, so the simulation can be turned into an animation
 * (scripts/make_slice_gif.py). Not a benchmark: it downloads the full grid for
 * every saved frame, so use a modest grid.
 *
 * Usage:
 *   ./stencil_slice --nx 128 --steps 240 --every 4 --slice-out results/wave
 *   python3 scripts/make_slice_gif.py --prefix results/wave --out docs/media/wave_slice.gif
 *
 * Extra flags (everything else is parsed by stencil::parse_args):
 *   --every N          save a frame every N steps (default 5); frame 0 is the initial state
 *   --slice-out PATH   output prefix (default "slices"); writes PATH.f32 and PATH.json
 *
 * Output format (also written by scripts/reference_slice.py):
 *   PATH.f32   float32 frames, frame-major, each frame ny rows of nx values (x fastest)
 *   PATH.json  nx, ny, nz, plane_index, frames, every, steps, dt, dx, velocity, ...
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "stencil/config.hpp"
#include "stencil/grid.hpp"
#include "stencil/solver.hpp"
#include "stencil/types.hpp"

namespace {

struct SliceOptions {
    int every              = 5;
    std::string out_prefix = "slices";
};

// Removes --every / --slice-out from argv (they are not known to parse_args).
std::vector<char*> extract_slice_options(int argc, char** argv, SliceOptions& opt) {
    std::vector<char*> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--every" || a == "--slice-out") {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + a);
            const std::string v = argv[++i];
            if (a == "--every")
                opt.every = std::stoi(v);
            else
                opt.out_prefix = v;
        } else {
            rest.push_back(argv[i]);
        }
    }
    return rest;
}

const char* physics_name(stencil::Physics p) {
    return p == stencil::Physics::HeatDiffusion ? "heat" : "wave";
}

const char* kernel_name(stencil::KernelVariant v) {
    switch (v) {
        case stencil::KernelVariant::Naive:
            return "naive";
        case stencil::KernelVariant::Tiled:
            return "tiled";
        default:
            return "multigpu";
    }
}

}  // namespace

int main(int argc, char** argv) {
    SliceOptions opt;
    stencil::Config cfg;
    try {
        std::vector<char*> rest = extract_slice_options(argc, argv, opt);
        cfg                     = stencil::parse_args(static_cast<int>(rest.size()), rest.data());
        if (opt.every < 1)
            throw std::runtime_error("--every must be >= 1");
    } catch (const std::exception& e) {
        std::fprintf(
            stderr,
            "Error: %s\n\nUsage: %s [--nx N] [--steps N] [--every N] [--slice-out PREFIX]\n"
            "          [--physics wave|heat] [--kernel naive|tiled] [--velocity F]\n"
            "          [--device N]\n",
            e.what(), argv[0]);
        return EXIT_FAILURE;
    }

    // Every solver requires a cubic grid; --nx alone selects an N^3 cube.
    cfg.ny = cfg.nx;
    cfg.nz = cfg.nx;

    try {
        auto solver                = stencil::make_solver(cfg);
        const stencil::GridDims& d = solver->dims();
        const std::size_t plane_z  = d.nz / 2;

        const std::string bin_path = opt.out_prefix + ".f32";
        std::ofstream bin(bin_path, std::ios::binary);
        if (!bin)
            throw std::runtime_error("cannot open " + bin_path);

        stencil::HostGrid field(d);
        std::vector<float> plane(d.nx * d.ny);
        int frames = 0;

        auto capture = [&]() {
            solver->download(field);
            for (std::size_t iy = 0; iy < d.ny; ++iy)
                for (std::size_t ix = 0; ix < d.nx; ++ix)
                    plane[iy * d.nx + ix] = static_cast<float>(field.at(ix, iy, plane_z));
            bin.write(reinterpret_cast<const char*>(plane.data()),
                      static_cast<std::streamsize>(plane.size() * sizeof(float)));
            ++frames;
        };

        capture();  // frame 0: initial condition
        for (int step = 1; step <= cfg.steps; ++step) {
            solver->step();
            if (step % opt.every == 0)
                capture();
        }
        bin.close();

        const std::string json_path = opt.out_prefix + ".json";
        std::ofstream js(json_path);
        if (!js)
            throw std::runtime_error("cannot open " + json_path);
        js << "{\n"
           << "  \"nx\": " << d.nx << ",\n"
           << "  \"ny\": " << d.ny << ",\n"
           << "  \"nz\": " << d.nz << ",\n"
           << "  \"plane\": \"xy\",\n"
           << "  \"plane_index\": " << plane_z << ",\n"
           << "  \"frames\": " << frames << ",\n"
           << "  \"every\": " << opt.every << ",\n"
           << "  \"steps\": " << cfg.steps << ",\n"
           << "  \"dt\": " << static_cast<double>(d.dt) << ",\n"
           << "  \"dx\": " << static_cast<double>(d.dx) << ",\n"
           << "  \"velocity\": " << static_cast<double>(cfg.velocity) << ",\n"
           << "  \"physics\": \"" << physics_name(cfg.physics) << "\",\n"
           << "  \"kernel\": \"" << kernel_name(cfg.variant) << "\",\n"
           << "  \"dtype\": \"float32\"\n"
           << "}\n";

        std::printf("Wrote %d frames (%zu x %zu, z = %zu) to %s and %s\n", frames, d.nx, d.ny,
                    plane_z, bin_path.c_str(), json_path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
