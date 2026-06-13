#pragma once

/**
 * include/stencil/config.hpp
 *
 * Runtime configuration loaded from CLI args or a JSON file.
 * Keeps all solver tunables in one place so benchmarks and tests
 * share the same parsing logic.
 */

#include "stencil/types.hpp"
#include <algorithm>
#include <algorithm>
#include <algorithm>

#include <cstring>
#include <stdexcept>
#include <string>

namespace stencil {

// ── Solver physics selection ──────────────────────────────────────────────────
enum class Physics {
    WavePropagation,   ///< Acoustic wave equation (default)
    HeatDiffusion,     ///< Parabolic heat equation
};

// ── Kernel variant ────────────────────────────────────────────────────────────
enum class KernelVariant {
    Naive,             ///< Global-memory baseline
    Tiled,             ///< Shared-memory tiled
    MultiGPU,          ///< Domain-decomposed across devices
};

// ── Config struct ─────────────────────────────────────────────────────────────
struct Config {
    // Grid
    std::size_t nx           = 256;
    std::size_t ny           = 256;
    std::size_t nz           = 256;
    real_t      dx           = static_cast<real_t>(10.0);   // metres
    real_t      dy           = static_cast<real_t>(10.0);
    real_t      dz           = static_cast<real_t>(10.0);

    // Time stepping
    int         steps        = 100;
    int         warmup_steps = 10;
    real_t      velocity     = static_cast<real_t>(1500.0); // m/s (water)
    real_t      diffusivity  = static_cast<real_t>(1.0e-4); // m²/s (heat)

    // Source pulse
    real_t      pulse_cx     = static_cast<real_t>(0.5);    // normalised centre
    real_t      pulse_cy     = static_cast<real_t>(0.5);
    real_t      pulse_cz     = static_cast<real_t>(0.5);
    real_t      pulse_sigma  = static_cast<real_t>(5.0);    // grid cells

    // Execution
    Physics       physics  = Physics::WavePropagation;
    KernelVariant variant  = KernelVariant::Naive;
    int           device   = 0;
    bool          verbose  = false;

    // I/O
    std::string   output_path;  // JSON results file (empty = no output)

    // ── Derived grid ──────────────────────────────────────────────────────────
    GridDims make_grid() const {
        GridDims g;
        g.nx = nx;
        g.ny = ny;
        g.nz = nz;
        g.dx = dx;
        g.dy = dy;
        g.dz = dz;
        // CFL-stable time step
        const real_t min_h = std::min({dx, dy, dz});
        g.dt = GridDims::CFL_LIMIT * min_h / velocity;
        return g;
    }
};

// ── CLI parser ────────────────────────────────────────────────────────────────
/**
 * parse_args — fills a Config from argv.
 *
 * Supported flags:
 *   --nx N  --ny N  --nz N
 *   --dx F  --dy F  --dz F
 *   --steps N  --warmup N
 *   --velocity F  --diffusivity F
 *   --physics wave|heat
 *   --kernel naive|tiled|multigpu
 *   --device N
 *   --output path/to/results.json
 *   --verbose
 */
inline Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* flag) { return std::strcmp(argv[i], flag) == 0; };
        auto next_int   = [&]() -> std::size_t {
            if (++i >= argc) throw std::runtime_error("missing value for flag");
            return static_cast<std::size_t>(std::stoul(argv[i]));
        };
        auto next_float = [&]() -> real_t {
            if (++i >= argc) throw std::runtime_error("missing value for flag");
            return static_cast<real_t>(std::stod(argv[i]));
        };
        auto next_str = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for flag");
            return argv[i];
        };

        if      (eq("--nx"))          cfg.nx           = next_int();
        else if (eq("--ny"))          cfg.ny           = next_int();
        else if (eq("--nz"))          cfg.nz           = next_int();
        else if (eq("--dx"))          cfg.dx           = next_float();
        else if (eq("--dy"))          cfg.dy           = next_float();
        else if (eq("--dz"))          cfg.dz           = next_float();
        else if (eq("--steps"))       cfg.steps        = static_cast<int>(next_int());
        else if (eq("--warmup"))      cfg.warmup_steps = static_cast<int>(next_int());
        else if (eq("--velocity"))    cfg.velocity     = next_float();
        else if (eq("--diffusivity")) cfg.diffusivity  = next_float();
        else if (eq("--device"))      cfg.device       = static_cast<int>(next_int());
        else if (eq("--output"))      cfg.output_path  = next_str();
        else if (eq("--verbose"))     cfg.verbose      = true;
        else if (eq("--physics")) {
            const std::string v = next_str();
            if      (v == "wave") cfg.physics = Physics::WavePropagation;
            else if (v == "heat") cfg.physics = Physics::HeatDiffusion;
            else throw std::runtime_error("unknown physics: " + v);
        }
        else if (eq("--kernel")) {
            const std::string v = next_str();
            if      (v == "naive")     cfg.variant = KernelVariant::Naive;
            else if (v == "tiled")     cfg.variant = KernelVariant::Tiled;
            else if (v == "multigpu")  cfg.variant = KernelVariant::MultiGPU;
            else throw std::runtime_error("unknown kernel: " + v);
        }
        else {
            throw std::runtime_error(std::string("unknown flag: ") + argv[i]);
        }
    }
    return cfg;
}

}  // namespace stencil
