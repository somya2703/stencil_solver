/**
 * src/utils/grid.cpp
 *
 * Host-side grid utilities: validation, CFL checking, and
 * non-inline helpers for HostGrid.
 */

#include "stencil/grid.hpp"
#include <algorithm>
#include "stencil/types.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <cstdio>

namespace stencil {

// ── Validation ────────────────────────────────────────────────────────────────
/**
 * validate_grid — throw if the grid dimensions are unusable.
 *
 * Checks:
 *   - All dimensions > 2 * STENCIL_RADIUS  (at least one interior cell)
 *   - dx, dy, dz, dt > 0
 *   - CFL condition for acoustic wave equation
 */
void validate_grid(const GridDims& g, real_t v_max) {
    const int R = STENCIL_RADIUS;
    std::ostringstream err;
    bool bad = false;

    auto check = [&](bool cond, const char* msg) {
        if (!cond) { err << "  " << msg << "\n"; bad = true; }
    };

    check(g.nx > static_cast<std::size_t>(2 * R),
          "nx must be > 2 * STENCIL_RADIUS");
    check(g.ny > static_cast<std::size_t>(2 * R),
          "ny must be > 2 * STENCIL_RADIUS");
    check(g.nz > static_cast<std::size_t>(2 * R),
          "nz must be > 2 * STENCIL_RADIUS");
    check(g.dx > real_t{0}, "dx must be positive");
    check(g.dy > real_t{0}, "dy must be positive");
    check(g.dz > real_t{0}, "dz must be positive");
    check(g.dt > real_t{0}, "dt must be positive");

    if (!bad && v_max > real_t{0}) {
        const real_t min_h = std::min({g.dx, g.dy, g.dz});
        const real_t cfl   = v_max * g.dt / min_h;
        if (cfl > GridDims::CFL_LIMIT) {
            err << "  CFL violated: v*dt/dx = " << cfl
                << " > " << GridDims::CFL_LIMIT << "\n";
            bad = true;
        }
    }

    if (bad)
        throw std::invalid_argument("Invalid grid:\n" + err.str());
}

/**
 * print_grid_info — dump grid configuration to stdout.
 */
void print_grid_info(const GridDims& g, const char* label) {
    const int R = STENCIL_RADIUS;
    std::printf("\n── Grid: %s ───────────────────────────────\n", label);
    std::printf("  Total cells   : %zu × %zu × %zu  =  %.2f M\n",
                g.nx, g.ny, g.nz,
                static_cast<double>(g.size()) / 1e6);
    std::printf("  Interior      : %zu × %zu × %zu  =  %.2f M\n",
                g.interior_nx(), g.interior_ny(), g.interior_nz(),
                static_cast<double>(g.interior_size()) / 1e6);
    std::printf("  Spacing       : dx=%.2f m  dy=%.2f m  dz=%.2f m\n",
                static_cast<double>(g.dx),
                static_cast<double>(g.dy),
                static_cast<double>(g.dz));
    std::printf("  Time step     : dt=%.4e s\n", static_cast<double>(g.dt));
    std::printf("  Stencil radius: R=%d  (%d-th order)\n", R, 2*R);
    std::printf("  Memory/field  : %.2f MB\n",
                static_cast<double>(g.size() * sizeof(real_t)) / 1e6);
    std::printf("  Fields (3×p + vel2): %.2f MB\n",
                static_cast<double>(g.size() * sizeof(real_t) * 4) / 1e6);
    std::printf("\n");
}

/**
 * l2_error — RMS difference between two HostGrids (for correctness checks).
 */
double l2_error(const HostGrid& a, const HostGrid& b) {
    if (a.dims.size() != b.dims.size())
        throw std::invalid_argument("l2_error: grid size mismatch");

    double sum = 0.0;
    const std::size_t N = a.dims.size();
    for (std::size_t i = 0; i < N; ++i) {
        const double diff = static_cast<double>(a.data[i])
                          - static_cast<double>(b.data[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum / static_cast<double>(N));
}

/**
 * max_abs_error — L∞ difference between two HostGrids.
 */
double max_abs_error(const HostGrid& a, const HostGrid& b) {
    if (a.dims.size() != b.dims.size())
        throw std::invalid_argument("max_abs_error: grid size mismatch");

    double max_err = 0.0;
    const std::size_t N = a.dims.size();
    for (std::size_t i = 0; i < N; ++i) {
        const double diff = std::abs(static_cast<double>(a.data[i])
                                   - static_cast<double>(b.data[i]));
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

}  // namespace stencil
