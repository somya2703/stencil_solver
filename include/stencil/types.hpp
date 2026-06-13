#pragma once

/**
 * include/stencil/types.hpp
 *
 * Foundational types, constants, and macros shared across all translation
 * units (host C++, device CUDA, and CPU-fallback OpenMP paths).
 *
 * Nothing in this header pulls in CUDA runtime headers — it is safe to
 * include from pure C++ translation units.  CUDA-specific annotations
 * (__host__, __device__) are guarded so clang-tidy can parse this without
 * nvcc.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ── Compiler portability ──────────────────────────────────────────────────────
#ifdef __CUDACC__
#  define STENCIL_HD __host__ __device__
#  define STENCIL_D  __device__
#  define STENCIL_FI __forceinline__
#else
#  define STENCIL_HD
#  define STENCIL_D
#  define STENCIL_FI inline
#endif

// ── Floating-point precision ──────────────────────────────────────────────────
// Default: single precision.  Pass -DSTENCIL_FP64 to CMake for double.
#ifdef STENCIL_FP64
using real_t = double;
#else
using real_t = float;
#endif

// ── Stencil compile-time parameters ──────────────────────────────────────────
// STENCIL_RADIUS = half-order per axis.
//   1 → 2nd-order,  2 → 4th,  3 → 6th,  4 → 8th  (default)
#ifndef STENCIL_RADIUS
#  define STENCIL_RADIUS 4
#endif

// Shared-memory tile dimensions (tiled kernel).
// TILE_X must equal the warp size (32) for coalesced loads.
#ifndef TILE_X
#  define TILE_X 32
#endif
#ifndef TILE_Y
#  define TILE_Y 8
#endif
// Z-pencil depth kept in registers (not shared memory).
#ifndef PENCIL_Z
#  define PENCIL_Z 16
#endif

// ── CUDA error-checking macros ────────────────────────────────────────────────
// In CPU-fallback mode these are compiled out.
#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>

#  define CUDA_CHECK(call)                                               \
     do {                                                                \
       cudaError_t _e = (call);                                          \
       if (_e != cudaSuccess) {                                          \
         fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                    \
                 __FILE__, __LINE__, cudaGetErrorString(_e));            \
         std::exit(EXIT_FAILURE);                                        \
       }                                                                 \
     } while (0)

#  define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#else
// CPU-fallback stubs — nothing to check
#  define CUDA_CHECK(call)    (call)
#  define CUDA_CHECK_LAST()   do {} while (0)
#endif  // STENCIL_CPU_FALLBACK

// ── NVTX range markers ────────────────────────────────────────────────────────
#ifdef ENABLE_NVTX
#  include <nvtx3/nvtx3.hpp>
#  define NVTX_RANGE(name)  nvtx3::scoped_range _nvtx_rng_ ## __LINE__ {name}
#  define NVTX_PUSH(name)   nvtxRangePushA(name)
#  define NVTX_POP()        nvtxRangePop()
#else
#  define NVTX_RANGE(name)  do {} while (0)
#  define NVTX_PUSH(name)   do {} while (0)
#  define NVTX_POP()        do {} while (0)
#endif

// ── Grid descriptor ───────────────────────────────────────────────────────────
/**
 * GridDims — physical dimensions and spacing of the 3-D computational domain.
 *
 * Dimensions include halo cells on all sides (halo width = STENCIL_RADIUS).
 * The interior (owned) region is [R, nx-R) × [R, ny-R) × [R, nz-R).
 *
 * Memory layout: row-major, z-slowest (Fortran-column-major in z).
 *   linear index = iz * nx*ny  +  iy * nx  +  ix
 */
namespace stencil {

struct GridDims {
    std::size_t nx = 0;  ///< total cells in X (interior + 2*R halos)
    std::size_t ny = 0;  ///< total cells in Y
    std::size_t nz = 0;  ///< total cells in Z
    real_t      dx = 1;  ///< grid spacing in X  [m]
    real_t      dy = 1;  ///< grid spacing in Y  [m]
    real_t      dz = 1;  ///< grid spacing in Z  [m]
    real_t      dt = 1;  ///< time step           [s]

    STENCIL_HD std::size_t size() const noexcept { return nx * ny * nz; }

    STENCIL_HD std::size_t idx(std::size_t x, std::size_t y, std::size_t z) const noexcept {
        return z * nx * ny + y * nx + x;
    }

    /// Number of interior (non-halo) points along each axis
    STENCIL_HD std::size_t interior_nx() const noexcept { return nx - 2 * STENCIL_RADIUS; }
    STENCIL_HD std::size_t interior_ny() const noexcept { return ny - 2 * STENCIL_RADIUS; }
    STENCIL_HD std::size_t interior_nz() const noexcept { return nz - 2 * STENCIL_RADIUS; }
    STENCIL_HD std::size_t interior_size() const noexcept {
        return interior_nx() * interior_ny() * interior_nz();
    }

    /// CFL stability limit for acoustic wave equation: dt <= CFL * dx / v_max
    static constexpr real_t CFL_LIMIT = static_cast<real_t>(0.45);
};

// ── Finite-difference coefficients ───────────────────────────────────────────
/**
 * FDCoeffs<RADIUS> — centred FD weights for the second derivative.
 *
 * Approximation:  u'' ≈ c0*u[i] + sum_{r=1}^{R} c[r-1]*(u[i+r] + u[i-r])
 *
 * Derived via standard Taylor expansion / method of undetermined coefficients.
 * Coefficients satisfy: c0 + 2*sum(c) = 0  (consistency with zero DC mode).
 */
template <int R>
struct FDCoeffs;

template <>
struct FDCoeffs<1> {
    static constexpr real_t c[1] = {static_cast<real_t>(1)};
    static constexpr real_t c0   = static_cast<real_t>(-2);
};

template <>
struct FDCoeffs<2> {
    static constexpr real_t c[2] = {
        static_cast<real_t>(4.0 / 3.0),
        static_cast<real_t>(-1.0 / 12.0)};
    static constexpr real_t c0 = static_cast<real_t>(-5.0 / 2.0);
};

template <>
struct FDCoeffs<3> {
    static constexpr real_t c[3] = {
        static_cast<real_t>(3.0 / 2.0),
        static_cast<real_t>(-3.0 / 20.0),
        static_cast<real_t>(1.0 / 90.0)};
    static constexpr real_t c0 = static_cast<real_t>(-49.0 / 18.0);
};

template <>
struct FDCoeffs<4> {
    static constexpr real_t c[4] = {
        static_cast<real_t>(8.0 / 5.0),
        static_cast<real_t>(-1.0 / 5.0),
        static_cast<real_t>(8.0 / 315.0),
        static_cast<real_t>(-1.0 / 560.0)};
    static constexpr real_t c0 = static_cast<real_t>(-205.0 / 72.0);
};

// ── Roofline / bandwidth helpers ──────────────────────────────────────────────
/**
 * Theoretical memory traffic per stencil step (bytes).
 *
 * Per interior point, the stencil reads:
 *   p_cur:  2*R points per axis × 3 axes + 1 centre  = 6R+1 reads
 *   p_prev: 1 read  (time integration)
 *   vel2:   1 read
 * Writes:
 *   p_next: 1 write
 *
 * Total = (6R + 3) reads + 1 write = (6R+4) × sizeof(real_t) per point.
 * (Assumes no L2 reuse across steps — conservative / matches Nsight DRAM metric.)
 */
inline double bandwidth_bytes_per_step(const GridDims& g) {
    const int    R = STENCIL_RADIUS;
    const double N = static_cast<double>(g.interior_size());
    return N * static_cast<double>(6 * R + 4) * sizeof(real_t);
}

/**
 * Theoretical FLOP count per stencil step.
 *
 * Per interior point, per axis:
 *   2R multiply-adds (neighbour contributions) + 1 multiply (centre) = 4R+1 flops
 * Three axes:  3*(4R+1)
 * Time integration (2*p_cur - p_prev + dt2*vel2*lap):  5 flops
 */
inline double flops_per_step(const GridDims& g) {
    const int    R = STENCIL_RADIUS;
    const double N = static_cast<double>(g.interior_size());
    return N * static_cast<double>(3 * (4 * R + 1) + 5);
}

}  // namespace stencil
