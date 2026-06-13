#pragma once

/**
 * include/stencil/kernels.cuh
 *
 * Public interface for all stencil kernel launchers.
 * This header is included by solver/ and tests/; it is the only
 * file that needs to know about CUDA runtime types (cudaStream_t).
 *
 * Each launcher:
 *   1. Computes inv_dx2 / inv_dy2 / inv_dz2 / dt2 from GridDims
 *   2. Selects a block/grid configuration
 *   3. Launches the __global__ kernel
 *   4. Calls CUDA_CHECK_LAST()
 */

#include "stencil/types.hpp"

#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>
#else
// In CPU-fallback mode, streams are not used — define a compatible type.
#  ifndef __DRIVER_TYPES_H__
using cudaStream_t = void*;
#  endif
#endif

#include "stencil/config.hpp"
namespace stencil {

// ── Coefficient upload (call once before first step) ─────────────────────────
/// Upload FD coefficients to __constant__ memory on the current device.
void upload_fd_coefficients();

// ── Naive kernel (global-memory baseline) ────────────────────────────────────
/**
 * launch_naive — baseline stencil: every thread reads all 6R+1 neighbours
 * directly from global memory.  No data reuse between threads.
 *
 * Profiling target: demonstrates memory-bandwidth bottleneck.
 */
void launch_naive(
    const real_t*    d_p_cur,
    const real_t*    d_p_prev,
    const real_t*    d_vel2,
          real_t*    d_p_next,
    const GridDims&  grid,
    cudaStream_t     stream = nullptr);

// ── Tiled kernel (shared-memory + register pencil) ───────────────────────────
/**
 * launch_tiled — optimised stencil:
 *   - XY tile loaded into shared memory (TILE_X × TILE_Y + 2R halo on each side)
 *   - Z-pencil kept in registers (PENCIL_Z depth)
 *   - Single global-memory load per cell, multiple reuses in shared memory
 *
 * Profiling target: demonstrates bandwidth reduction and occupancy improvement.
 */
void launch_tiled(
    const real_t*    d_p_cur,
    const real_t*    d_p_prev,
    const real_t*    d_vel2,
          real_t*    d_p_next,
    const GridDims&  grid,
    cudaStream_t     stream = nullptr);


// ── Multi-GPU slab kernel ─────────────────────────────────────────────────────
/**
 * launch_slab — tiled kernel for one Z-slab of a domain-decomposed run.
 * Identical to launch_tiled() but semantically distinct: the Z-face halos
 * in dims are pre-filled by IHaloExchange before this call.
 */
void launch_slab(
    const real_t*    d_p_cur,
    const real_t*    d_p_prev,
    const real_t*    d_vel2,
          real_t*    d_p_next,
    const GridDims&  dims,
    cudaStream_t     stream = nullptr);

// ── Convenience: launch by variant ───────────────────────────────────────────

inline void launch(KernelVariant variant,
                   const real_t*   d_p_cur,
                   const real_t*   d_p_prev,
                   const real_t*   d_vel2,
                         real_t*   d_p_next,
                   const GridDims& grid,
                   cudaStream_t    stream = nullptr) {
    switch (variant) {
        case KernelVariant::Naive:
            launch_naive(d_p_cur, d_p_prev, d_vel2, d_p_next, grid, stream);
            break;
        case KernelVariant::Tiled:
            launch_tiled(d_p_cur, d_p_prev, d_vel2, d_p_next, grid, stream);
            break;
        default:
            // MultiGPU handled by the comm layer
            launch_tiled(d_p_cur, d_p_prev, d_vel2, d_p_next, grid, stream);
            break;
    }
}

}  // namespace stencil
