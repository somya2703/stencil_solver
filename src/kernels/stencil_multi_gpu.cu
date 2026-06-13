/**
 * src/kernels/stencil_multi_gpu.cu
 *
 * Multi-GPU stencil wrapper.
 *
 * Each MPI rank owns a Z-slab of the global domain.  This file provides:
 *
 *   launch_slab()  — launches the tiled kernel on one GPU's slab, but
 *                    skips the two outermost halo planes in Z (those are
 *                    filled by IHaloExchange, not by computation).
 *
 * The key difference from launch_tiled() is that launch_slab() operates
 * on a local GridDims (slab + 2R halos) and the Z boundaries of the
 * local grid are the halo planes exchanged with neighbours — so the
 * kernel still uses R guard cells in Z, but those guard cells come from
 * the communication layer, not from Dirichlet zeros.
 *
 * Execution model per time step:
 *   1. launch_slab()    — compute all interior Z cells on this rank's GPU
 *   2. halo->exchange() — fill Z-face halos from neighbour ranks (NCCL/MPI)
 *   3. rotate buffers   — same ring-buffer swap as the single-GPU solver
 *
 * Overlapping compute and communication (future optimisation):
 *   The slab can be split into a compute-halo region and an interior region.
 *   The interior region is computed on stream A; the Z-face kernel + halo
 *   exchange run on stream B with an event dependency.  This is not
 *   implemented here but the split-launch interface supports it.
 *
 * Note: this file intentionally does NOT re-implement the Laplacian kernel.
 * It delegates to kernel_tiled (compiled in stencil_tiled.cu) via the
 * same launch_tiled() function — CUDA separable compilation links them
 * together. This keeps the kernel code DRY and ensures naive/tiled/multigpu
 * all use the same arithmetic.
 */

#include "stencil/comm.hpp"
#include "stencil/kernels.cuh"
#include "stencil/types.hpp"

#ifndef STENCIL_CPU_FALLBACK
#include <cuda_runtime.h>

namespace stencil {

/**
 * launch_slab — runs the tiled stencil on a local Z-slab.
 *
 * Identical to launch_tiled() except the caller guarantees that:
 *   - dims.nz includes 2R halo planes (R bottom + R top)
 *   - Those halo planes are pre-filled by IHaloExchange before this call
 *   - The kernel therefore has valid stencil arms for all interior points
 *
 * @param d_p_cur    Current time level (slab, with halos)
 * @param d_p_prev   Previous time level (slab, with halos)
 * @param d_vel2     Velocity-squared field (slab, with halos)
 * @param d_p_next   Output field (slab, with halos)
 * @param dims       Local slab GridDims (nx, ny, nz_local + 2R)
 * @param stream     CUDA stream
 */
void launch_slab(
    const real_t*   d_p_cur,
    const real_t*   d_p_prev,
    const real_t*   d_vel2,
          real_t*   d_p_next,
    const GridDims& dims,
    cudaStream_t    stream)
{
    NVTX_RANGE("launch_slab");
    // Reuse the tiled kernel — it already handles interior-only computation
    // and treats the Z-face halos as read-only boundary data.
    launch_tiled(d_p_cur, d_p_prev, d_vel2, d_p_next, dims, stream);
}

}  // namespace stencil

#else  // CPU fallback

namespace stencil {
void launch_slab(
    const real_t* p_cur, const real_t* p_prev, const real_t* vel2,
          real_t* p_next, const GridDims& dims, cudaStream_t stream)
{
    launch_tiled(p_cur, p_prev, vel2, p_next, dims, stream);
}
}  // namespace stencil

#endif  // STENCIL_CPU_FALLBACK
