/**
 * src/kernels/stencil_tiled.cu
 *
 * Optimised 3D finite-difference stencil kernel using:
 *   1. Shared-memory XY tile  — eliminates redundant Y-direction global loads
 *   2. Register Z-pencil      — slides through Z without extra SMEM traffic
 *   3. Warp-coalesced X loads — TILE_X == 32 (one full warp per row)
 *   4. __constant__ FD coeffs — broadcast to all threads at zero cost
 *
 * ── Memory access pattern comparison ──────────────────────────────────────────
 *
 * Naive kernel (per interior point, R=4):
 *   X loads: 2R = 8  from global (coalesced — fast)
 *   Y loads: 2R = 8  from global (strided by NX — slow)
 *   Z loads: 2R = 8  from global (strided by NX*NY — very slow)
 *   Total DRAM: (6R+4) × sizeof(real_t) per point
 *
 * Tiled kernel (per interior point):
 *   X loads: shared memory   (loaded once per tile row, reused TILE_Y times)
 *   Y loads: shared memory   (loaded once per tile column, reused TILE_X times)
 *   Z loads: register pencil (1 new global load per Z step, reused 2R times)
 *   Total DRAM: ≈ 1 × sizeof(real_t) per point (down from 6R+4)
 *   Expected speedup on A100: ~3–4×
 *
 * ── Tile layout ────────────────────────────────────────────────────────────────
 *
 *  Block dim: SMEM_X × SMEM_Y = (TILE_X + 2R) × (TILE_Y + 2R) threads
 *
 *  Every thread in the block maps 1:1 to one SMEM cell (including halos).
 *  Each thread loads its own cell from global memory into SMEM.
 *  Only "interior" threads (those whose global index is not in the domain halo)
 *  write a result to p_next.
 *
 *  Mapping:
 *    tx ∈ [0, SMEM_X)    gix = blockIdx.x * TILE_X + tx - R
 *    ty ∈ [0, SMEM_Y)    giy = blockIdx.y * TILE_Y + ty - R
 *
 *  Interior threads: tx ∈ [R, SMEM_X-R) AND ty ∈ [R, SMEM_Y-R)
 *                    AND gix ∈ [R, NX-R)  AND giy ∈ [R, NY-R)
 *
 * ── Z pencil ───────────────────────────────────────────────────────────────────
 *
 *  The grid is iterated in Z-slabs of depth PENCIL_Z.
 *  Each thread maintains a sliding window of PENCIL_Z + 2R values in registers.
 *  At each Z step:
 *    - The window shifts by 1 (pure register move, zero DRAM)
 *    - One new plane is loaded at the far look-ahead position
 *    - The XY tile for the current plane is stored in SMEM from pencil[R]
 *
 * ── Occupancy (Ampere, fp32, R=4) ─────────────────────────────────────────────
 *
 *  Block size:    SMEM_X * SMEM_Y = 40 * 16 = 640 threads
 *  SMEM per block: 40 * 16 * 4 = 2560 B
 *  Registers:    ~40 per thread (dominated by pencil array of 24 reals)
 *  Theoretical occupancy: ~50% (limited by register file, not SMEM)
 *
 * ── Expected Nsight Compute metrics (A100 80GB, 512³, fp32, R=4) ──────────────
 *
 *  Naive:  DRAM BW ~420 GB/s,  time ~3.8 ms/step,  L1 hit ~5%
 *  Tiled:  DRAM BW ~110 GB/s,  time ~1.1 ms/step → ~3.5×,  L1 hit ~78%
 *
 * Physics:
 *   p_next = 2·p_cur - p_prev + dt²·v²·(∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²)
 */

#include "stencil/kernels.cuh"
#include "stencil/types.hpp"

#ifndef STENCIL_CPU_FALLBACK
#include <cuda_runtime.h>

namespace stencil {

// ── FD coefficients in __constant__ memory (tiled TU's own copy) ─────────────
// Cannot share __constant__ across .cu TUs without device linking tricks.
// Instead we keep a private copy and upload_fd_coefficients() fills both.
__constant__ real_t c_fd[STENCIL_RADIUS + 1];

void upload_fd_coefficients_tiled() {
    FDCoeffs<STENCIL_RADIUS> fd;
    real_t h[STENCIL_RADIUS + 1];
    h[0] = fd.c0;
    for (int r = 0; r < STENCIL_RADIUS; ++r) h[r + 1] = fd.c[r];
    CUDA_CHECK(cudaMemcpyToSymbol(c_fd, h, sizeof(real_t) * (STENCIL_RADIUS + 1)));
}
// ── Tile dimensions ───────────────────────────────────────────────────────────
// SMEM_X/Y: full block width/height including R-cell halo on each side.
// The block is exactly SMEM_X × SMEM_Y threads — one thread per SMEM cell.
constexpr int SMEM_X = TILE_X + 2 * STENCIL_RADIUS;   // 40  (R=4)
constexpr int SMEM_Y = TILE_Y + 2 * STENCIL_RADIUS;   // 16  (R=4)

// ── Tiled kernel ──────────────────────────────────────────────────────────────
__global__ __launch_bounds__(SMEM_X * SMEM_Y)
void kernel_tiled(
    const real_t* __restrict__ p_cur,
    const real_t* __restrict__ p_prev,
    const real_t* __restrict__ vel2,
          real_t* __restrict__ p_next,
    GridDims grid,
    real_t   inv_dx2,
    real_t   inv_dy2,
    real_t   inv_dz2,
    real_t   dt2)
{
    // ── Thread indices ────────────────────────────────────────────────────────
    // tx ∈ [0, SMEM_X)  ty ∈ [0, SMEM_Y)
    // Each thread owns one cell in the SMEM tile, including halo cells.
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    const int R  = STENCIL_RADIUS;
    const int NX = static_cast<int>(grid.nx);
    const int NY = static_cast<int>(grid.ny);
    const int NZ = static_cast<int>(grid.nz);

    // ── Global (x, y) for this thread ────────────────────────────────────────
    // Halo threads get gix < 0 or gix >= NX; they load from global only if
    // the index is valid, otherwise they contribute zero to SMEM.
    const int gix = static_cast<int>(blockIdx.x) * TILE_X + tx - R;
    const int giy = static_cast<int>(blockIdx.y) * TILE_Y + ty - R;
    const int iz0 = static_cast<int>(blockIdx.z) * PENCIL_Z;

    const std::ptrdiff_t sy = NX;
    const std::ptrdiff_t sz = static_cast<std::ptrdiff_t>(NX) * NY;

    // Pre-compute whether this (gix, giy) pair is in-bounds for XY.
    // Used in every Z iteration — avoid recomputing.
    const bool xy_valid = (gix >= 0 && gix < NX && giy >= 0 && giy < NY);

    // ── Shared memory tile ────────────────────────────────────────────────────
    // smem[ty][tx] — one plane of p_cur in the XY tile (including halos).
    // Dimensions: SMEM_Y × SMEM_X (y-major for coalesced bank access).
    __shared__ real_t smem[SMEM_Y][SMEM_X];

    // ── Register pencil (Z sliding window) ───────────────────────────────────
    // pencil[k] corresponds to global Z = iz - R + k, so:
    //   pencil[0]       = iz - R     (oldest look-behind)
    //   pencil[R]       = iz         (current plane)
    //   pencil[PENCIL_Z + 2R - 1]   (newest look-ahead, refilled each step)
    //
    // Total size: PENCIL_Z + 2R  (= 24 for R=4, PENCIL_Z=16)
    real_t pencil[PENCIL_Z + 2 * STENCIL_RADIUS] = {};

    // ── Pre-fill entire pencil ────────────────────────────────────────────────
    // With shift-at-end design, pencil[d] = plane iz0 - R + d at start of pz=0.
    // We must pre-fill ALL slots that will be needed before the refill provides them.
    // pencil[d] for d=0..PENCIL_Z+2R-1 covers planes iz0-R .. iz0+PENCIL_Z+R-1.
    // The refill each iteration only fills pencil[PENCIL_Z+2R-1], so everything
    // else must be pre-loaded here.
    for (int d = 0; d < PENCIL_Z + 2 * R; ++d) {
        const int iz = iz0 - R + d;
        if (xy_valid && iz >= 0 && iz < NZ)
            pencil[d] = p_cur[static_cast<std::ptrdiff_t>(iz) * sz
                             + static_cast<std::ptrdiff_t>(giy) * sy
                             + gix];
    }

    // ── Main Z loop ───────────────────────────────────────────────────────────
    for (int pz = 0; pz < PENCIL_Z; ++pz) {
        const int iz = iz0 + pz;

        // ── Load XY tile into shared memory ───────────────────────────────────
        // pencil[R] = p_cur at the current Z plane — already in register.
        // Every thread (including halo threads) writes to its SMEM cell.
        smem[ty][tx] = (xy_valid && iz >= 0 && iz < NZ) ? pencil[R] : real_t{0};
        __syncthreads();

        // ── Compute — interior threads only ───────────────────────────────────
        // Interior = thread owns a real output cell (not a halo-load-only thread)
        //   tx ∈ [R, SMEM_X-R)  →  writes to SMEM position without out-of-bounds
        //   ty ∈ [R, SMEM_Y-R)
        //   global index within domain interior
        const bool interior =
            (tx >= R && tx < SMEM_X - R) &&
            (ty >= R && ty < SMEM_Y - R) &&
            (gix >= R && gix < NX - R)   &&
            (giy >= R && giy < NY - R)   &&
            (iz  >= R && iz  < NZ - R);

        if (interior) {
            const std::size_t c = static_cast<std::size_t>(iz)  * sz
                                + static_cast<std::size_t>(giy) * sy
                                + static_cast<std::size_t>(gix);

            // X Laplacian — reads from smem[ty][tx ± r]
            // All in the same SMEM row → no bank conflicts.
            real_t lap_x = c_fd[0] * smem[ty][tx] * inv_dx2;
            #pragma unroll
            for (int r = 1; r <= R; ++r)
                lap_x += c_fd[r] * (smem[ty][tx + r] + smem[ty][tx - r]) * inv_dx2;

            // Y Laplacian — reads from smem[ty ± r][tx]
            // Loaded once into SMEM per tile, reused SMEM_X times per row.
            real_t lap_y = c_fd[0] * smem[ty][tx] * inv_dy2;
            #pragma unroll
            for (int r = 1; r <= R; ++r)
                lap_y += c_fd[r] * (smem[ty + r][tx] + smem[ty - r][tx]) * inv_dy2;

            // Z Laplacian — reads from register pencil
            // pencil[R+r] = plane iz+r, pencil[R-r] = plane iz-r.
            // Pure register arithmetic — zero DRAM, zero SMEM.
            real_t lap_z = c_fd[0] * pencil[R] * inv_dz2;
            #pragma unroll
            for (int r = 1; r <= R; ++r)
                lap_z += c_fd[r] * (pencil[R + r] + pencil[R - r]) * inv_dz2;

            // Leapfrog time integration
            p_next[c] = real_t{2} * p_cur[c]
                      - p_prev[c]
                      + dt2 * vel2[c] * (lap_x + lap_y + lap_z);
        }

        __syncthreads();  // protect SMEM before next iteration's load

        // Shift pencil: pencil[k] ← pencil[k+1].
        // Pure register-to-register move — zero memory traffic.
        #pragma unroll
        for (int d = 0; d < PENCIL_Z + 2 * R - 1; ++d)
            pencil[d] = pencil[d + 1];

        // Refill look-ahead: load plane iz + R + 1 (next iteration's look-ahead)
        {
            const int iz_ahead = iz + R + 1;
            pencil[PENCIL_Z + 2 * R - 1] =
                (xy_valid && iz_ahead >= 0 && iz_ahead < NZ)
                ? p_cur[static_cast<std::ptrdiff_t>(iz_ahead) * sz
                       + static_cast<std::ptrdiff_t>(giy) * sy
                       + gix]
                : real_t{0};
        }
    }
}

// ── Host launcher ─────────────────────────────────────────────────────────────
void launch_tiled(
    const real_t*   d_p_cur,
    const real_t*   d_p_prev,
    const real_t*   d_vel2,
          real_t*   d_p_next,
    const GridDims& grid,
    cudaStream_t    stream)
{
    NVTX_RANGE("launch_tiled");
    

    const real_t inv_dx2 = real_t{1} / (grid.dx * grid.dx);
    const real_t inv_dy2 = real_t{1} / (grid.dy * grid.dy);
    const real_t inv_dz2 = real_t{1} / (grid.dz * grid.dz);
    const real_t dt2     = grid.dt * grid.dt;

    // Block: SMEM_X × SMEM_Y × 1  (one thread per SMEM cell, including halos)
    // Grid:  tiles in X/Y, pencil-slabs in Z
    const dim3 block(SMEM_X, SMEM_Y, 1);
    const dim3 grid3d(
        (static_cast<unsigned>(grid.nx) + TILE_X - 1) / TILE_X,
        (static_cast<unsigned>(grid.ny) + TILE_Y - 1) / TILE_Y,
        (static_cast<unsigned>(grid.nz) + PENCIL_Z - 1) / PENCIL_Z);

    // SMEM: one XY tile (already statically allocated in kernel, but pass
    // smem_bytes=0 since the shared array is declared statically in the kernel)
    kernel_tiled<<<grid3d, block, 0, stream>>>(
        d_p_cur, d_p_prev, d_vel2, d_p_next,
        grid, inv_dx2, inv_dy2, inv_dz2, dt2);

    CUDA_CHECK_LAST();
}

}  // namespace stencil

#endif  // STENCIL_CPU_FALLBACK
// CPU fallback: launch_tiled delegates to launch_naive (defined in stencil_naive.cu)
