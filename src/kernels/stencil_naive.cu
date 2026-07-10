/**
 * src/kernels/stencil_naive.cu
 *
 * Naive global-memory 3D finite-difference stencil kernel.
 *
 * Every thread reads its 6R+1 stencil points directly from global memory
 * with no data reuse between threads.  Adjacent threads in X load
 * overlapping ranges — e.g. for R=4 and 32 threads per warp, each
 * interior point is loaded ~8 times across the warp.
 *
 * This is the BEFORE state for Nsight-driven optimisation.
 * Expected Nsight Compute findings:
 *   - DRAM bandwidth close to peak (memory-bound)
 *   - L1/L2 hit rate low (~0%)  for Z and Y directions
 *   - Achieved occupancy: moderate (~50%) due to register pressure
 *   - Memory throughput: ~40-60% of peak (uncoalesced Y/Z loads)
 *
 * Physics: acoustic wave equation (second-order leapfrog)
 *   p_next = 2·p_cur - p_prev + dt² · v²(x) · ∇²p_cur
 */

#include "stencil/kernels.cuh"
#include "stencil/types.hpp"

#ifndef STENCIL_CPU_FALLBACK
#include <cuda_runtime.h>

namespace stencil {

// ── FD coefficients in __constant__ memory ───────────────────────────────────
// Indexed: c_fd[0] = c0 (centre), c_fd[r] = c[r-1] for r=1..R
// Lives in the 64 KB constant cache — broadcast to all threads in a warp
// with zero extra memory traffic.
__constant__ real_t c_fd[STENCIL_RADIUS + 1];

// Defined in stencil_tiled.cu — uploads this TU's private c_fd copy.
void upload_fd_coefficients_tiled();

void upload_fd_coefficients() {
    FDCoeffs<STENCIL_RADIUS> fd;
    real_t h[STENCIL_RADIUS + 1];
    h[0] = fd.c0;
    for (int r = 0; r < STENCIL_RADIUS; ++r) h[r + 1] = fd.c[r];
    CUDA_CHECK(cudaMemcpyToSymbol(c_fd, h, sizeof(real_t) * (STENCIL_RADIUS + 1)));
    upload_fd_coefficients_tiled();

}

// ── Naive kernel ──────────────────────────────────────────────────────────────
__global__ void kernel_naive(
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
    const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int iz = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

    const int R  = STENCIL_RADIUS;
    const int NX = static_cast<int>(grid.nx);
    const int NY = static_cast<int>(grid.ny);
    const int NZ = static_cast<int>(grid.nz);

    // Skip halo cells — only compute interior points
    if (ix < R || ix >= NX - R ||
        iy < R || iy >= NY - R ||
        iz < R || iz >= NZ - R)
        return;

    const std::size_t sy = grid.nx;            // stride in Y
    const std::size_t sz = grid.nx * grid.ny;  // stride in Z
    const std::size_t c  = static_cast<std::size_t>(iz) * sz
                         + static_cast<std::size_t>(iy) * sy
                         + static_cast<std::size_t>(ix);

    // ── 3D Laplacian — separate axes, each with its own spacing ──────────────
    // X: stride 1 → coalesced loads across warp (fast)
    // Y: stride ny → stride ny elements between loads (slow — strided)
    // Z: stride nx*ny → very strided (slow)
    // This is the dominant bottleneck; the tiled kernel fixes Y and partial Z.
    real_t lap_x = c_fd[0] * p_cur[c] * inv_dx2;
    real_t lap_y = c_fd[0] * p_cur[c] * inv_dy2;
    real_t lap_z = c_fd[0] * p_cur[c] * inv_dz2;

    #pragma unroll
    for (int r = 1; r <= R; ++r) {
        lap_x += c_fd[r] * (p_cur[c + r]      + p_cur[c - r])      * inv_dx2;
        lap_y += c_fd[r] * (p_cur[c + r * sy] + p_cur[c - r * sy]) * inv_dy2;
        lap_z += c_fd[r] * (p_cur[c + r * sz] + p_cur[c - r * sz]) * inv_dz2;
    }

    // ── Leapfrog time integration ─────────────────────────────────────────────
    p_next[c] = real_t{2} * p_cur[c]
              - p_prev[c]
              + dt2 * vel2[c] * (lap_x + lap_y + lap_z);
}

// ── Host launcher ─────────────────────────────────────────────────────────────
void launch_naive(
    const real_t*   d_p_cur,
    const real_t*   d_p_prev,
    const real_t*   d_vel2,
          real_t*   d_p_next,
    const GridDims& grid,
    cudaStream_t    stream)
{
    NVTX_RANGE("launch_naive");

    const real_t inv_dx2 = real_t{1} / (grid.dx * grid.dx);
    const real_t inv_dy2 = real_t{1} / (grid.dy * grid.dy);
    const real_t inv_dz2 = real_t{1} / (grid.dz * grid.dz);
    const real_t dt2     = grid.dt * grid.dt;

    // Block: 32×4×2 = 256 threads.
    // X=32 ensures warp-coalesced loads in X.
    // Y=4, Z=2 give enough threads to hide memory latency without
    // excessive register pressure (tested on A100 / RTX 3090).
    const dim3 block(32, 4, 2);
    const dim3 grid3d(
        (static_cast<unsigned>(grid.nx) + block.x - 1) / block.x,
        (static_cast<unsigned>(grid.ny) + block.y - 1) / block.y,
        (static_cast<unsigned>(grid.nz) + block.z - 1) / block.z);

    kernel_naive<<<grid3d, block, 0, stream>>>(
        d_p_cur, d_p_prev, d_vel2, d_p_next,
        grid, inv_dx2, inv_dy2, inv_dz2, dt2);

    CUDA_CHECK_LAST();
}

}  // namespace stencil

// ── CPU fallback ──────────────────────────────────────────────────────────────
#else  // STENCIL_CPU_FALLBACK

#include <omp.h>
#include <cstring>

namespace stencil {

// In fallback mode, constant memory is just a plain array.
static real_t c_fd[STENCIL_RADIUS + 1];

void upload_fd_coefficients() {
    FDCoeffs<STENCIL_RADIUS> fd;
    c_fd[0] = fd.c0;
    for (int r = 0; r < STENCIL_RADIUS; ++r) c_fd[r + 1] = fd.c[r];
}

void launch_naive(
    const real_t*   p_cur,
    const real_t*   p_prev,
    const real_t*   vel2,
          real_t*   p_next,
    const GridDims& grid,
    void* /*stream*/)
{
    const int R  = STENCIL_RADIUS;
    const int NX = static_cast<int>(grid.nx);
    const int NY = static_cast<int>(grid.ny);
    const int NZ = static_cast<int>(grid.nz);

    const real_t inv_dx2 = real_t{1} / (grid.dx * grid.dx);
    const real_t inv_dy2 = real_t{1} / (grid.dy * grid.dy);
    const real_t inv_dz2 = real_t{1} / (grid.dz * grid.dz);
    const real_t dt2     = grid.dt * grid.dt;

    const std::ptrdiff_t sy = NX;
    const std::ptrdiff_t sz = NX * NY;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int iz = R; iz < NZ - R; ++iz)
    for (int iy = R; iy < NY - R; ++iy)
    for (int ix = R; ix < NX - R; ++ix) {
        const std::ptrdiff_t c = iz * sz + iy * sy + ix;

        real_t lap_x = c_fd[0] * p_cur[c] * inv_dx2;
        real_t lap_y = c_fd[0] * p_cur[c] * inv_dy2;
        real_t lap_z = c_fd[0] * p_cur[c] * inv_dz2;

        for (int r = 1; r <= R; ++r) {
            lap_x += c_fd[r] * (p_cur[c + r]      + p_cur[c - r])      * inv_dx2;
            lap_y += c_fd[r] * (p_cur[c + r * sy] + p_cur[c - r * sy]) * inv_dy2;
            lap_z += c_fd[r] * (p_cur[c + r * sz] + p_cur[c - r * sz]) * inv_dz2;
        }

        p_next[c] = real_t{2} * p_cur[c]
                  - p_prev[c]
                  + dt2 * vel2[c] * (lap_x + lap_y + lap_z);
    }
}

// Tiled launcher is the same as naive in CPU fallback (no shared memory concept)
void launch_tiled(
    const real_t* p_cur, const real_t* p_prev, const real_t* vel2,
          real_t* p_next, const GridDims& grid, void* stream) {
    launch_naive(p_cur, p_prev, vel2, p_next, grid, stream);
}

}  // namespace stencil

#endif  // STENCIL_CPU_FALLBACK
