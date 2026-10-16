/**
 * src/kernels/stencil_tiled.cu
 *
 * Shared-memory + register-pencil 3D stencil, v2: 32 x TILE_Y blocks where every
 * thread owns one output column. Halo cells (axis-aligned only; the star stencil
 * never reads corners) are loaded by the first R threads of each row/column.
 *
 *   smem[TILE_Y + 2R][TILE_X + 2R]   one XY plane of p_cur
 *   pencil[PENCIL_Z + 2R]            per-thread Z sliding window (registers)
 *
 * p_next = 2*p_cur - p_prev + dt^2 * v^2 * laplacian(p_cur)
 */

#include "stencil/kernels.cuh"
#include "stencil/types.hpp"

#ifndef STENCIL_CPU_FALLBACK
#include <cuda_runtime.h>

namespace stencil {

// Private copy of the FD coefficients (filled by upload_fd_coefficients()).
__constant__ real_t c_fd[STENCIL_RADIUS + 1];

void upload_fd_coefficients_tiled() {
    FDCoeffs<STENCIL_RADIUS> fd;
    real_t h[STENCIL_RADIUS + 1];
    h[0] = fd.c0;
    for (int r = 0; r < STENCIL_RADIUS; ++r)
        h[r + 1] = fd.c[r];
    CUDA_CHECK(cudaMemcpyToSymbol(c_fd, h, sizeof(real_t) * (STENCIL_RADIUS + 1)));
}

constexpr int SMEM_X = TILE_X + 2 * STENCIL_RADIUS;
constexpr int SMEM_Y = TILE_Y + 2 * STENCIL_RADIUS;

__global__ __launch_bounds__(TILE_X* TILE_Y) void kernel_tiled(
    const real_t* __restrict__ p_cur, const real_t* __restrict__ p_prev,
    const real_t* __restrict__ vel2, real_t* __restrict__ p_next, GridDims grid, real_t inv_dx2,
    real_t inv_dy2, real_t inv_dz2, real_t dt2) {
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int R  = STENCIL_RADIUS;
    const int NX = static_cast<int>(grid.nx);
    const int NY = static_cast<int>(grid.ny);
    const int NZ = static_cast<int>(grid.nz);

    // This thread's output column. The tile is offset by R so that no thread
    // starts inside the domain halo.
    const int bx  = static_cast<int>(blockIdx.x) * TILE_X;
    const int by  = static_cast<int>(blockIdx.y) * TILE_Y;
    const int gix = bx + tx + R;
    const int giy = by + ty + R;
    const int iz0 = static_cast<int>(blockIdx.z) * PENCIL_Z;

    const std::ptrdiff_t sy = NX;
    const std::ptrdiff_t sz = static_cast<std::ptrdiff_t>(NX) * NY;

    const bool xy_valid = (gix < NX && giy < NY);          // may supply data
    const bool xy_out   = (gix < NX - R && giy < NY - R);  // owns an output cell

    __shared__ real_t smem[SMEM_Y][SMEM_X];

    real_t pencil[PENCIL_Z + 2 * STENCIL_RADIUS] = {};

    // Pre-fill: pencil[d] = plane (iz0 - R + d) of this thread's own column.
    for (int d = 0; d < PENCIL_Z + 2 * R; ++d) {
        const int iz = iz0 - R + d;
        if (xy_valid && iz >= 0 && iz < NZ)
            pencil[d] = p_cur[static_cast<std::ptrdiff_t>(iz) * sz +
                              static_cast<std::ptrdiff_t>(giy) * sy + gix];
    }

    for (int pz = 0; pz < PENCIL_Z; ++pz) {
        const int iz       = iz0 + pz;
        const bool z_valid = (iz < NZ);

        // Own cell from the register pencil.
        smem[ty + R][tx + R] = (xy_valid && z_valid) ? pencil[R] : real_t{0};

        if (z_valid) {
            const std::ptrdiff_t zoff = static_cast<std::ptrdiff_t>(iz) * sz;
            // Left / right halo columns of this row, loaded by threads tx < R.
            if (tx < R) {
                const int xl     = bx + tx;               // smem column tx
                const int xr     = bx + R + TILE_X + tx;  // smem column TILE_X + R + tx
                smem[ty + R][tx] = (giy < NY && xl < NX)
                                       ? p_cur[zoff + static_cast<std::ptrdiff_t>(giy) * sy + xl]
                                       : real_t{0};
                smem[ty + R][TILE_X + R + tx] =
                    (giy < NY && xr < NX) ? p_cur[zoff + static_cast<std::ptrdiff_t>(giy) * sy + xr]
                                          : real_t{0};
            }
            // Top / bottom halo rows of this column, loaded by threads ty < R.
            if (ty < R) {
                const int yt     = by + ty;               // smem row ty
                const int yb     = by + R + TILE_Y + ty;  // smem row TILE_Y + R + ty
                smem[ty][tx + R] = (gix < NX && yt < NY)
                                       ? p_cur[zoff + static_cast<std::ptrdiff_t>(yt) * sy + gix]
                                       : real_t{0};
                smem[TILE_Y + R + ty][tx + R] =
                    (gix < NX && yb < NY) ? p_cur[zoff + static_cast<std::ptrdiff_t>(yb) * sy + gix]
                                          : real_t{0};
            }
        }
        __syncthreads();

        if (xy_out && iz >= R && iz < NZ - R) {
            const std::size_t c = static_cast<std::size_t>(iz) * sz +
                                  static_cast<std::size_t>(giy) * sy +
                                  static_cast<std::size_t>(gix);

            real_t lap_x = c_fd[0] * smem[ty + R][tx + R] * inv_dx2;
#pragma unroll
            for (int r = 1; r <= R; ++r)
                lap_x += c_fd[r] * (smem[ty + R][tx + R + r] + smem[ty + R][tx + R - r]) * inv_dx2;

            real_t lap_y = c_fd[0] * smem[ty + R][tx + R] * inv_dy2;
#pragma unroll
            for (int r = 1; r <= R; ++r)
                lap_y += c_fd[r] * (smem[ty + R + r][tx + R] + smem[ty + R - r][tx + R]) * inv_dy2;

            real_t lap_z = c_fd[0] * pencil[R] * inv_dz2;
#pragma unroll
            for (int r = 1; r <= R; ++r)
                lap_z += c_fd[r] * (pencil[R + r] + pencil[R - r]) * inv_dz2;

            p_next[c] = real_t{2} * pencil[R] - p_prev[c] + dt2 * vel2[c] * (lap_x + lap_y + lap_z);
        }

        __syncthreads();  // protect smem before the next plane's stores

#pragma unroll
        for (int d = 0; d < PENCIL_Z + 2 * R - 1; ++d)
            pencil[d] = pencil[d + 1];

        {
            const int iz_ahead           = iz + R + 1;
            pencil[PENCIL_Z + 2 * R - 1] = (xy_valid && iz_ahead < NZ)
                                               ? p_cur[static_cast<std::ptrdiff_t>(iz_ahead) * sz +
                                                       static_cast<std::ptrdiff_t>(giy) * sy + gix]
                                               : real_t{0};
        }
    }
}

void launch_tiled(const real_t* d_p_cur, const real_t* d_p_prev, const real_t* d_vel2,
                  real_t* d_p_next, const GridDims& grid, cudaStream_t stream) {
    NVTX_RANGE("launch_tiled");

    const real_t inv_dx2 = real_t{1} / (grid.dx * grid.dx);
    const real_t inv_dy2 = real_t{1} / (grid.dy * grid.dy);
    const real_t inv_dz2 = real_t{1} / (grid.dz * grid.dz);
    const real_t dt2     = grid.dt * grid.dt;

    const std::size_t R2 = 2 * static_cast<std::size_t>(STENCIL_RADIUS);
    const std::size_t ix = grid.nx > R2 ? grid.nx - R2 : 1;
    const std::size_t iy = grid.ny > R2 ? grid.ny - R2 : 1;

    const dim3 block(TILE_X, TILE_Y, 1);
    const dim3 grid3d(static_cast<unsigned>((ix + TILE_X - 1) / TILE_X),
                      static_cast<unsigned>((iy + TILE_Y - 1) / TILE_Y),
                      static_cast<unsigned>((grid.nz + PENCIL_Z - 1) / PENCIL_Z));

    kernel_tiled<<<grid3d, block, 0, stream>>>(d_p_cur, d_p_prev, d_vel2, d_p_next, grid, inv_dx2,
                                               inv_dy2, inv_dz2, dt2);
    CUDA_CHECK_LAST();
}

}  // namespace stencil

#endif  // STENCIL_CPU_FALLBACK
// CPU fallback: launch_tiled delegates to launch_naive (defined in stencil_naive.cu)
