#pragma once

/**
 * include/stencil/comm.hpp
 *
 * IHaloExchange — abstract interface for halo communication between GPUs.
 *
 * The multi-GPU solver decomposes the domain along Z into slabs, one per
 * rank/GPU.  After each kernel step the Z-face halos (STENCIL_RADIUS planes
 * on the bottom and top of each slab) must be filled with data from
 * neighbouring ranks before the next step can begin.
 *
 * Two concrete implementations:
 *   NcclHaloExchange  — GPU-direct P2P via NCCL (single-node, fastest)
 *   MpiHaloExchange   — MPI_Sendrecv via host staging (multi-node)
 *
 * Usage:
 *   auto comm = make_halo_exchange(cfg, rank, nranks, local_dims, stream);
 *   comm->exchange(d_field, local_dims);   // call after each kernel step
 *
 * Domain decomposition (Z-slab):
 *
 *   Global Z: [0, NZ_global)
 *   Rank r owns Z-slab: [r * NZ_local, (r+1) * NZ_local)
 *   With halos:         [r * NZ_local - R, (r+1) * NZ_local + R)
 *
 *   After each step, rank r sends:
 *     its bottom interior face [R .. 2R) → rank r-1 top halo [NZ_local+R .. NZ_local+2R)
 *     its top    interior face [NZ_local-R .. NZ_local) → rank r+1 bottom halo [0 .. R)
 *
 *   Boundary ranks use Dirichlet zero (halo already zero from cudaMemset).
 */

#include "stencil/config.hpp"
#include "stencil/types.hpp"

#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>
#endif

#include <memory>
#include <string>

namespace stencil {

// ── Abstract interface ────────────────────────────────────────────────────────
class IHaloExchange {
public:
    virtual ~IHaloExchange() = default;

    IHaloExchange(const IHaloExchange&)            = delete;
    IHaloExchange& operator=(const IHaloExchange&) = delete;

    /**
     * exchange — fill this rank's Z-face halos from neighbours.
     *
     * @param d_field   Device pointer to the full (slab + halos) field.
     *                  Layout: [nz_with_halos][ny][nx], row-major.
     * @param dims      GridDims of the local slab (including halos).
     * @param stream    CUDA stream for GPU-direct ops (ignored by MPI impl).
     */
    virtual void exchange(
        real_t*         d_field,
        const GridDims& dims,
        cudaStream_t    stream = nullptr) = 0;

    /// Human-readable transport name ("nccl", "mpi").
    virtual std::string name() const = 0;

    int rank()   const noexcept { return rank_; }
    int nranks() const noexcept { return nranks_; }

protected:
    IHaloExchange(int rank, int nranks)
        : rank_(rank), nranks_(nranks) {}

    int rank_   = 0;
    int nranks_ = 1;
};

// ── Factory ───────────────────────────────────────────────────────────────────
/**
 * make_halo_exchange — create the appropriate IHaloExchange implementation.
 *
 * Selection priority:
 *   NCCL available && single-node  → NcclHaloExchange
 *   MPI available                  → MpiHaloExchange
 *   Single GPU                     → nullptr (no exchange needed)
 *
 * @param rank    MPI rank of this process (0 if MPI disabled)
 * @param nranks  Total number of MPI ranks (1 if MPI disabled)
 */
std::unique_ptr<IHaloExchange> make_halo_exchange(
    int rank,
    int nranks,
    const Config& cfg);

// ── Slab decomposition helpers ────────────────────────────────────────────────
/**
 * local_nz — number of Z cells owned by rank r (interior only, no halos).
 * Total NZ_global is split as evenly as possible across nranks.
 */
inline std::size_t local_nz(std::size_t nz_global, int rank, int nranks) {
    const std::size_t base  = nz_global / static_cast<std::size_t>(nranks);
    const std::size_t extra = nz_global % static_cast<std::size_t>(nranks);
    return base + (static_cast<std::size_t>(rank) < extra ? 1 : 0);
}

/**
 * local_dims — build GridDims for the local slab owned by this rank.
 * Adds STENCIL_RADIUS halo cells on each Z face.
 */
inline GridDims local_dims(const GridDims& global, int rank, int nranks) {
    GridDims d      = global;
    d.nz            = local_nz(global.nz, rank, nranks)
                    + 2 * static_cast<std::size_t>(STENCIL_RADIUS);
    return d;
}

}  // namespace stencil
