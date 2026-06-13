/**
 * src/comm/nccl_exchange.cpp
 *
 * NcclHaloExchange — GPU-direct Z-face halo exchange using NCCL.
 *
 * Why NCCL over plain cudaMemcpyPeer:
 *   - NCCL automatically selects NVLink > PCIe P2P > SHM based on topology
 *   - Collective operations (allreduce for reductions) use ring/tree algorithms
 *   - No host staging — data never leaves the GPU memory subsystem
 *
 * Protocol (two-phase, non-blocking):
 *   1. ncclGroupStart()
 *   2. ncclSend: bottom interior face → rank-1 top halo slot
 *      ncclRecv: top halo slot        ← rank+1 bottom interior face
 *      ncclSend: top interior face    → rank+1 bottom halo slot
 *      ncclRecv: bottom halo slot     ← rank-1 top interior face
 *   3. ncclGroupEnd()
 *
 * Memory layout (Z-major slab):
 *   Field index:  iz * nx * ny + iy * nx + ix
 *   Bottom halo:  iz ∈ [0, R)
 *   Bottom interior face: iz ∈ [R, 2R)        → send to rank-1
 *   Top interior face:    iz ∈ [nz-2R, nz-R)  → send to rank+1
 *   Top halo:     iz ∈ [nz-R, nz)
 *
 * Boundary conditions:
 *   rank 0:       no bottom neighbour → bottom halo stays zero (Dirichlet)
 *   rank nranks-1: no top neighbour  → top halo stays zero
 */

#include "stencil/comm.hpp"
#include "stencil/types.hpp"

#ifdef ENABLE_NCCL
#  include <nccl.h>

#  define NCCL_CHECK(call)                                                  \
     do {                                                                   \
       ncclResult_t _r = (call);                                            \
       if (_r != ncclSuccess) {                                             \
         fprintf(stderr, "[NCCL ERROR] %s:%d  %s\n",                       \
                 __FILE__, __LINE__, ncclGetErrorString(_r));               \
         std::exit(EXIT_FAILURE);                                           \
       }                                                                    \
     } while (0)
#endif

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace stencil {

#ifdef ENABLE_NCCL

// ── NcclHaloExchange ──────────────────────────────────────────────────────────
class NcclHaloExchange final : public IHaloExchange {
public:
    NcclHaloExchange(int rank, int nranks, ncclComm_t comm)
        : IHaloExchange(rank, nranks), comm_(comm) {}

    ~NcclHaloExchange() override {
        if (comm_) ncclCommDestroy(comm_);
    }

    std::string name() const override { return "nccl"; }

    void exchange(real_t* d_field, const GridDims& dims,
                  cudaStream_t stream) override
    {
        const int R  = STENCIL_RADIUS;
        const int nz = static_cast<int>(dims.nz);
        const std::size_t plane = dims.nx * dims.ny;   // elements per XY plane
        const std::size_t halo  = static_cast<std::size_t>(R) * plane;

        // Pointers to each region of the slab
        real_t* bot_halo     = d_field;                          // iz ∈ [0,  R)
        real_t* bot_interior = d_field + halo;                   // iz ∈ [R,  2R)
        real_t* top_interior = d_field + (nz - 2*R) * plane;    // iz ∈ [nz-2R, nz-R)
        real_t* top_halo     = d_field + (nz - R)   * plane;    // iz ∈ [nz-R, nz)

        // Derive peer ranks (-1 = no neighbour)
        const int down = (rank_ > 0)          ? rank_ - 1 : -1;
        const int up   = (rank_ < nranks_ - 1) ? rank_ + 1 : -1;

        // ── All four ops inside one NCCL group for maximum overlap ────────────
        NCCL_CHECK(ncclGroupStart());

        // Send bottom interior face down; receive top halo from down
        if (down >= 0) {
            NCCL_CHECK(ncclSend(bot_interior, halo,
                                ncclFloat,   // real_t == float (fp32 default)
                                down, comm_, stream));
            NCCL_CHECK(ncclRecv(bot_halo, halo,
                                ncclFloat,
                                down, comm_, stream));
        }

        // Send top interior face up; receive bottom halo from up
        if (up >= 0) {
            NCCL_CHECK(ncclSend(top_interior, halo,
                                ncclFloat,
                                up, comm_, stream));
            NCCL_CHECK(ncclRecv(top_halo, halo,
                                ncclFloat,
                                up, comm_, stream));
        }

        NCCL_CHECK(ncclGroupEnd());
        // ncclGroupEnd is asynchronous w.r.t. the CUDA stream —
        // the kernel launched on the same stream after this call
        // will wait for the NCCL ops to complete automatically.
    }

private:
    ncclComm_t comm_ = nullptr;
};

#endif  // ENABLE_NCCL

// ── CPU-fallback stub ─────────────────────────────────────────────────────────
// Used when NCCL is not available (single-GPU runs, CI without NCCL).
class NullHaloExchange final : public IHaloExchange {
public:
    NullHaloExchange() : IHaloExchange(0, 1) {}
    std::string name() const override { return "null"; }
    void exchange(real_t*, const GridDims&, cudaStream_t) override {}
};

// ── Factory helpers (called by make_halo_exchange in comm.hpp) ────────────────
std::unique_ptr<IHaloExchange> make_nccl_exchange(int rank, int nranks) {
#ifdef ENABLE_NCCL
    // Initialise NCCL communicator.
    // In a real multi-node setup ncclUniqueId is broadcast via MPI;
    // here we use ncclCommInitAll which handles single-node automatically.
    std::vector<ncclComm_t> comms(nranks);
    std::vector<int> devs(nranks);
    for (int i = 0; i < nranks; ++i) devs[i] = i;

    NCCL_CHECK(ncclCommInitAll(comms.data(), nranks, devs.data()));
    return std::make_unique<NcclHaloExchange>(rank, nranks, comms[rank]);
#else
    (void)rank; (void)nranks;
    throw std::runtime_error(
        "NCCL not available — rebuild with -DSTENCIL_ENABLE_NCCL=ON");
#endif
}

std::unique_ptr<IHaloExchange> make_null_exchange() {
    return std::make_unique<NullHaloExchange>();
}

}  // namespace stencil
