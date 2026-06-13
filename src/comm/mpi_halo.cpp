/**
 * src/comm/mpi_halo.cpp
 *
 * MpiHaloExchange — Z-face halo exchange via MPI_Sendrecv.
 *
 * Used for multi-node runs where NCCL GPU-direct is not available across
 * the network fabric, or as a portable fallback for testing.
 *
 * Protocol:
 *   Each step requires two host-staged exchanges:
 *     1. bottom: rank sends [R, 2R) → rank-1, receives into [0, R) from rank-1
 *     2. top:    rank sends [nz-2R, nz-R) → rank+1, receives into [nz-R, nz)
 *
 * Staging strategy:
 *   Device → host copy (cudaMemcpy D→H) → MPI_Sendrecv → host → device copy.
 *   Double-buffered: two pinned host staging buffers to overlap D→H with send.
 *
 * Performance notes:
 *   - For intra-node communication, NCCL is ~3–5× faster (no host staging).
 *   - MPI path is used for multi-node (InfiniBand / RoCE) where GPU-direct
 *     RDMA is not guaranteed.
 *   - With CUDA-aware MPI (OpenMPI 4+ with UCX) the D→H copy can be skipped.
 */

#include "stencil/comm.hpp"
#include "stencil/types.hpp"

#ifdef ENABLE_MPI
#  include <mpi.h>
#endif

#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace stencil {

#ifdef ENABLE_MPI

// ── MpiHaloExchange ───────────────────────────────────────────────────────────
class MpiHaloExchange final : public IHaloExchange {
public:
    MpiHaloExchange(int rank, int nranks, bool cuda_aware)
        : IHaloExchange(rank, nranks), cuda_aware_(cuda_aware) {}

    ~MpiHaloExchange() override {
        free_staging();
    }

    std::string name() const override {
        return cuda_aware_ ? "mpi-cuda-aware" : "mpi-staged";
    }

    void exchange(real_t* d_field, const GridDims& dims,
                  cudaStream_t stream) override
    {
        const int R = STENCIL_RADIUS;
        const std::size_t plane = dims.nx * dims.ny;
        const std::size_t halo  = static_cast<std::size_t>(R) * plane;
        const std::size_t nz    = dims.nz;

        ensure_staging(halo);

        const int down = (rank_ > 0)           ? rank_ - 1 : MPI_PROC_NULL;
        const int up   = (rank_ < nranks_ - 1) ? rank_ + 1 : MPI_PROC_NULL;
        const int tag_down = 0, tag_up = 1;

        // ── Bottom exchange (send down / recv from down) ───────────────────
        real_t* d_bot_halo     = d_field;
        real_t* d_bot_interior = d_field + halo;

        if (!cuda_aware_) {
#ifndef STENCIL_CPU_FALLBACK
            // D→H: copy interior face to send buffer
            CUDA_CHECK(cudaMemcpyAsync(h_send_bot_.data(), d_bot_interior,
                                       halo * sizeof(real_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
        }

        MPI_Sendrecv(
            cuda_aware_ ? d_bot_interior : h_send_bot_.data(),
            static_cast<int>(halo), mpi_real_t(), down, tag_down,
            cuda_aware_ ? d_bot_halo : h_recv_bot_.data(),
            static_cast<int>(halo), mpi_real_t(), down, tag_up,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (!cuda_aware_) {
#ifndef STENCIL_CPU_FALLBACK
            CUDA_CHECK(cudaMemcpyAsync(d_bot_halo, h_recv_bot_.data(),
                                       halo * sizeof(real_t),
                                       cudaMemcpyHostToDevice, stream));
#endif
        }

        // ── Top exchange (send up / recv from up) ─────────────────────────
        real_t* d_top_halo     = d_field + (nz - R) * plane;
        real_t* d_top_interior = d_field + (nz - 2*R) * plane;

        if (!cuda_aware_) {
#ifndef STENCIL_CPU_FALLBACK
            CUDA_CHECK(cudaMemcpyAsync(h_send_top_.data(), d_top_interior,
                                       halo * sizeof(real_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
        }

        MPI_Sendrecv(
            cuda_aware_ ? d_top_interior : h_send_top_.data(),
            static_cast<int>(halo), mpi_real_t(), up, tag_up,
            cuda_aware_ ? d_top_halo : h_recv_top_.data(),
            static_cast<int>(halo), mpi_real_t(), up, tag_down,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (!cuda_aware_) {
#ifndef STENCIL_CPU_FALLBACK
            CUDA_CHECK(cudaMemcpyAsync(d_top_halo, h_recv_top_.data(),
                                       halo * sizeof(real_t),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
        }
    }

private:
    bool   cuda_aware_;
    std::size_t staging_size_ = 0;

    // Pinned host staging buffers (only used when !cuda_aware_)
    std::vector<real_t> h_send_bot_, h_recv_bot_;
    std::vector<real_t> h_send_top_, h_recv_top_;

    static MPI_Datatype mpi_real_t() {
        return (sizeof(real_t) == 8) ? MPI_DOUBLE : MPI_FLOAT;
    }

    void ensure_staging(std::size_t halo) {
        if (halo <= staging_size_) return;
        free_staging();
#ifndef STENCIL_CPU_FALLBACK
        // Allocate pinned memory for lower latency H2D/D2H
        auto alloc = [&](std::vector<real_t>& v) {
            v.resize(halo);
            cudaHostRegister(v.data(), halo * sizeof(real_t),
                             cudaHostRegisterDefault);
        };
        alloc(h_send_bot_); alloc(h_recv_bot_);
        alloc(h_send_top_); alloc(h_recv_top_);
#else
        h_send_bot_.resize(halo); h_recv_bot_.resize(halo);
        h_send_top_.resize(halo); h_recv_top_.resize(halo);
#endif
        staging_size_ = halo;
    }

    void free_staging() {
#ifndef STENCIL_CPU_FALLBACK
        for (auto* v : {&h_send_bot_, &h_recv_bot_,
                        &h_send_top_, &h_recv_top_}) {
            if (!v->empty()) {
                cudaHostUnregister(v->data());
                v->clear();
            }
        }
#endif
        staging_size_ = 0;
    }
};

#endif  // ENABLE_MPI

// ── Factory (called by make_halo_exchange) ────────────────────────────────────
std::unique_ptr<IHaloExchange> make_mpi_exchange(int rank, int nranks) {
#ifdef ENABLE_MPI
    // Detect CUDA-aware MPI via environment variable (set by mpirun or user)
    const char* cuda_aware_env = std::getenv("OMPI_MCA_opal_cuda_support");
    const bool cuda_aware = (cuda_aware_env && std::strcmp(cuda_aware_env, "true") == 0);

    if (rank == 0) {
        std::printf("[comm] MPI halo exchange: %s\n",
                    cuda_aware ? "CUDA-aware" : "host-staged");
    }

    return std::make_unique<MpiHaloExchange>(rank, nranks, cuda_aware);
#else
    (void)rank; (void)nranks;
    throw std::runtime_error(
        "MPI not available — rebuild with -DSTENCIL_ENABLE_MPI=ON");
#endif
}

// ── make_halo_exchange — top-level factory (declared in comm.hpp) ─────────────
std::unique_ptr<IHaloExchange> make_null_exchange();   // defined in nccl_exchange.cpp
std::unique_ptr<IHaloExchange> make_nccl_exchange(int rank, int nranks);

std::unique_ptr<IHaloExchange> make_halo_exchange(
    int rank, int nranks, const Config& cfg)
{
    if (nranks <= 1) {
        return make_null_exchange();
    }

#if defined(ENABLE_NCCL)
    (void)cfg;
    return make_nccl_exchange(rank, nranks);
#elif defined(ENABLE_MPI)
    (void)cfg;
    return make_mpi_exchange(rank, nranks);
#else
    (void)rank; (void)nranks; (void)cfg;
    throw std::runtime_error(
        "Multi-GPU requires MPI or NCCL — enable with CMake options");
#endif
}

}  // namespace stencil
