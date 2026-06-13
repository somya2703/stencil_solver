#pragma once

/**
 * include/stencil/grid.hpp
 *
 * Host-side grid management: allocation, initialisation, and
 * host↔device transfer helpers.
 *
 * DeviceGrid owns three time-level buffers (prev / cur / next) and a
 * velocity field on the GPU, triple-buffered to avoid synchronisation
 * stalls between time steps.
 */

#include "stencil/types.hpp"
#include <algorithm>
#include <algorithm>
#include <algorithm>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef STENCIL_CPU_FALLBACK
#  include <cuda_runtime.h>
#endif

namespace stencil {

// ── Host grid (CPU memory) ────────────────────────────────────────────────────
/**
 * HostGrid — a flat std::vector wrapping a 3-D field.
 * Used for initialisation, reference solutions, and result verification.
 */
struct HostGrid {
    std::vector<real_t> data;
    GridDims            dims;

    explicit HostGrid(const GridDims& g, real_t fill = real_t{0})
        : data(g.size(), fill), dims(g) {}

    real_t& at(std::size_t x, std::size_t y, std::size_t z) {
        return data[dims.idx(x, y, z)];
    }
    const real_t& at(std::size_t x, std::size_t y, std::size_t z) const {
        return data[dims.idx(x, y, z)];
    }

    real_t*       ptr()       noexcept { return data.data(); }
    const real_t* ptr() const noexcept { return data.data(); }
    std::size_t   bytes()     const noexcept { return data.size() * sizeof(real_t); }
};

// ── Initialisation helpers ────────────────────────────────────────────────────
/**
 * init_gaussian — fill grid with a 3-D Gaussian pressure pulse.
 *
 * @param g       Grid to fill
 * @param cx,cy,cz  Pulse centre as fraction of domain [0,1]
 * @param sigma   Standard deviation in grid cells
 */
inline void init_gaussian(HostGrid& g,
                          real_t cx, real_t cy, real_t cz,
                          real_t sigma) {
    const auto& d = g.dims;
    const real_t icx = cx * static_cast<real_t>(d.nx);
    const real_t icy = cy * static_cast<real_t>(d.ny);
    const real_t icz = cz * static_cast<real_t>(d.nz);
    const real_t inv2s2 = real_t{1} / (real_t{2} * sigma * sigma);

    for (std::size_t iz = 0; iz < d.nz; ++iz)
    for (std::size_t iy = 0; iy < d.ny; ++iy)
    for (std::size_t ix = 0; ix < d.nx; ++ix) {
        const real_t dx = static_cast<real_t>(ix) - icx;
        const real_t dy = static_cast<real_t>(iy) - icy;
        const real_t dz = static_cast<real_t>(iz) - icz;
        g.at(ix, iy, iz) = std::exp(-(dx*dx + dy*dy + dz*dz) * inv2s2);
    }
}

/**
 * init_constant_velocity — fill a velocity-squared field with v^2.
 */
inline void init_constant_velocity(HostGrid& g, real_t v_ms) {
    std::fill(g.data.begin(), g.data.end(), v_ms * v_ms);
}

/**
 * init_layered_velocity — two-layer velocity model (useful for RTM tests).
 * Top half: v_top^2, bottom half: v_bot^2.
 */
inline void init_layered_velocity(HostGrid& g, real_t v_top, real_t v_bot) {
    const auto& d = g.dims;
    for (std::size_t iz = 0; iz < d.nz; ++iz)
    for (std::size_t iy = 0; iy < d.ny; ++iy)
    for (std::size_t ix = 0; ix < d.nx; ++ix) {
        const real_t v = (iz < d.nz / 2) ? v_top : v_bot;
        g.at(ix, iy, iz) = v * v;
    }
}

// ── Device grid (CUDA memory) ─────────────────────────────────────────────────
#ifndef STENCIL_CPU_FALLBACK

/**
 * DeviceGrid — owns all GPU-side memory for one solver instance.
 *
 * Triple-buffered time levels (prev/cur/next) plus a velocity-squared field.
 * Rotation via rotate() advances the buffer ring without any data copies.
 */
class DeviceGrid {
public:
    explicit DeviceGrid(const GridDims& g) : dims_(g) {
        const std::size_t n = g.size() * sizeof(real_t);
        for (auto& p : buf_) CUDA_CHECK(cudaMalloc(&p, n));
        CUDA_CHECK(cudaMalloc(&vel2_, n));
        clear();
    }

    ~DeviceGrid() {
        for (auto& p : buf_) { if (p) cudaFree(p); }
        if (vel2_) cudaFree(vel2_);
    }

    // Non-copyable
    DeviceGrid(const DeviceGrid&)            = delete;
    DeviceGrid& operator=(const DeviceGrid&) = delete;

    // Movable
    DeviceGrid(DeviceGrid&& o) noexcept
        : dims_(o.dims_), idx_(o.idx_), vel2_(o.vel2_) {
        for (int i = 0; i < 3; ++i) { buf_[i] = o.buf_[i]; o.buf_[i] = nullptr; }
        o.vel2_ = nullptr;
    }

    const GridDims& dims() const noexcept { return dims_; }

    real_t* prev()  noexcept { return buf_[idx_]; }
    real_t* cur()   noexcept { return buf_[(idx_ + 1) % 3]; }
    real_t* next()  noexcept { return buf_[(idx_ + 2) % 3]; }
    real_t* vel2()  noexcept { return vel2_; }

    const real_t* prev() const noexcept { return buf_[idx_]; }
    const real_t* cur()  const noexcept { return buf_[(idx_ + 1) % 3]; }

    /// Advance the ring: next becomes cur, cur becomes prev.
    void rotate() noexcept { idx_ = (idx_ + 1) % 3; }

    /// Zero all time-level buffers.
    void clear() {
        const std::size_t n = dims_.size() * sizeof(real_t);
        for (auto& p : buf_) CUDA_CHECK(cudaMemset(p, 0, n));
    }

    /// Upload from a HostGrid into the current time-level buffer.
    void upload_cur(const HostGrid& h, cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(cur(), h.ptr(), h.bytes(),
                                   cudaMemcpyHostToDevice, s));
    }
    void upload_prev(const HostGrid& h, cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(prev(), h.ptr(), h.bytes(),
                                   cudaMemcpyHostToDevice, s));
    }
    void upload_vel2(const HostGrid& h, cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(vel2_, h.ptr(), h.bytes(),
                                   cudaMemcpyHostToDevice, s));
    }

    /// Download the current time-level buffer into a HostGrid.
    void download_cur(HostGrid& h, cudaStream_t s = nullptr) const {
        CUDA_CHECK(cudaMemcpyAsync(h.ptr(), cur(), h.bytes(),
                                   cudaMemcpyDeviceToHost, s));
    }

private:
    GridDims dims_;
    int      idx_  = 0;
    real_t*  buf_[3] = {};
    real_t*  vel2_   = nullptr;
};

#else  // CPU fallback: DeviceGrid wraps HostGrid triple-buffer

class DeviceGrid {
public:
    explicit DeviceGrid(const GridDims& g)
        : dims_(g), bufs_{HostGrid(g), HostGrid(g), HostGrid(g)}, vel2_(g) {}

    const GridDims& dims() const noexcept { return dims_; }

    real_t* prev()  noexcept { return bufs_[idx_].ptr(); }
    real_t* cur()   noexcept { return bufs_[(idx_+1)%3].ptr(); }
    real_t* next()  noexcept { return bufs_[(idx_+2)%3].ptr(); }
    real_t* vel2()  noexcept { return vel2_.ptr(); }

    void rotate() noexcept { idx_ = (idx_ + 1) % 3; }
    void clear() {
        for (auto& b : bufs_) std::fill(b.data.begin(), b.data.end(), real_t{0});
    }
    void upload_cur(const HostGrid& h,  void* = nullptr) { bufs_[(idx_+1)%3] = h; }
    void upload_prev(const HostGrid& h, void* = nullptr) { bufs_[idx_] = h; }
    void upload_vel2(const HostGrid& h, void* = nullptr) { vel2_ = h; }
    void download_cur(HostGrid& h,      void* = nullptr) const { h = bufs_[(idx_+1)%3]; }

private:
    GridDims  dims_;
    int       idx_ = 0;
    HostGrid  bufs_[3];
    HostGrid  vel2_;
};

#endif  // STENCIL_CPU_FALLBACK


// ── Non-inline utilities (implemented in src/utils/grid.cpp) ─────────────────
/// Throw std::invalid_argument if grid dimensions or CFL are invalid.
void validate_grid(const GridDims& g, real_t v_max = real_t{0});

/// Print a human-readable grid summary to stdout.
void print_grid_info(const GridDims& g, const char* label = "");

/// RMS difference between two HostGrids (correctness verification).
double l2_error(const HostGrid& a, const HostGrid& b);

/// L∞ difference between two HostGrids.
double max_abs_error(const HostGrid& a, const HostGrid& b);

}  // namespace stencil
