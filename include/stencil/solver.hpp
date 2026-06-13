#pragma once

#include "stencil/config.hpp"
#include "stencil/grid.hpp"
#include "stencil/timer.hpp"
#include "stencil/types.hpp"

#include <memory>
#include <string>

namespace stencil {

class ISolver {
public:
    virtual ~ISolver() = default;
    virtual void            init(const Config& cfg)          = 0;
    virtual void            step()                           = 0;
    virtual PerfResult      run(int n_steps, int warmup = 0) = 0;
    virtual void            download(HostGrid& out) const    = 0;
    virtual const GridDims& dims() const                     = 0;
    virtual std::string     name() const                     = 0;
};

std::unique_ptr<ISolver> make_solver(const Config& cfg);

}  // namespace stencil
