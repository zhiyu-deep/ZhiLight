#pragma once
#include <bmengine/core/core.h>

namespace nn {
using namespace bmengine;
void gemm_grouped(
    const core::Context& ctx,
    core::DataType dtype,
    const std::vector<core::Tensor>& inputs,   // B: (n, k)
    const std::vector<core::Tensor>& weights,  // A: (m, k)
    std::vector<core::Tensor>& results         // C: (n, m) !
);
}
