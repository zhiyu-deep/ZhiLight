#pragma once

#include <bmengine/core/core.h>

namespace nn {
namespace awq {

using namespace bmengine;

core::Tensor awq_dequantize(
    const core::Context& ctx,
    core::Tensor _kernel,
    core::Tensor _scaling_factors,
    core::Tensor _zeros,
    int split_k_iters,
    int thx,
    int thy);

core::Tensor awq_gemm(
    const core::Context& ctx,
    core::Tensor _in_feats,
    core::Tensor _kernel,
    core::Tensor _scaling_factors,
    core::Tensor _zeros,
    size_t split_k_iters);
}

}