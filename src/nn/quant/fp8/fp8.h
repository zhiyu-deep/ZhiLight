#pragma once

#include <bmengine/core/core.h>

namespace nn::fp8 {

using namespace bmengine;

core::Tensor cvt_half_to_fp8(const core::Context& ctx, const core::Tensor& input, float scale, int round_up=32);

core::Tensor cvt_fp8_to_half(const core::Context& ctx, const core::Tensor& input, float scale);

core::Tensor calc_scale(
    const core::Context& ctx,
    const core::Tensor& input,
    float MAX_E4M3=448
);

core::Tensor dynamic_scaled_quant(
    const core::Context& ctx,
    const core::Tensor& input,
    float MAX_E4M3=448
);

void mma_fp8(cudaStream_t stream, int8_t *A, int8_t *B, half *C, float alpha, size_t M, size_t N, size_t K);
}
