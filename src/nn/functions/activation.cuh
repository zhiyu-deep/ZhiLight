#pragma once
#include <bmengine/core/core.h>

namespace nn {

using namespace bmengine;

inline  __device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

inline  __device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

void gelu_inplace(const core::Tensor& inp, cudaStream_t stream);

void silu_inplace(const core::Tensor& inp, cudaStream_t stream);

}  // namespace nn