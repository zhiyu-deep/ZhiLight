#include "nn/nn.h"
#include <bmengine/core/core.h>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "utils/exception.h"

namespace nn {

template<typename T>
static __global__ void BM_KERNEL(multiply)(size_t n, const T* a, float b, T* c) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * T(b);
    }
}

void multiply(const core::Context& ctx, const core::Tensor& a, float b, core::Tensor* c) {
    size_t n = a.numel();
    int threads = round_up_thread(n);
    int blocks = round_up(n, threads) / threads;
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(a.dtype(), {
        BM_KERNEL(multiply)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n, a.data<scalar_t>(), b, c->mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

} // namespace nn
