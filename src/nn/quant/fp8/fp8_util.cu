// Author: Gaojunmin@zhihu.com

#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include <bmengine/logger/std_log_op.hpp>
#include <assert.h>

#include <cstdint>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <assert.h>
#include "../awq/awq.h"
#include "fp8.h"

__device__ __forceinline__ uint16_t half2_to_e4m3(const uint32_t a)
{
    uint16_t val;
#if __CUDA_ARCH__ >= 890
    asm volatile("{ cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}\n" : "=h"(val) : "r"(a));
#else
    assert(false);
#endif
    return val;
}

__device__ __forceinline__ uint32_t e4m3_to_half2(const uint16_t a)
{
    uint32_t val;
#if __CUDA_ARCH__ >= 890
    asm volatile("{ cvt.rn.f16x2.e4m3x2 %0, %1;}\n" : "=r"(val) : "h"(a));
#else
    assert(false);
#endif
    return val;
}

// (N/1024/2), 1024
__global__ void KERNEL_cvt_half_fp8(const half2 *__restrict__ in, // (N/2)
                                    uint16_t *__restrict__ out,   // (N/2)
                                    uint32_t N,
                                    float scale) {
    uint32_t n_2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_2 < N / 2) {
        half2 scale_h2 = __float2half2_rn(scale);
        half2 h2 = in[n_2];
        half2 h2s = __hmul2(h2, scale_h2);
        out[n_2] = half2_to_e4m3(*(uint32_t*) &h2s);
    }
}

// (N/1024/2), 1024
template<bool IS_BF16=false>
__global__ void T_KERNEL_cvt_half_fp8(const half2 *__restrict__ in, // (N/2)
                                      uint16_t *__restrict__ out,   // (N/2)
                                      uint32_t N,
                                      float* scale_ptr) {
    uint32_t n_2 = blockIdx.x * blockDim.x + threadIdx.x;
    nv_bfloat16 bf2[2];
    if (n_2 < N / 2) {
        float scale = 1.f / *scale_ptr;
        half2 h2s;
        if constexpr(IS_BF16) {
            *(half2*)&bf2 = in[n_2];
            h2s = __floats2half2_rn(scale * (float)bf2[0], scale * (float)bf2[1]);
        } else {
            half2 scale_h2 = __float2half2_rn(scale);
            half2 h2 = in[n_2];
            h2s = __hmul2(h2, scale_h2);
        }
        out[n_2] = half2_to_e4m3(*(uint32_t*) &h2s);
    }
}

// (N/1024/2), 1024
__global__ void KERNEL_cvt_fp8_half(const uint16_t *__restrict__ in, // (N/2)
                                    half2 *__restrict__ out,         // (N/2)
                                    uint32_t N,
                                    float scale) {
    uint32_t n_2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_2 < N / 2) {
        half2 scale_h2 = __float2half2_rn(scale);
        uint16_t fp8x2 = in[n_2];
        uint32_t h2 = e4m3_to_half2(fp8x2);
        out[n_2] = __hmul2(*(half2*) &h2, scale_h2);
    }
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0)
          ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
          : __uint_as_float(
            atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

// Compute the absolute maximum m of the input tensor and store
// m / float8_e4m3::max() in *scale. Each thread block performs a
// reduction tree and the memory in scale is atomically updated.
// So to get the right answer, *scale needs to be initialized to
// a value <= 0.0 and we need to wait for all thread blocks to
// finish before consuming *scale.
template <typename scalar_t>
__global__ void segmented_max_reduction(float* __restrict__ scale,
                                        const scalar_t* __restrict__ input,
                                        int64_t num_elems,
                                        float MAX_E4M3=448) {
    __shared__ float cache[1024];
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // First store maximum for all values processes by
    // the current thread in cache[threadIdx.x]
    scalar_t tmp = 0.0;
    while (i < num_elems) {
        float x = static_cast<float>(input[i]);
        tmp = max(tmp, fabs(x));
        i += blockDim.x * gridDim.x;
    }
    cache[threadIdx.x] = tmp;

    __syncthreads();

    // Now perform parallel reduction within the thread block
    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x]) {
            cache[threadIdx.x] = cache[threadIdx.x + ib];
        }
        __syncthreads();
        ib /= 2;
    }
    // Finally, since cache[0] contains the maximum for this thread block,
    // atomically write the max to the target location
    if (threadIdx.x == 0) {
        atomicMaxFloat(scale, cache[0] / MAX_E4M3);
    }
}

namespace nn::fp8 {

using namespace bmengine;

core::Tensor cvt_half_to_fp8(const core::Context& ctx, const core::Tensor& input, float scale, int round_up_m) {
    BM_ASSERT_EQ(input.dtype(), core::DataType::kHalf, "");
    core::Tensor fp8_out = ctx.tensor(input.shape(), core::DataType::kInt8, "", round_up_m * input.size(-1));
    auto stream = ctx.current_stream()->ptr;
    KERNEL_cvt_half_fp8<<<round_up(input.numel() / 2, 1024) / 1024, 1024, 0, stream>>>(
        input.data<half2>(),
        fp8_out.mutable_data<uint16_t>(),
        input.numel(),
        scale
    );
    BM_CUDART_ASSERT(cudaGetLastError());
    return fp8_out;
}

core::Tensor cvt_fp8_to_half(const core::Context& ctx, const core::Tensor& input, float scale) {
    core::Tensor half_out = ctx.tensor(input.shape(), core::DataType::kHalf);
    auto stream = ctx.current_stream()->ptr;
    KERNEL_cvt_fp8_half<<<round_up(input.numel() / 2, 1024) / 1024, 1024, 0, stream>>>(
        input.data<uint16_t>(),
        half_out.mutable_data<half2>(),
        input.numel(),
        scale
    );
    BM_CUDART_ASSERT(cudaGetLastError());
    return half_out;
}

core::Tensor calc_scale(
    const core::Context& ctx,
    const core::Tensor& input,
    float MAX_E4M3
) {
    const cudaStream_t stream = ctx.current_stream()->ptr;
    core::Tensor scale = ctx.tensor({1}, core::DataType::kFloat);  // [1]
    BM_CUDART_ASSERT(cudaMemsetAsync(scale.data(), 0, sizeof(float), stream));

    int64_t num_tokens = input.numel() / input.size(-1);
    int64_t num_elems = input.numel();
    dim3 grid(num_tokens);
    dim3 block(1024);
    BM_DTYPE_DISPATCH_HALF(input.dtype(), {
        segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
            scale.data<float>(), input.data<scalar_t>(), num_elems, MAX_E4M3);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return scale;
}

core::Tensor dynamic_scaled_quant(
    const core::Context& ctx,
    const core::Tensor& input,
    float MAX_E4M3
) {
    const cudaStream_t stream = ctx.current_stream()->ptr;
    size_t round_up_size = std::max(32 * input.size(-1), 1024UL);
    core::Tensor out = ctx.tensor(input.shape(), core::DataType::kInt8, "", round_up_size);
    core::Tensor scale = calc_scale(ctx, input, MAX_E4M3);

    dim3 block(1024);
    dim3 grid(round_up(input.numel() / 2, 1024) / 1024);
    if (input.dtype() == core::DataType::kHalf) {
        T_KERNEL_cvt_half_fp8<false><<<grid, block, 0, stream>>>(
            input.data<half2>(),
            out.mutable_data<uint16_t>(),
            input.numel(),
            scale.data<float>()
        );
    } else {
        T_KERNEL_cvt_half_fp8<true><<<grid, block, 0, stream>>>(
            input.data<half2>(),
            out.mutable_data<uint16_t>(),
            input.numel(),
            scale.data<float>()
        );
    }
    BM_CUDART_ASSERT(cudaGetLastError());

    out.quant_scale = std::make_shared<core::Tensor>();
    *out.quant_scale = scale;
    return out;
}

} // namespace nn::fp8