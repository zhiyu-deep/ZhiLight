#include "nn/quant/int8/quant_kernel.h"
#include "nn/functions/activation.cuh"
#include <bmengine/core/core.h>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/logger/std_log_op.hpp>

namespace int8_op {

using namespace bmengine;
using bmengine::core::DataType;

// gridDim(M) blockDim(32)
template<typename T, typename SCALE_T=T>
static __global__ void KERNEL_quant_group_32(
    const T* __restrict__ inp,   // (M * K)
    int8_t* __restrict__ out,    // (M * K)
    SCALE_T* __restrict__ scale, // (M)
    uint32_t M
) {
    static __shared__ SCALE_T s[32];
    uint32_t m = blockIdx.x * blockDim.y + threadIdx.y;
    if (m < M) {
        size_t offset = m * 32;
        int i = threadIdx.x;
        float v = inp[offset + i];
        float abs_max = functions::warpReduceMaxB<T>(fabsf(v));

        out[offset + i] = int8_t(nearbyintf(v * 127.0f / abs_max));
        if (threadIdx.x == 0) {
//            s[threadIdx.y] = abs_max / 127.0f;
            scale[m] = abs_max / 127.0f;
        }
    }

//    __syncthreads();
//    if (threadIdx.y == 0 && m < M && threadIdx.x < blockDim.y)
//        scale[blockIdx.x * blockDim.y + threadIdx.x] = s[threadIdx.x];
}

template<typename T, typename SCALE_T=T>
static __global__ void KERNEL_quant_group_32_v2(
    const T* __restrict__ inp,   // (M * K)
    int8_t* __restrict__ out,    // (M * K)
    SCALE_T* __restrict__ scale, // (M)
    uint32_t M
) {
    __align__(16) SCALE_T s[32];
    static_assert(sizeof(SCALE_T) == 2);

    uint32_t m = blockIdx.x * 32;
    for (int k = 0; k < 32; k++, m++) {
        if (m < M) {
            size_t offset = m * 32;
            int i = threadIdx.x;
            float v = inp[offset + i];
            float abs_max = functions::warpReduceMaxB<T>(fabsf(v));

            out[offset + i] = int8_t(nearbyintf(v * 127.0f / abs_max));
            if (threadIdx.x == 0) {
//            s[threadIdx.y] = abs_max / 127.0f;
                s[k] = abs_max / 127.0f;
            }
        }
    }

    m = blockIdx.x * 32;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        *(int4*)(&scale[m + i * 8]) = *(int4*)(&s[i * 8]);
    }

//    __syncthreads();
//    if (threadIdx.y == 0 && m < M && threadIdx.x < blockDim.y)
//        scale[blockIdx.x * blockDim.y + threadIdx.x] = s[threadIdx.x];
}

std::tuple<core::Tensor, core::Tensor> quant_group_32(
    const core::Context& ctx,
    const core::Tensor& input) {
    size_t k = input.size(-1);
    size_t M = input.numel() / k;
    BM_ASSERT_EQ(k, 32, "[quant_group_32]");
//    BM_ASSERT_EQ(M % 32, 0, "[quant_group_32]");

    auto output = ctx.tensor(input.size(), DataType::kInt8, "", 32 * k);
    auto scale_shape = get_scale_shape(input);
    auto output_scale = ctx.tensor(scale_shape, input.dtype());

    cudaStream_t stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_HALF(input.dtype(), {
        KERNEL_quant_group_32<scalar_t><<<round_up(M, 32) / 32, {32, 32}, 0, stream>>>(
        input.data<scalar_t>(),
        output.mutable_data<int8_t>(),
        output_scale.mutable_data<scalar_t>(),
        M);
    });
    BM_CUDART_ASSERT(cudaGetLastError());

    return {output, output_scale};
}

// gridDim(M / 32) blockDim(32)
template<typename T, typename SCALE_T=T>
static __global__ void KERNEL_dequant_group_32(
    const int8_t* __restrict__ q, // (M * K)
    SCALE_T* __restrict__ scale,  // (M)
    T* __restrict__ out,          // (M * K)
    uint32_t M
) {
    uint32_t m = blockIdx.x * blockDim.y + threadIdx.y;
    size_t offset = m * 32;
    int i = threadIdx.x;
    if (m < M) {
        // out[offset + i] = float(q[offset + i]) * float(scale[m]);
        out[offset + i] = float(__ldcs(q + (offset + i))) * float(scale[m]);
    }
}

void dequant_group_32(
    const core::Context& ctx,
    const core::Tensor& q,
    const core::Tensor& scale,
    core::Tensor* output
) {
    size_t M = q.numel() / q.size(-1);
    BM_ASSERT_EQ(q.size(-1), 32, "dequant_group_32");
    if (output->numel() == 0)
        *output = ctx.tensor(q.size(), scale.dtype());

    cudaStream_t stream = ctx.current_stream()->ptr;
    BM_DTYPE_DISPATCH_FLOAT(scale.dtype(), {
        KERNEL_dequant_group_32<scalar_t><<<round_up(M, 32) / 32, {32, 32}, 0, stream>>>(
            q.data<int8_t>(),
            scale.data<scalar_t>(),
            output->mutable_data<scalar_t>(),
            M);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim(M) blockDim(32)
template<typename T, typename SCALE_T=T>
static __global__ void KERNEL_dequant_group(
    const int8_t* __restrict__ q, // (M * K)
    const SCALE_T* __restrict__ scale,  // (M)
    T* __restrict__ out,          // (M * K)
    uint32_t M,
    float q_zero
) {
    uint32_t m = blockIdx.x;
    size_t offset = size_t(m) * blockDim.x + threadIdx.x;
    if (q_zero != 0.f) {
        float v = reinterpret_cast<const uint8_t*>(q)[offset];
        out[offset] = (v - q_zero) * float(scale[m]);
    } else {
        out[offset] = (float(__ldcs(q + offset)) - q_zero) * float(scale[m]);
    }
}

void dequant_group(
    const core::Context& ctx,
    const core::Tensor& q,
    const core::Tensor& scale,
    core::Tensor* output,
    int q_zero
) {
    const size_t G = q.size(-1);
    const size_t M = q.numel() / G;
    BM_ASSERT_EQ(G % 32, 0, "Wrong group size");
    BM_ASSERT_LE(G, 1024, "Wrong group size");
    BM_ASSERT_EQ(M, scale.numel(), "shape mismatch");
    BM_ASSERT(output, "No output");
    if (output->numel() == 0)
        *output = ctx.tensor(q.size(), scale.dtype());
     BM_ASSERT_EQ(q.shape(), output->shape(), "shape mismatch");

    cudaStream_t stream = ctx.current_stream()->ptr;
    BM_DTYPE_DISPATCH_HALF(output->dtype(), {
        if (scale.dtype() == core::DataType::kFloat) {
            KERNEL_dequant_group<scalar_t, float><<<M, G, 0, stream>>>(
                q.data<int8_t>(),
                scale.data<float>(),
                output->mutable_data<scalar_t>(),
                M,
                q_zero);
        } else {
            KERNEL_dequant_group<scalar_t, scalar_t><<<M, G, 0, stream>>>(
                q.data<int8_t>(),
                scale.data<scalar_t>(),
                output->mutable_data<scalar_t>(),
                M,
                q_zero);
        }
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim(M) blockDim(32)
template<typename T, typename SCALE_T=T>
static __global__ void KERNEL_dequant_group_fuse_add(
    const int8_t* __restrict__ q, // (M * K)
    SCALE_T* __restrict__ scale,  // (M)
    const T* __restrict__ c,      // (M * K)
    T* __restrict__ out,          // (M * K)
    uint32_t M
) {
    uint32_t m = blockIdx.x * blockDim.y + threadIdx.y;
    size_t offset = m * 32;
    int i = threadIdx.x;
    if (m < M) {
        out[offset + i] = float(q[offset + i]) * float(scale[m]) + float(c[offset + i]);
    }
}

core::Tensor dequant_group_fuse_add(
    const core::Context& ctx,
    const core::Tensor& q,
    const core::Tensor& scale,
    const core::Tensor& c
) {
    size_t M = q.numel() / q.size(-1);
    BM_ASSERT_EQ(q.size(-1), 32, "dequant_group_32");
    BM_ASSERT_EQ(q.numel(), c.numel(), "size mismatch");
    BM_ASSERT_EQ(scale.dtype(), c.dtype(), "dtype mismatch");
    auto output = ctx.tensor(c.size(), c.dtype());

    cudaStream_t stream = ctx.current_stream()->ptr;
    BM_DTYPE_DISPATCH_FLOAT(scale.dtype(), {
        KERNEL_dequant_group_fuse_add<scalar_t><<<round_up(M, 32) / 32, {32, 32}, 0, stream>>>(
            q.data<int8_t>(),
            scale.data<scalar_t>(),
            c.data<scalar_t>(),
            output.mutable_data<scalar_t>(),
            M);
    });
    BM_CUDART_ASSERT(cudaGetLastError());

    return output;
}

// gridDim(M) blockDim(32)
template<int WS, typename T, typename SCALE_T=T>
static __global__ void KERNEL_dequant_sum_quant_g32(
    const T* __restrict__ my,                  // (M, GROUP_SIZE)
    const int8_t* __restrict__ q_others,       // (WS - 1, M, GROUP_SIZE)
    const SCALE_T* __restrict__ scale_others,  // (WS - 1, M)
    int8_t* __restrict__ out_q,                // (M, GROUP_SIZE)
    SCALE_T* __restrict__ out_scale,           // (M)
    int M
) {
    int stride = M * 32;
    size_t offset = blockIdx.x * 32;
    int i = threadIdx.x;
    size_t offset_i = offset + i;

    float sum = my[offset + i];
    // float sum = float(__ldcs(my + offset_i));
#pragma unroll
    for (int r = 0; r < WS - 1; ++r) {
        sum += float(q_others[offset_i + r * stride]) * float(scale_others[r * M + blockIdx.x]);
//        float q = float(__ldcs(q_others + offset_i + r * stride));
//        float s = float(__ldcs(scale_others + r * M + blockIdx.x));
//        sum += q * s;
    }

    float abs_max = functions::warpReduceMaxB<T>(fabsf(sum));

    out_q[offset_i] = int8_t(nearbyintf(sum * 127.0f / abs_max));

    if (threadIdx.x == 0) {
        out_scale[blockIdx.x] = abs_max / 127.0f;
    }
}


void dequant_sum_quant_g32(
    const core::Context& ctx,
    const core::Tensor& my,           // (M, GROUP_SIZE)
    const core::Tensor& q_others,     // (WS - 1, M, GROUP_SIZE)
    const core::Tensor& scale_others, // (WS - 1, M)
    core::Tensor* q_sum,              // (M, GROUP_SIZE)
    core::Tensor* scale_sum
) {
    BM_ASSERT(q_sum, "");
    BM_ASSERT(scale_sum, "");
    BM_ASSERT_EQ(q_others.ndim(), 3, "");
    size_t WS = q_others.size(0) + 1;
    size_t M = q_others.size(1);
    size_t GROUP_SIZE = q_others.size(2);
    BM_ASSERT(WS == 2 || WS == 4 || WS == 8, "");
    BM_ASSERT_EQ(WS - 1, scale_others.size(0), "");
    BM_ASSERT_EQ(M, scale_others.size(1), "");
    BM_ASSERT_EQ(M, my.size(0), "");
    BM_ASSERT_EQ(GROUP_SIZE, my.size(1), "");
    BM_ASSERT_EQ(M, q_sum->size(0), "");
    BM_ASSERT_EQ(GROUP_SIZE, q_sum->size(1), "");
    BM_ASSERT_EQ(M, scale_sum->size(0), "");

    cudaStream_t stream = ctx.current_stream()->ptr;
    BM_DTYPE_DISPATCH_HALF(my.dtype(), {
        auto kernel = KERNEL_dequant_sum_quant_g32<2, scalar_t>;
        if (WS == 2) {
            kernel = KERNEL_dequant_sum_quant_g32<2, scalar_t>;
        } else if (WS == 4) {
            kernel = KERNEL_dequant_sum_quant_g32<4, scalar_t>;
        } else if (WS == 8) {
            kernel = KERNEL_dequant_sum_quant_g32<8, scalar_t>;
        } else {
            throw std::runtime_error("Wrong World size.");
        }
        kernel<<<M, GROUP_SIZE, 0, stream>>>(
            my.data<scalar_t>(),
            q_others.data<int8_t>(),
            scale_others.data<scalar_t>(),
            q_sum->mutable_data<int8_t>(),
            scale_sum->mutable_data<scalar_t>(),
            M
        );
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}
} // namespace nn
