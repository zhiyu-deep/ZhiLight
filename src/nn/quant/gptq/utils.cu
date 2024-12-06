#include <bmengine/core/core.h>
#include <bmengine/functions/gemm.h>
#include <bmengine/logger/std_log_op.hpp>
#include <assert.h>

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "compat.cuh"
#include "matrix_view.cuh"
#include "qdq_4.cuh"

#define WARP_SIZE 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

using namespace bmengine;

namespace nn {
namespace gptq {

// （n / 32), (32)
__global__ void KERNEL_un_shuffle(
    uint32_t* q_ptr,
    const int DIM0,
    const int size_n
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= size_n) return;
    q_ptr += n;
    int de_sfl[8] = {0, 4, 1, 5, 2, 6, 3, 7}; // de-interleave index
    for (int k = 0; k < DIM0; k++) {
        uint32_t q_in = *q_ptr;
        uint32_t q_out = 0;
        for (int s = 0; s < 8; s++) {
            uint32_t bit4 = (q_in >> de_sfl[s] * 4) & 0x0F;
            q_out |= bit4 << s * 4;
        }
        *q_ptr = q_out;

        q_ptr += size_n;
    }
}

void un_shuffle(
    const core::Context& ctx,
    core::Tensor& input
) {
    size_t n_x = DIVIDE(input.size(1), 32);
    auto stream = ctx.current_stream()->ptr;
    KERNEL_un_shuffle<<<n_x, 32, 0, stream>>>(
        input.data<uint32_t>(),
        input.size(0),
        input.size(1)
    );
}

// （numel / 32), (32)
__global__ void KERNEL_inc_zero(
    uint16_t* q_ptr,
    const int NUMEL
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= NUMEL) return;
//     q_ptr[n] += 0x1111;
    // deal with overflow: 0xF + 1 = 0
    uint16_t q = q_ptr[n];
    if ((q & 0xF) == 0xF) q -= 0xF; else q += 0x1;
    if ((q & 0xF0) == 0xF0) q -= 0xF0; else q += 0x10;
    if ((q & 0xF00) == 0xF00) q -= 0xF00; else q += 0x100;
    if ((q & 0xF000) == 0xF000) q -= 0xF000; else q += 0x1000;
    q_ptr[n] = q;
}

// increase quant weight by 1 for every 4-bit
void increase_zero(
    const core::Context& ctx,
    core::Tensor& input
) {
    size_t n_x = DIVIDE(input.numel(), 32);
    auto stream = ctx.current_stream()->ptr;
    KERNEL_inc_zero<<<n_x * 2, 32, 0, stream>>>(
        input.data<uint16_t>(),
        input.numel() * 2
    );
}

// （numel / 1024), (1024)
__global__ void KERNEL_subtract8(
    uint16_t* q_ptr,
    const size_t NUMEL
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= NUMEL) return;
    // deal with overflow
    uint16_t q = q_ptr[n];
    if ((q & 0xF) >= 0x8) q -= 0x8; else q += 0x8;
    if ((q & 0xF0) >= 0x80) q -= 0x80; else q += 0x80;
    if ((q & 0xF00) >= 0x800) q -= 0x800; else q += 0x800;
    if ((q & 0xF000) >= 0x8000) q -= 0x8000; else q += 0x8000;
    q_ptr[n] = q;
}

// subtract 8 for every 4-bit
void subtract8(
    const core::Context& ctx,
    core::Tensor& input
) {
    BM_ASSERT_EQ(input.dtype(), core::DataType::kInt32, "");
    size_t n_x = DIVIDE(input.numel(), 1024);
    auto stream = ctx.current_stream()->ptr;
    KERNEL_subtract8<<<n_x * 2, 1024, 0, stream>>>(
        input.data<uint16_t>(),
        input.numel() * 2
    );
}

// （K / 8, N / 32 / 8), (32)
__global__ void KERNEL_shuffle_awq(
    uint32_t* q_in, // (dim_in, dim_out / 8) =>
    uint32_t* q_out, // (dim_in / 8, dim_out)
    const int size_n,
    bool use_exllama
) {
    int k_8 = blockIdx.x; // k / 8
    int n_8 = blockIdx.y * blockDim.x + threadIdx.x; // n / 8
    const int N_8 = size_n / 8;
    if (n_8 >= N_8) return;

    int de_sfl[8] = {0, 4, 1, 5, 2, 6, 3, 7}; // de-interleave index
    for (int i = 0; i < 8; ++i) de_sfl[i] *= 4;

    uint32_t dq[8][8];
    for (int r = 0; r < 8; ++r) {
        int k = k_8 * 8 + r;
        uint32_t q = q_in[k * N_8 + n_8];
        for (int s = 0; s < 8; s++) {
            dq[r][s] = (q >> de_sfl[s]) & 0x0F;
        }
    }

    int sfl[8] = {0, 2, 4, 6, 1, 3, 5, 7}; // interleave index
    for (int c = 0; c < 8; ++c) {
        uint32_t q = 0;
        for (int s = 0; s < 8; s++) {
            if (use_exllama)
                q += dq[sfl[s]][c] << (s * 4);
            else
                q += dq[s][c] << (s * 4);
        }
        int n = n_8 * 8 + c;
        q_out[k_8 * size_n + n] = q;
    }
}

core::Tensor shuffle_awq(
    const core::Context& ctx,
    core::Tensor& input,
    bool use_exllama
) {
    core::Tensor output = ctx.tensor({input.size(0) / 8, input.size(1) * 8}, input.dtype());
    size_t n_x = DIVIDE(input.size(1), 32);
    dim3 gridDim(output.size(0), n_x);
    auto stream = ctx.current_stream()->ptr;
    KERNEL_shuffle_awq<<<gridDim, 32, 0, stream>>>(
        input.data<uint32_t>(),
        output.data<uint32_t>(),
        output.size(1),
        use_exllama
    );
    return output;
}

// （n / 1024), (1024)
__global__ void KERNEL_q4_to_q8(
    const uint32_t* q4,
    uint2* q8,
    const size_t num_element
) {
    size_t n = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= num_element) return;

    uint32_t q_in = q4[n];
    uint2 out = {0, 0};
    for (int s = 0; s < 4; s++) {
        uint32_t bit4 = (q_in >> s * 4) & 0x0F;
        out.x |= bit4 << s * 8;
    }
    for (int s = 0; s < 4; s++) {
        uint32_t bit4 = (q_in >> (s + 4) * 4) & 0x0F;
        out.y |= bit4 << s * 8;
    }
    q8[n] = out;
}

core::Tensor q4_to_q8(
    const core::Context& ctx,
    const core::Tensor& input // int32
) {
    size_t n_x = DIVIDE(input.numel(), 1024);
    auto stream = ctx.current_stream()->ptr;
    auto shape = input.shape();
    shape[1] *= 8;
    auto out = ctx.tensor(shape, bmengine::core::DataType::kInt8);
    KERNEL_q4_to_q8<<<n_x * 2, 1024, 0, stream>>>(
        input.data<uint32_t>(),
        out.mutable_data<uint2>(),
        input.numel()
    );
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

// （n / 1024), (1024)
__global__ void KERNEL_q8_to_q4(
    const uint32_t* q8,
    uint16_t* q4,
    const size_t num_element
) {
    size_t n = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= num_element) return;

    uint32_t q_in = q8[n];
    uint16_t out = 0;
    for (int s = 0; s < 4; s++) {
        uint32_t bit4 = (q_in >> s * 8) & 0x0F;
        out |= bit4 << s * 4;
    }
    q4[n] = out;
}

core::Tensor q8_to_q4(
    const core::Context& ctx,
    const core::Tensor& input // int8
) {
    size_t n_x = DIVIDE(input.numel(), 1024);
    auto stream = ctx.current_stream()->ptr;
    auto shape = input.shape();
    shape[1] /= 2;
    auto out = ctx.tensor(shape, bmengine::core::DataType::kInt8);
    KERNEL_q8_to_q4<<<n_x / sizeof(int), 1024, 0, stream>>>(
        input.data<uint32_t>(),
        out.mutable_data<uint16_t>(),
        input.numel() / sizeof(int)
    );
    BM_CUDART_ASSERT(cudaGetLastError());
    return out; // int8
}

// （n / 32), (32)
__global__ void KERNEL_int32_to_int16(
    const uint32_t* src,
    uint16_t* dst,
    const int num_element
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_element) {
        int i = src[n];
        assert(i < 65536);
        dst[n] = uint16_t(i);
    }
}

core::Tensor int32_to_int16(
    const core::Context& ctx,
    const core::Tensor& input
) {
    BM_ASSERT_EQ(input.dtype(), bmengine::core::DataType::kInt32, "");
    size_t n_x = DIVIDE(input.numel(), 32);
    auto stream = ctx.current_stream()->ptr;
    auto shape = input.shape();
    shape[shape.size() - 1] /= 2;
    auto out = ctx.tensor(shape, input.dtype());
    KERNEL_int32_to_int16<<<n_x * 2, 32, 0, stream>>>(
        input.data<uint32_t>(),
        out.mutable_data<uint16_t>(),
        input.numel()
    );
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

// （n / 32), (32)
template<typename OUT_T>
__global__ void KERNEL_reverse_perm(
    const uint32_t* src,
    OUT_T* dst,
    const int num_element
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_element) {
        int i = src[n];
        assert(i < 65536);
        assert(n < 65536);
        dst[i] = OUT_T(n);
    }
}

core::Tensor reverse_perm(
    const core::Context& ctx,
    const core::Tensor& input
) {
    BM_ASSERT(input.numel(), "g_perm is empty");
    BM_ASSERT_EQ(input.dtype(), bmengine::core::DataType::kInt32, "");
    size_t n_x = DIVIDE(input.numel(), 32);
    auto stream = ctx.current_stream()->ptr;
    auto shape = input.shape();
    shape[shape.size() - 1] /= 2; // sizeof(int) / sizeof(int16)
    auto out = ctx.tensor(shape, input.dtype());
    KERNEL_reverse_perm<uint16_t><<<n_x * 2, 32, 0, stream>>>(
        input.data<uint32_t>(),
        out.mutable_data<uint16_t>(),
        input.numel()
    );
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

// (M / BLOCK_M), (1024)
template<int BLOCK_M=1>
__global__ void KERNEL_permute_input(
    const half* __restrict__ g_input,    // (M, K)
    const uint16_t* __restrict__ q_perm, // (K)
    half* __restrict__ g_out,            // (M, K)
    const int M,
    const int K_8
) {
    const int num_thread = blockDim.x;
    const int K = K_8 * 8;

    extern __shared__ half smem_permute_input[];

    // load q_perm to shared memory
    int16_t* smem_perm = (int16_t*)smem_permute_input;
    for (int k = threadIdx.x; k < K; k += num_thread) {
        smem_perm[k] = q_perm[k];
    }
//    for (int k_8 = threadIdx.x; k_8 < K_8; k_8 += num_thread) {
//        reinterpret_cast<uint4 *>(smem_perm)[k_8] = reinterpret_cast<const uint4 *>(q_perm)[k_8];
//    }
    __syncthreads();

    // Point A to shared memory
    half* smem_out = smem_permute_input + K;

    const int start_m = blockIdx.x * BLOCK_M;
    const int end_m = min(M, start_m + BLOCK_M);
    for (int m = start_m; m < end_m; ++m) {
        // slice matrix
        const half *__restrict__ in = g_input + m * K;
        half *__restrict__ out = g_out + m * K;

        // do permute in shared memory
        for (int k = threadIdx.x; k < K; k += num_thread) {
            smem_out[k] = in[smem_perm[k]];
        }
        __syncthreads();

        // write out
        for (int k = threadIdx.x; k < K; k += num_thread) {
            out[k] = smem_out[k];
        }
        __syncthreads();
//        for (int k_8 = threadIdx.x; k_8 < K_8; k_8 += num_thread) {
//            reinterpret_cast<int4 *>(out)[k_8] = reinterpret_cast<int4 *>(smem_out)[k_8];
//        }
    }
}

core::Tensor permute_input(
    const core::Context& ctx,
    const core::Tensor& input,
    const core::Tensor& q_perm
) {
    const size_t K = input.size(-1);
    const size_t M = input.numel() / K;

    BM_ASSERT(input.dtype() == core::DataType::kHalf, "A must be half");
    BM_ASSERT_EQ(K, q_perm.size(0) * 2, "q_perm is not int16");

    core::Tensor out = ctx.tensor(input.shape(), input.dtype());

    auto stream = ctx.current_stream()->ptr;
    int dyn_mem_size = K * 2 * sizeof(half);
    BM_CUDART_ASSERT(cudaFuncSetAttribute(
        KERNEL_permute_input<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_mem_size));
    KERNEL_permute_input<1><<<M, 512, dyn_mem_size, stream>>>(
        input.data<half>(),
        q_perm.data<uint16_t>(),
        out.mutable_data<half>(),
        M,
        K / 8);
    BM_CUDART_ASSERT(cudaGetLastError());

    return out;
}

// (N), (32)
__global__ void KERNEL_w4_to_int8(
    const uint32_t* __restrict__ G_qweight, // (N, K / 8)
    const uint8_t* __restrict__ G_qzeros,   // (N, K / group_size)
    const uint8_t* __restrict__ G_q_scales, // (N, K / group_size)
    int8_t* __restrict__ i8_out,            // (N, K)
    const int K_8,
    const int GROUP_SIZE_8
) {
    const int n = blockIdx.x;
    // slice matrix
    const uint32_t* __restrict__ q_weight = G_qweight + n * K_8;
    const uint8_t* __restrict__ q_zeros = G_qzeros + n * (K_8 / GROUP_SIZE_8);
    const uint8_t* __restrict__ q_scales = G_q_scales + n * (K_8 / GROUP_SIZE_8);
    int2* __restrict__ out = (int2*)(i8_out + n * K_8 * 8);

    for (int k_8 = threadIdx.x; k_8 < K_8; k_8 += WARP_SIZE) {
        // load and decode weight
        // 77775555 33331111  66664444 22220000
        uint32_t x = q_weight[k_8];
        // uint32_t x = 0x76543210; // fake x
        // shuffle_4bit_8(&x, 0);
        // printf("%#08X\n", x);
        uint32_t scale0 = G_q_scales ? q_scales[k_8 / GROUP_SIZE_8] : 1;
        uint32_t zero0 = q_zeros[k_8 / GROUP_SIZE_8];
        uint32_t zero;
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(zero) : "r"(zero0), "n"(0), "n"(0));
        // asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(scale) : "r"(scale0), "n"(0), "n"(0));
        uint32_t s_zero = zero * scale0;

        uint32_t x1 = (x >> 4) & 0x0F0F0F0F; // 7,3,6,2
        uint32_t x0 = x & 0x0F0F0F0F; // 5,1,4,0
        uint32_t ret[2];
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(ret[0]) : "r"(x0), "r"(x1), "n"(0x6420));
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(ret[1]) : "r"(x0), "r"(x1), "n"(0x7531));
//        printf("ret[0] %#08X\n", ret[0]);
//        printf("ret[1] %#08X\n", ret[1]);

        uint32_t sub[2];
        sub[0] = __vsub4(ret[0] * scale0, s_zero);
        sub[1] = __vsub4(ret[1] * scale0, s_zero);
//        sub[0] = __vsub4(ret[0], zero);
//        sub[1] = __vsub4(ret[1], zero);
//        printf("sub0 %#08X\n", sub0);
//        printf("sub1 %#08X\n", sub1);

        out[k_8] = *reinterpret_cast<int2*>(&sub[0]);
    }
}

core::Tensor w4_to_int8(
    const core::Context& ctx,
    const core::Tensor& permuted_weight, // (N, K / 8)
    const core::Tensor& qzeros,          // (N, K / group_size)
    const core::Tensor* q_scale
) {
    const size_t K = permuted_weight.size(-1) * 8;
    const size_t N = permuted_weight.size(-2);
    const size_t GROUP_SIZE = K / qzeros.size(-1);

    BM_ASSERT(qzeros.dtype() == core::DataType::kInt8, "qzeros must be int8");

    auto stream = ctx.current_stream()->ptr;
    core::Tensor int8_weight = ctx.tensor({N, K}, core::DataType::kInt8);
    KERNEL_w4_to_int8<<<N, 32, 0, stream>>>(
        permuted_weight.data<uint32_t>(),
        qzeros.data<uint8_t>(),
        q_scale ? q_scale->data<uint8_t>() : nullptr,
        int8_weight.mutable_data<int8_t>(),
        K / 8,
        GROUP_SIZE / 8);
    BM_CUDART_ASSERT(cudaGetLastError());

    return int8_weight;
}

}  // namespace gptq
}  // namespace nn
