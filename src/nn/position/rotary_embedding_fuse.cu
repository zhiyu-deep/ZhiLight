#include "rotary_embedding.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/utils.cuh>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/logger/std_log_op.hpp>
#include "utils/env.h"
#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector_types.h>

namespace nn {

using bmengine::core::DataType;
using bmengine::core::Tensor;

// gridDim (seq_len, all_num_heads),  blockDim (dim_head)
template<typename T_IN, typename T>
static __global__ void KERNEL_rotary_embedding_qk(
    const int32_t *__restrict__ pos, // (seq_len)
    const T_IN *__restrict__ in,     // (seq_len, all_num_heads * dim_head)
    T *__restrict__ q,               // (seq_len, num_heads * dim_head)
    T *__restrict__ k,               // (seq_len, num_kv_heads * dim_head)
    T *__restrict__ v,               // (seq_len, num_kv_heads * dim_head)
    int all_num_heads,
    int num_heads,
    int num_kv_heads,
    int dim_head,
    float rope_theta) {
    const int head = blockIdx.y;
    int offset = ((blockIdx.x * all_num_heads) + head) * dim_head;
    int target_pos = pos[blockIdx.x];
    int col = threadIdx.x;

    // v
    if (head >= num_heads + num_kv_heads) {
        int row = head - num_heads - num_kv_heads;
        int offset_v = ((blockIdx.x * num_kv_heads) + row) * dim_head;
        v[offset_v + col] = in[offset + col];
        return;
    }

    int half_dim = dim_head / 2;
    float t;
    if (col < half_dim) {
        float freq = target_pos * powf(rope_theta, -float(col * 2) / dim_head);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        t = float(in[offset + col]) * cos_freq - float(in[offset + col + half_dim]) * sin_freq;
    } else {
        float freq = target_pos * powf(rope_theta, -float((col - half_dim) * 2) / dim_head);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        t = float(in[offset + col]) * cos_freq + float(in[offset + col - half_dim]) * sin_freq;
    }

    if (head >= num_heads) {
        // k
        int row = head - num_heads;
        int offset_k = ((blockIdx.x * num_kv_heads) + row) * dim_head;
        k[offset_k + col] = t;
    } else {
        // q
        int offset_q = ((blockIdx.x * num_heads) + head) * dim_head;
        q[offset_q + col] = t;
    }
}

// fuse kernel for attention
void rotary_embedding_qk(
    const core::Context& ctx,
    const core::Tensor& pos, // (seq_len)
    const core::Tensor& in,  // (seq_len, all_num_heads * dim_head)
    core::Tensor& out_q,     // (seq_len, num_heads * dim_head)
    core::Tensor& out_k,     // (seq_len, num_kv_heads * dim_head)
    core::Tensor& out_v,     // (seq_len, num_kv_heads * dim_head)
    size_t num_heads,
    size_t num_kv_heads,
    size_t dim_head,
    float rope_theta,
    core::DataType dtype) {
    size_t seq_len = pos.size(0);
    BM_ASSERT_EQ(pos.size(0), in.size(0), "batch mismatch");

    out_q = ctx.tensor({seq_len, num_heads * dim_head}, dtype);
    out_k = ctx.tensor({seq_len, num_kv_heads * dim_head}, dtype);
    out_v = ctx.tensor({seq_len, num_kv_heads * dim_head}, dtype);

    dim3 gridDim(seq_len, num_heads + 2 * num_kv_heads);
    auto stream = ctx.current_stream()->ptr;

    if (in.dtype() == core::DataType::kFloat) {
        BM_DTYPE_DISPATCH_FLOAT(dtype, {
            KERNEL_rotary_embedding_qk<<<gridDim, dim_head, 0, stream>>>(
                pos.data<int32_t>(),
                in.data<float>(),
                out_q.mutable_data<scalar_t>(),
                out_k.mutable_data<scalar_t>(),
                out_v.mutable_data<scalar_t>(),
                num_heads + 2 * num_kv_heads,
                num_heads,
                num_kv_heads,
                dim_head,
                rope_theta);
        });
    } else {
        BM_ASSERT_EQ(in.dtype(), dtype, "");
        BM_DTYPE_DISPATCH_FLOAT(dtype, {
            KERNEL_rotary_embedding_qk<scalar_t><<<gridDim, dim_head, 0, stream>>>(
                pos.data<int32_t>(),
                in.data<scalar_t>(),
                out_q.mutable_data<scalar_t>(),
                out_k.mutable_data<scalar_t>(),
                out_v.mutable_data<scalar_t>(),
                num_heads + 2 * num_kv_heads,
                num_heads,
                num_kv_heads,
                dim_head,
                rope_theta);
        });
    }
    BM_CUDART_ASSERT(cudaGetLastError());
}

} // namespace nn
