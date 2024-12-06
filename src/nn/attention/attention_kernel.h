#pragma once
#include <bmengine/core/core.h>
#include <bmengine/functions/transpose.h>
#include "kvcache/ragged_buffer_kernel.h"

namespace nn {
using namespace bmengine;
using bmengine::functions::transpose_2_1;

void attn_softmax(
    const core::Context& ctx,
    float scale,
    const core::Tensor& attn_score, // (batch, num_heads, len_q, len_buf)
    const core::Tensor& mask,       // (batch, len_q, len_buf)
    const core::Tensor&
        position_bias // if relative (batch, num_head, len_q, len_buf) else if core::Tensor()
);

void mul_qk_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& batch_q,
    const core::Tensor& buf_lens,
    const core::Tensor& key_buf_addrs,
    core::Tensor& total_score
);

void mul_qk_softmax_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& batch_q,        // (batch, num_kv_heads, n_rep * len_q, dim_head）
    const core::Tensor& buf_lens,       // (batch)
    const core::Tensor& key_buf_addrs,  // (batch) => (num_kv_heads, len_buf, dim_head)
    const core::Tensor& mask,           // (batch) => (len_q, len_buf)
    const core::Tensor& position_bias,  // (batch) => (num_kv_heads, len_q, len_buf)
    float scale,
    int max_len_buf,
    core::Tensor& total_score           // (batch) => (num_kv_heads, len_q, len_buf)
);

void attention_qkv_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& batch_q,        // (batch, len_q, num_heads, dim_head）
    const core::Tensor& buf_lens,       // (batch)
    const core::Tensor& key_buf_addrs,  // (batch) => (num_heads, len_buf, dim_head)
    const core::Tensor& val_buf_addrs,  // (batch) => (num_heads, len_buf, dim_head)
    const core::Tensor& mask,           // (batch) => (len_q, len_buf)
    const core::Tensor& position_bias,  // (batch) => (num_heads, len_q, len_buf)
    float scale,
    int max_len_buf,
    core::Tensor& output                 // (batch, len_q, num_heads, dim_head
);

struct AttentionWorkspace {
    core::Tensor cache;
    core::Tensor local_max;
    core::Tensor local_sum_exp;
};

AttentionWorkspace get_mqa_workspace(
    const core::Context& ctx,
    const core::Tensor& batch_q,
    int max_len_buf,
    bool is_quantized);

void multi_query_attention_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& batch_q,        // (batch, len_q, num_kv_heads * m_query, dim_head)
    const core::Tensor& buf_lens,       // (batch)
    const core::Tensor& key_buf_addrs,  // (batch) => (num_kv_heads, len_buf, dim_head)
    const core::Tensor& val_buf_addrs,  // (batch) => (num_kv_heads, len_buf, dim_head)
    const core::Tensor& mask,           // (batch) => (len_q, len_buf)
    const float scale,
    const int max_len_buf,
    core::Tensor& output,               // (batch, len_q, num_kv_heads * m_query, dim_head)
    const int m_query = 8,
    int algo_id = -1,
    const AttentionWorkspace& ws = {},
    const core::Tensor& scale_key_addrs = core::Tensor(),
    const core::Tensor& scale_val_addrs = core::Tensor(),
    core::DataType dequant_dtype = core::DataType::kHalf
);

// for encode only
void multi_query_self_attention(
    const core::Context& ctx,
    const core::Tensor& query,          // (len_q, num_kv_heads * m_query, dim_head)
    const core::Tensor& key_buf,        // (num_kv_heads, len_buf, dim_head)
    const core::Tensor& val_buf,        // (num_kv_heads, len_buf, dim_head)
    const core::Tensor& mask,           // (len_q, len_buf)
    float scale,
    core::Tensor& output,               // (len_q, num_kv_heads * m_query, dim_head)
    int high_precision
);
} // namespace nn
