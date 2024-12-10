#pragma once
#include <bmengine/core/core.h>

#include <memory>

namespace int8_op {

using namespace bmengine;

static inline std::vector<size_t> get_scale_shape(const core::Tensor& x) {
    auto input_shape = x.shape();
    return std::vector<size_t>(input_shape.begin(), input_shape.end() - 1); // remove last dim
}

void quant_calc_scale(
    const core::Context& ctx,
    const core::Tensor& input,
    core::Tensor* output,
    core::Tensor* output_scale,
    int q_max=127,
    int q_zero=0);

core::Tensor quant_calc_scale(
    const core::Context& ctx, const core::Tensor& input, int q_max=127, int q_zero=0);

void set_quant_scale(core::Tensor& tensor, const core::Tensor& scale);

core::Tensor quant_scale_back(
    const core::Context& ctx,
    const core::Tensor& input,   // [m, n]
    const core::Tensor* scale_x, // [m]
    const core::Tensor* scale_y, // [n]
    core::DataType out_type = core::DataType::kDouble,
    core::Tensor* output = nullptr
);

void quant_scale_back3(
    const core::Context& ctx,
    const core::Tensor& input,   // (M, N)
    const core::Tensor* scale_x, // (M,)
    const core::Tensor* scale_y,  // (N,)
    int dim_q,
    int dim_kv,
    core::Tensor* q,
    core::Tensor* k,
    core::Tensor* v
);

void layernorm_quant(
    const core::Context& ctx,
    const core::Tensor& input,  // (batch, seq_len, dim_model)
    const core::Tensor& weight, // (dim_model)
    core::Tensor* output,       // (batch, seq_len, dim_model)
    core::Tensor* output_int8,  // (batch, seq_len, dim_model)
    core::Tensor* scale_output, // (batch, seq_len)
    float eps,
    float scale);

core::Tensor quant_back_element_add_scale(
    const core::Context& ctx,
    const core::Tensor& input,   // (M, N)
    const core::Tensor* scale_x, // (M,)
    const core::Tensor* scale_y, // (N,)
    const core::Tensor& input_b,
    float scale);

core::Tensor quant_back_transpose(
    const core::Context& ctx,
    const bmengine::core::Tensor& input, // (batch, len_q, num_heads, dim_head)
    const core::Tensor* scale_x,
    const core::Tensor* scale_y);

core::Tensor quant_back_act_mul(
    const core::Context& ctx,
    const core::Tensor& A,         // (M, N)
    const core::Tensor* a_scale_x, // (M,)
    const core::Tensor* a_scale_y, // (N,)
    const core::Tensor& B,         // (M, N)
    const core::Tensor* b_scale_x, // (M,)
    const core::Tensor* b_scale_y, // (N,)
    const std::string& act_type);

void quant_back_copy_to_buffer(
    const core::Context& ctx,
    int num_heads,
    int len_kv,
    int len_buf,
    int dim_head,
    const core::Tensor* placement,
    const core::Tensor& src,
    const core::Tensor* scale_x,
    const core::Tensor* scale_y,
    const core::Tensor& dst);

std::tuple<core::Tensor, core::Tensor> quant_group_32(
    const core::Context& ctx,
    const core::Tensor& input);

void dequant_group(
    const core::Context& ctx,
    const core::Tensor& q,
    const core::Tensor& scale,
    core::Tensor* out,
    int q_zero
);

void dequant_group_32(
    const core::Context& ctx,
    const core::Tensor& q,
    const core::Tensor& scale,
    core::Tensor* out
);

core::Tensor dequant_group_fuse_add(
    const core::Context& ctx,
    const core::Tensor& q,
    const core::Tensor& scale,
    const core::Tensor& c
);

void dequant_sum_quant_g32(
    const core::Context& ctx,
    const core::Tensor& my,           // (M, GROUP_SIZE)
    const core::Tensor& q_others,     // (WS - 1, M, GROUP_SIZE)
    const core::Tensor& scale_others, // (WS - 1, M)
    core::Tensor* q_sum,              // (M, GROUP_SIZE)
    core::Tensor* scale_sum
);

} // namespace nn
