#pragma once
#include <bmengine/core/core.h>

namespace nn {
using namespace bmengine;

void copy_to_buffer_batch_place(
    cudaStream_t stream,
    const core::Tensor& batch_place,
    const core::Tensor& src,
    core::Tensor* dst);

void copy_to_buffer_batch_place(
    cudaStream_t stream,
    const core::Tensor& batch_place,
    const core::Tensor& placement,
    const core::Tensor& src,
    core::Tensor* dst);

void copy_to_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& src,
    const core::Tensor& placement,
    const core::Tensor& buf_lens,
    const core::Tensor& buf_addr
);
void copy_to_rag_buffer2(
    const core::Context& ctx,
    const core::Tensor& placement,
    const core::Tensor& buf_lens,
    const core::Tensor& k_src,
    const core::Tensor& v_src,
    core::Tensor* buf_k_addr,
    core::Tensor* buf_v_addr,
    bool is_scale = false
);

} // namespace nn
