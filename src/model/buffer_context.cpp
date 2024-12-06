#include <bmengine/c10d/c10d.h>
#include <bmengine/functions/element.h>
#include <bmengine/functions/tensor_ops.h>
#include <bmengine/functions/typecast.h>
#include <bmengine/logger/std_log_op.hpp>
#include <numeric>

#include "model/buffer_context.h"
#include "model/allocate_util.hpp"
#include "model/model.h"
#include "kvcache/paged_kvcache.h"
#include "kvcache/transformer_buffer.h"
#include "utils/env.h"

namespace model {

using kvcache::PagedKVCache;
using kvcache::TransformerBuffer;

TransformerBufferContext::TransformerBufferContext(
    const ModelBase& m, int batch_size, bool parallel, int world_size, bool BSHD)
    : BufferContext(m, parallel), BSHD(BSHD) {
    int num_kv_heads = parallel ? m.num_kv_heads / world_size : m.num_kv_heads;
    buf_k_ = std::make_shared<TransformerBuffer>(
        batch_size, m.num_layers, num_kv_heads, m.dim_head, m.dtype, parallel, BSHD);
    buf_v_ = std::make_shared<TransformerBuffer>(
        batch_size, m.num_layers, num_kv_heads, m.dim_head, m.dtype, parallel, BSHD);
}

TransformerBufferContext::~TransformerBufferContext() = default;

Tensor* TransformerBufferContext::buf_k(size_t layer) {
    return &(*buf_k_)[layer];
}

Tensor* TransformerBufferContext::buf_v(size_t layer) {
    return &(*buf_v_)[layer];
}

const Tensor* TransformerBufferContext::block_table(size_t layer) {
    return nullptr;
}

void TransformerBufferContext::set_layer_devices(const std::vector<int>& layer_devices) {
    buf_k_->set_layer_devices(layer_devices);
    buf_v_->set_layer_devices(layer_devices);
}

void TransformerBufferContext::resize_transformer_buf(const Context& ctx, size_t new_length) {
    buf_k_->resize(ctx, new_length);
    buf_v_->resize(ctx, new_length);
    kvcache_len_ = new_length;
}

size_t TransformerBufferContext::add_sequence(
    const Context& ctx, const std::vector<int32_t>& prompt_tokens) {
    BM_ASSERT(false, "not implemented");
    return 0;
}

size_t TransformerBufferContext::remove_sequence(const Context& ctx, size_t seq_id) {
    BM_ASSERT(false, "not implemented");
    return 0;
}
size_t TransformerBufferContext::add_queries(
    const Context& ctx, std::vector<std::vector<int32_t>> query_tokens) {
    BM_ASSERT(false, "not implemented");
    return 0;
}

PagedBufferContext::PagedBufferContext(
    const PageConfig& page_config, const ModelBase& m, bool parallel, int world_size)
    : BufferContext(m, parallel) {
    int num_kv_heads = parallel ? m.num_kv_heads / world_size : m.num_kv_heads;
    kvcache_ = std::make_shared<PagedKVCache>(
        page_config, m.num_layers, num_kv_heads, m.dim_head, m.dtype, parallel);
}

PagedBufferContext::~PagedBufferContext() = default;

Tensor* PagedBufferContext::buf_k(size_t layer) {
    return &(kvcache_->key_cache(layer));
}

Tensor* PagedBufferContext::buf_v(size_t layer) {
    return &(kvcache_->value_cache(layer));
}

const Tensor* PagedBufferContext::block_table(size_t layer) {
    return kvcache_->block_table(layer);
}

void PagedBufferContext::set_layer_devices(const std::vector<int>& layer_devices) {
    kvcache_->set_layer_devices(layer_devices);
}

void PagedBufferContext::resize_transformer_buf(const Context& ctx, size_t new_length) {
    BM_ASSERT(false, "not implemented");
}

size_t PagedBufferContext::add_sequence(
    const Context& ctx, const std::vector<int32_t>& prompt_tokens) {
    auto bs = kvcache_->add_sequence(ctx, prompt_tokens);
    kvcache_len_ = std::max(kvcache_len_, prompt_tokens.size());
    return bs;
}

size_t PagedBufferContext::remove_sequence(const Context& ctx, size_t seq_id) {
    auto bs = kvcache_->remove_sequence(ctx, seq_id);
    return bs;
}

size_t PagedBufferContext::add_queries(
    const core::Context& ctx, std::vector<std::vector<int32_t>> tokens) {
    kvcache_len_ = kvcache_->add_queries(ctx, tokens);
    return kvcache_len_;
}

size_t PagedBufferContext::kvcache_len() {
    return kvcache_->block_table(0)->size(1);
}
}
