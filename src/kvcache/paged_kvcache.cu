#include <map>
#include <memory>
#include <iostream>
#include <assert.h>

#include "bmengine/functions/init.h"
#include "kvcache/paged_kvcache.h"
#include "utils/matrix.hpp"

namespace kvcache {

using utils::Matrix2D;

template<class T>
static inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

// clang-format off
// gridDim (block_size, num_layers, 0),  blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(copy_block_kernel)(
    int32_t src_block_id,
    int32_t dst_block_id,
    size_t block_stride,
    size_t row_stride,
    T** __restrict__ cache_ptrs// (num_layers, (num_blocks, block_size, num_kv_heads, dim_head))
) {
    T*  cache_ptr = cache_ptrs[blockIdx.y];
    int row_id = blockIdx.x;
    size_t offset_src = src_block_id * block_stride + row_id * row_stride;
    size_t offset_dst = dst_block_id * block_stride + row_id * row_stride;
    for (int i = threadIdx.x; i < row_stride; i += blockDim.x) {
        cache_ptr[offset_dst + i] = cache_ptr[offset_src + i];
    }
}

// clang-format on

void copy_block(
    int32_t src_block_id,
    int32_t dst_block_id,
    int32_t num_layers,
    const std::vector<core::Tensor>& kv_caches, // (num_blocks, block_size, num_kv_heads, dim_head)
    cudaStream_t stream) {
    if (kv_caches.size() == 0)
        return;
    BM_ASSERT_EQ(kv_caches[0].ndim(), 4, "dim mismatch");
    size_t block_size = kv_caches[0].size(1);
    size_t block_stride = kv_caches[0].stride(0);
    size_t row_stride = kv_caches[0].stride(1);
    dim3 gridDim(block_size, num_layers, 1);
    dim3 blockDim(std::min(size_t(1024), round_up(row_stride, 32)), 1, 1);
    auto dtype = kv_caches[0].dtype();

    BM_DTYPE_DISPATCH(dtype, {
        std::vector<scalar_t*> pointers;
        scalar_t** device_pointers;
        BM_CUDART_ASSERT(cudaMalloc((void**) &device_pointers, num_layers * sizeof(scalar_t*)));
        for (size_t i = 0; i < num_layers; ++i)
            pointers.emplace_back(kv_caches[i].data<scalar_t>());
        BM_CUDART_ASSERT(cudaMemcpyAsync(
            device_pointers,
            pointers.data(),
            num_layers * sizeof(scalar_t*),
            cudaMemcpyHostToDevice,
            stream));
        BM_KERNEL(copy_block_kernel)<<<gridDim, blockDim, 0, stream>>>(
            src_block_id, dst_block_id, block_stride, row_stride, device_pointers);
        BM_CUDART_ASSERT(cudaFree(device_pointers));
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

PagedKVCache::PagedKVCache(
    const PageConfig& page_config,
    int num_layers,
    int num_heads,
    int dim_head,
    core::DataType dtype,
    bool parallel)
    : page_config(page_config),
      block_allocator(page_config),
      key_caches(num_layers),
      value_caches(num_layers),
      KVCache(0, num_layers, num_heads, dim_head, dtype, parallel, true) {
    BM_ASSERT(num_layers > 0, "num_layers must be greater than 0");
    BM_ASSERT(num_heads > 0, "num_heads must be greater than 0");
    BM_ASSERT(dim_head > 0, "dim_head must be greater than 0");
}
PagedKVCache::~PagedKVCache() {
    for (size_t seq_id = 0; seq_id < h_block_tables.size(); ++seq_id)
        for (auto iter = h_block_tables[seq_id].rbegin(); iter != h_block_tables[seq_id].rend();
             ++iter)
            iter->reset();
}

const core::Tensor& PagedKVCache::operator[](int i) const {
    BM_ASSERT(false, "not implementated. use key_cache() or value_cache() interface.");
    return key_caches[i]; // not reached
}
core::Tensor& PagedKVCache::operator[](int i) {
    BM_ASSERT(false, "not implementated. use key_cache() or value_cache() interface.");
    return key_caches[i]; // not reached
}

core::Tensor& PagedKVCache::key_cache(int i) {
    BM_ASSERT(
        i >= 0 && i < num_layers,
        "Invalid layer index: i must in [0, " + std::to_string(num_layers) + "), but got "
            + std::to_string(i));

    return key_caches[i];
}

core::Tensor& PagedKVCache::value_cache(int i) {
    BM_ASSERT(
        i >= 0 && i < num_layers,
        "Invalid layer index: i must in [0, " + std::to_string(num_layers) + "), but got "
            + std::to_string(i));

    return value_caches[i];
}

const core::Tensor* PagedKVCache::block_table(int i) const {
    return &block_table_;
}

size_t PagedKVCache::add_sequence(const core::Context& ctx, std::vector<int32_t> prompt_tokens) {
    size_t logical_blocks = ceil_div(prompt_tokens.size(), page_config.page_block_size);
    used_logical_blocks = std::max(used_logical_blocks, logical_blocks);
    h_block_tables.emplace_back(std::vector<std::unique_ptr<LogicalBlock>> {});
    h_seqlens.emplace_back(0);

    auto hit_blocks = block_allocator.find_prefix_blocks(prompt_tokens);
    for (size_t i = 0; i < hit_blocks.size(); i++) {
        auto phy_block_id = hit_blocks[i]->phy_block_id;
        h_block_tables[batch_size].emplace_back(
            std::make_unique<LogicalBlock>(std::move(hit_blocks[i])));
        h_seqlens[batch_size] += h_block_tables[batch_size][i]->num_seen_tokens();
    }

    for (size_t i = hit_blocks.size(); i < used_logical_blocks; i++) {
        h_block_tables[batch_size].emplace_back(
            std::make_unique<LogicalBlock>(block_allocator.allocate_block(
                i,
                i == 0 ? block_allocator.trie_root()
                       : h_block_tables[batch_size][i - 1]->get_phy_block())));
    }

    batch_size++;
    resize(ctx, prompt_tokens.size());

    sequence_add_tokens(
        ctx,
        batch_size - 1,
        std::vector<int32_t>(
            prompt_tokens.begin() + h_seqlens[batch_size - 1], prompt_tokens.end()));
    return batch_size;
}

size_t PagedKVCache::remove_sequence(const core::Context& ctx, size_t seq_id) {
    // free blocks in reverse order.
    for (auto iter = h_block_tables[seq_id].rbegin(); iter != h_block_tables[seq_id].rend(); ++iter)
        iter->reset();
    h_block_tables.erase(h_block_tables.begin() + seq_id);
    h_seqlens.erase(h_seqlens.begin() + seq_id);
    --batch_size;

    size_t max_seq_len = 0;
    for (int b = 0; b < batch_size; b++)
        max_seq_len = std::max(max_seq_len, h_seqlens[b]);
    resize(ctx, max_seq_len);
    return batch_size;
}

void PagedKVCache::sequence_add_tokens(
    const core::Context& ctx, size_t seq_id, std::vector<int32_t> tokens) {
    size_t start = h_seqlens[seq_id] / page_config.page_block_size;
    size_t n_start = h_seqlens[seq_id] % page_config.page_block_size;
    auto copy_start = tokens.begin();
    for (size_t i = start; i < used_logical_blocks; i++) {
        if (copy_start < tokens.end()) {
            auto parent = i == 0 ? block_allocator.trie_root()
                                 : h_block_tables[seq_id][i - 1]->get_phy_block();
            std::vector<int32_t> segment(
                copy_start, std::min(copy_start + page_config.page_block_size, tokens.end()));

            if (auto sybling =
                    h_block_tables[seq_id][i]->get_phy_block()->reusable_sybling(n_start, segment);
                sybling != nullptr) {
                if (sybling.get() != h_block_tables[seq_id][i]->get_phy_block())
                    h_block_tables[seq_id][i] = std::make_unique<LogicalBlock>(std::move(sybling));
            } else if (!h_block_tables[seq_id][i]->can_add_tokens(n_start, segment)) {
                // Copy-On-Write, don't copy the block until the output tokens are different.
                auto new_block = clone_block(ctx, h_block_tables[seq_id][i].get(), n_start);
                h_block_tables[seq_id][i] = std::move(new_block);
            }
            auto added_tokens = h_block_tables[seq_id][i]->add_tokens(n_start, segment);
            copy_start += added_tokens;
            h_seqlens[seq_id] += added_tokens;
        }
    }
}

size_t PagedKVCache::add_queries(
    const core::Context& ctx, std::vector<std::vector<int32_t>> query_tokens) {
    // TODO check batch match
    size_t max_seq_len = 0;
    for (int b = 0; b < query_tokens.size(); b++)
        max_seq_len = std::max(max_seq_len, h_seqlens[b] + query_tokens[b].size());
    resize(ctx, max_seq_len);
    for (int b = 0; b < query_tokens.size(); b++) {
        sequence_add_tokens(ctx, b, query_tokens[b]);
    }
    return block_table_.size(1);
}

void PagedKVCache::resize(const core::Context& ctx, size_t nw_length) {

    for (int i = 0; i < num_layers; i++) {
        if (key_caches[i].numel() == 0) {
            key_caches[i] = ctx.tensor(
                { page_config.num_blocks,
                  page_config.page_block_size,
                  size_t(num_heads),
                  size_t(dim_head) },
                dtype);
        }
        if (value_caches[i].numel() == 0) {
            value_caches[i] = ctx.tensor(
                { page_config.num_blocks,
                  page_config.page_block_size,
                  size_t(num_heads),
                  size_t(dim_head) },
                dtype);
        }
    }

    size_t logical_pages_per_seq = ceil_div(nw_length, page_config.page_block_size);
    used_logical_blocks = std::max(used_logical_blocks, logical_pages_per_seq);

    // align num blocks per batch.
    for (size_t b = 0; b < batch_size; b++) {
        auto& h_blocks = h_block_tables[b];
        auto origin_size = h_blocks.size();
        for (int n = origin_size; n < used_logical_blocks; n++) {
            auto parent = h_blocks[n - 1].get();
            // if parent is shared, use the same unfilled block for different sequence.
            // this delays  allocation to COW time.
            if (parent->get_phy_block()->ref_count() > 1
                && parent->get_phy_block()->has_child(std::vector<int32_t> {})) {
                h_blocks.emplace_back(std::make_unique<LogicalBlock>(
                    parent->get_phy_block()->get_child(std::vector<int32_t> {})));
            } else
                h_blocks.emplace_back(std::make_unique<LogicalBlock>(
                    block_allocator.allocate_block(n, parent->get_phy_block())));
        }
    }

    // update the block table.
    Matrix2D<int> h_block_table(batch_size, logical_pages_per_seq, 0);
    for (size_t b = 0; b < batch_size; b++)
        for (size_t n = 0; n < used_logical_blocks; n++)
            h_block_table(b, n) = h_block_tables[b][n]->phy_block_id();

    block_table_ = h_block_table.to_tensor(ctx);
}

std::unique_ptr<LogicalBlock> PagedKVCache::clone_block(
    const core::Context& ctx, LogicalBlock* block, size_t n_start) {
    auto new_block = block->get_phy_block()->clone(n_start);
    copy_block(
        block->phy_block_id(),
        new_block->phy_block_id,
        num_layers,
        key_caches,
        ctx.current_stream()->ptr);
    copy_block(
        block->phy_block_id(),
        new_block->phy_block_id,
        num_layers,
        value_caches,
        ctx.current_stream()->ptr);
    return std::make_unique<LogicalBlock>(std::move(new_block));
}

} // namespace kvcache