#pragma one

#include "kvcache/transformer_buffer.h"
#include "model/model_context.h"
#include "utils/lru_cache.hpp"
#include <deque>
#include <memory>
#include <stack>

namespace batch_generator {

using bmengine::core::Tensor;
using kvcache::TransformerBuffer;
using model::ModelContext;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
//typedef shared_ptr<TransformerBuffer> BufferPtr;
typedef TransformerBuffer* BufferPtr;
typedef std::pair<BufferPtr, BufferPtr> BufferPair;
//typedef std::pair<Tensor, Tensor> BlockBuffer;

// Cache KV by chunk
class PrefixCache {

    typedef vector<int> Key;  // or std::array?

    size_t block_size;
    size_t max_block;
    utils::LRUCache<Key, int, utils::IntVecHasher> lru_cache;

    Tensor cache_mem;
    std::deque<int> unused_block;

    vector<Key> to_blocks(const std::vector<int>& tokens) {
        size_t num_block = (tokens.size() - 2) / block_size;  // Reserve last token for search
        num_block = std::min(num_block, max_block);
        vector<Key> blocks;
        blocks.reserve(num_block);
        for (size_t i = 0; i < num_block; ++i) {
            auto begin = tokens.begin() + i * block_size;
            blocks.emplace_back(begin, begin + block_size);
            blocks.back().push_back(int(i)); // Add block index into key
        }
        return std::move(blocks);
    }

    void save_block(ModelContext& ctx, const BufferPair& kv_buffer, size_t start, size_t len, int block_id) {
        auto p = cache_mem.slice_dim0_len(block_id * 2, 2).chunk();
        kv_buffer.first->dump_slice(ctx, start, len, &p[0]);  // k
        kv_buffer.second->dump_slice(ctx, start, len, &p[1]);  // v
    }

    void load_block(ModelContext& ctx, const BufferPair& kv_buffer, size_t start, size_t len, int block_id) {
        auto p = cache_mem.slice_dim0_len(block_id * 2, 2).chunk();
        kv_buffer.first->load_slice(ctx, start, len, p[0]);  // k
        kv_buffer.second->load_slice(ctx, start, len, p[1]);  // v
    }

    int get_block_id() {
        if (unused_block.empty()) {
            return lru_cache.pop_back();
        } else {
            int block_id = unused_block.back();
            unused_block.pop_back();
            return block_id;
        }
    }

public:
    PrefixCache(ModelContext& ctx,
                size_t num_block,
                size_t block_size,
                size_t max_block,
                size_t num_layers,
                size_t num_heads,
                size_t dim_head,
                bmengine::core::DataType dtype)
        : block_size(block_size), max_block(max_block), lru_cache(num_block) {
        cache_mem = ctx.is_BSHD()
            ? ctx.tensor({num_block * 2, num_layers, block_size, num_heads, dim_head}, dtype)
            : ctx.tensor({num_block * 2, num_layers, num_heads, block_size, dim_head}, dtype);
        for (int i = 0; i < num_block; ++i) {
            unused_block.push_back(i);
        }
    }

    BufferPair match(const vector<int>& tokens, int* matched_len) {
        vector<Key> keys = to_blocks(tokens);
        return {0, 0};
    }

    void put(ModelContext& ctx, const vector<int>& tokens, const BufferPair& kv_buffer) {
        vector<Key> blocks = to_blocks(tokens);
        if (blocks.empty()) return;
        for (int i = int(blocks.size()) - 1; i >= 0; --i) {
            int dumpy;
            if (lru_cache.get(blocks[i], dumpy)) {
                continue; // exist
            }
            int block_id = get_block_id();
            save_block(ctx, kv_buffer, i * block_size, block_size, block_id);
            lru_cache.put(blocks[i], block_id);
        }
    }

    int get(ModelContext& ctx, const vector<int>& tokens, const BufferPair& kv_buffer) {
        vector<Key> blocks = to_blocks(tokens);
        size_t matched_len = 0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            int block_id;
            if (!lru_cache.get(blocks[i], block_id)) {
                break;
            }
            load_block(ctx, kv_buffer, i * block_size, block_size, block_id);
            matched_len += block_size;
        }
        return matched_len;
    }
};

}