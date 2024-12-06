#include <vector>
#include <cassert>
#include "bmengine/core/core.h"
#include "kvcache/block_allocator.h"
#include "utils/matrix.hpp"

namespace kvcache {

using namespace bmengine;

BlockTrieNode::BlockTrieNode(
    const PageConfig& page_config,
    BlockAllocator* allocator,
    int32_t logical_block_id,
    int32_t phy_block_id,
    BlockTrieNode* parent)
    : page_config(page_config),
      allocator(allocator),
      logical_block_id(logical_block_id),
      phy_block_id(phy_block_id),
      parent(parent) { }

BlockTrieNode::~BlockTrieNode() { }

int32_t BlockTrieNode::ref_count() {
    return allocator->block_ref(phy_block_id);
}

int32_t BlockTrieNode::inc_ref_count() {
    return allocator->inc_block_ref(phy_block_id);
}

int32_t BlockTrieNode::dec_ref_count() {
    return allocator->dec_block_ref(phy_block_id);
}

bool BlockTrieNode::full() {
    return segment.size() == page_config.page_block_size;
}

size_t BlockTrieNode::num_seen_tokens() {
    return segment.size();
}

bool BlockTrieNode::can_add_tokens(size_t start, std::vector<int32_t> tokens) {
    if (start == segment.size())
        return true;
    for (int i = 0; i + start < segment.size(), i < tokens.size(); i++)
        if (segment.at(i + start) != tokens[i])
            return false;
    return true;
}

std::shared_ptr<BlockTrieNode> BlockTrieNode::reusable_sybling(
    size_t start, std::vector<int32_t> tokens) {
    std::vector<int32_t> full_segment(
        segment.begin(), std::min(segment.begin() + start, segment.end()));
    auto added_tokens_ = std::min(tokens.size(), page_config.page_block_size - full_segment.size());
    for (size_t t = 0; t < added_tokens_; t++) {
        full_segment.emplace_back(tokens[t]);
    }
    if (parent->has_child(full_segment))
        return parent->get_child(full_segment);
    return nullptr;
}

std::shared_ptr<BlockTrieNode> BlockTrieNode::clone(size_t start) {
    auto new_block = allocator->allocate_block(logical_block_id, parent);
    auto new_segment = std::vector<int32_t>(segment.begin(), segment.begin() + start);
    new_block->segment = new_segment;
    parent->add_sub_block(new_block);
    return new_block;
}

size_t BlockTrieNode::add_tokens(size_t start, std::vector<int32_t> tokens) {
    if (tokens.size() == 0)
        return 0;
    size_t added_tokens;
    assert(segment.size() >= start);
    std::vector<int32_t> new_key;
    if (start > 0)
        new_key = std::vector<int32_t>(segment.begin(), segment.begin() + start);
    std::vector<int32_t> old_key(new_key);
    added_tokens = std::min(tokens.size(), page_config.page_block_size - new_key.size());
    for (size_t i = 0; i < added_tokens; i++)
        new_key.emplace_back(tokens[i]);
    segment = new_key;
    reindex_sub_block(old_key, new_key);
    return added_tokens;
}

std::shared_ptr<BlockTrieNode> BlockTrieNode::add_sub_block(std::shared_ptr<BlockTrieNode> child) {
    children[child->segment] = child;
    return child;
}

void BlockTrieNode::remove_from_parent() {
    if (auto it = parent->children.find(segment); it != parent->children.end())
        parent->children.erase(it);
    return;
}

BlockTrieNode* BlockTrieNode::reindex_sub_block(
    const std::vector<int32_t>& old_key, const std::vector<int32_t>& new_key) {
    auto it = parent->children.find(old_key);
    if (it != parent->children.end()) {
        auto child = it->second;
        auto new_it = parent->children.find(new_key);
        if (new_it != parent->children.end()) {
            BM_ASSERT(new_it->second.get() == this, "diverged child");
        } else {
            parent->children[new_key] = child;
        }
        parent->children.erase(it);
    } else {
        BM_ASSERT(ref_count() > 1, "empty parent");
    }
    return this;
}

std::vector<std::shared_ptr<BlockTrieNode>> BlockAllocator::find_prefix_blocks(
    std::vector<int32_t> prefix) {
    std::vector<std::shared_ptr<BlockTrieNode>> res;
    BlockTrieNode* root = trie_root_.get();
    auto start = prefix.begin();
    while (root && start < prefix.end()) {

        auto segment = std::vector<int32_t>(
            start, std::min(start + page_config.page_block_size, prefix.end()));
        if (root->has_child(segment)) {
            auto child = root->get_child(segment);
            res.emplace_back(child);
            start += page_config.page_block_size;
            root = child.get();
        } else
            root = nullptr;
    }
    return res;
}

std::shared_ptr<BlockTrieNode> BlockAllocator::allocate_block(
    int32_t logical_block_id, BlockTrieNode* parent) {
    std::vector<int32_t> res;
    // TODO use a queue which is more efficient.
    for (size_t i = 0; i < block_indices.size(); i++) {
        if (block_indices[i] == 0) {
            auto block =
                std::make_shared<BlockTrieNode>(page_config, this, logical_block_id, i, parent);
            parent->add_sub_block(block);
            return block;
        }
    }
    BM_ASSERT(false, "no free blocks left");
    return nullptr;
}

int32_t BlockAllocator::block_ref(int32_t phy_block_id) {
    if (phy_block_id >= 0)
        return block_indices[phy_block_id];
    return 0;
}

int32_t BlockAllocator::inc_block_ref(int32_t phy_block_id) {
    if (phy_block_id >= 0)
        block_indices[phy_block_id] += 1;
    return block_indices[phy_block_id];
}

int32_t BlockAllocator::dec_block_ref(int32_t phy_block_id) {
    if (phy_block_id >= 0)
        block_indices[phy_block_id] -= 1;
    return block_indices[phy_block_id];
}

} // namespace kvcache