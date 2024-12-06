#pragma once

#include <list>
#include <unordered_map>
#include <assert.h>

namespace utils {

struct IntVecHasher {
    size_t operator()(const std::vector<int>& vec) const {
        uint32_t seed = uint32_t(vec.size());
        for (auto& i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return size_t(seed);
    }
};

template<class K, class V, class HASH = std::hash<K>>
class LRUCache {
private:
    std::list<std::pair<K, V> > item_list;
    std::unordered_map<K, decltype(item_list.begin()), HASH> item_map;
    size_t cache_size_;
    void clean() {
        while (item_map.size() > cache_size_) {
            auto last_it = item_list.end();
            last_it--;
            item_map.erase(last_it->first);
            item_list.pop_back();
        }
    };

public:
    explicit LRUCache(int cache_size) : cache_size_(cache_size) {}

    V pop_back() {
        if (item_list.empty()) throw std::runtime_error("Empty");
        auto last_it = --item_list.end();
        V v = last_it->second;
        item_map.erase(last_it->first);
        item_list.pop_back();
        return v;
    }

    void put(const K& key, const V& val) {
        auto it = item_map.find(key);
        if (it != item_map.end()) {
            item_list.erase(it->second);
            item_map.erase(it);
        }
        if (item_map.size() >= cache_size_) pop_back();
        item_list.push_front(make_pair(key, val));
        item_map.insert(make_pair(key, item_list.begin()));
    }

    bool exist(const K& key) {
        return (item_map.find(key) != item_map.end());
    }

    bool get(const K& key, V& v) {
        auto it = item_map.find(key);
        if(it == item_map.end()) return false;
        item_list.splice(item_list.begin(), item_list, it->second);
        v = it->second->second;
        return true;
    }

};
}  // namespace utils