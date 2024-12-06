#pragma once
#include <cstdint>

template<typename T>
class ArrayGuard {
public:
    T* ptr;
    ArrayGuard() : ptr(nullptr) { }
    ArrayGuard(size_t s) : ptr(new T[s]) { }
    ~ArrayGuard() {
        if (ptr != nullptr)
            delete[] ptr;
    }
    ArrayGuard(const ArrayGuard&) = delete;
    ArrayGuard(ArrayGuard&& other) {
        ptr = other.ptr;
        other.ptr = nullptr;
    }
    ArrayGuard& operator=(const ArrayGuard&) = delete;
    ArrayGuard& operator=(ArrayGuard&& other) {
        ptr = other.ptr;
        other.ptr = nullptr;
        return *this;
    }

    uintptr_t uint_ptr() { return reinterpret_cast<uintptr_t>(ptr); }
};
