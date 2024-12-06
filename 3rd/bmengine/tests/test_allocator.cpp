#include "bmengine/core/core.h"
#include <numeric>
#include <iostream>
#include <thread>
#include <vector>

#define PRINT_USAGE(ctx) std::cout << __FILE__ << " " << __PRETTY_FUNCTION__ << ":" << __LINE__ << std::endl << "Used : " << (ctx).used_memory() << std::endl << std::endl

bmengine::core::Tensor work1(bmengine::core::Context &ctx) {
    auto A = ctx.tensor({75, 1024}, bmengine::core::DataType::kFloat);
    auto B = ctx.tensor({75, 1024}, bmengine::core::DataType::kFloat);
    auto C = ctx.tensor({75, 1024}, bmengine::core::DataType::kFloat);
    PRINT_USAGE(ctx);
    std::cout << "[ptr] A: " << A.data() << " B: " << B.data() << " C: " << C.data() << std::endl << std::endl;
    return B;
}

void work2(bmengine::core::Context &ctx, bmengine::core::Tensor *v) {
    const bmengine::core::Tensor &key_buf = v == nullptr ? ctx.tensor({32}, bmengine::core::DataType::kFloat) : *v;
    PRINT_USAGE(ctx);
}

void test_mem_cpy_overlap(bmengine::core::Context &ctx) {
    bmengine::core::Tensor t = ctx.tensor({4096}, bmengine::core::DataType::kInt32);
    size_t n = 3 * 1024;
    size_t offset = 1024;
    std::vector<int> ivec(n);
    std::iota(ivec.begin(), ivec.end(), 0);
    t.slice_dim0(offset, 4096).from_buffer(ivec.data());
    BM_CUDART_ASSERT(cudaMemcpy(t.data(), t.data<int>() + offset, n * sizeof(int), cudaMemcpyDeviceToDevice));
    std::vector<int> ivec2(n);
    t.slice_dim0(0, n).to_buffer(ivec2.data());
    BM_ASSERT(ivec == ivec2, "data mismatch after memory moved");
}


int main() {
    bmengine::core::Engine engine(
        {
            {0, 1ll * 1024 * 1024},
            {1, 2ll * 1024 * 1024},
            {2, 3ll * 1024 * 1024},
            {3, 4ll * 1024 * 1024}
        }
    );
    auto ctx = engine.create_context({0});
    bmengine::core::WithDevice device(ctx, 0);
    test_mem_cpy_overlap(ctx);
    {
        PRINT_USAGE(ctx);
        auto v = work1(ctx);
        std::cout << "[ptr] v: " << v.data() << std::endl << std::endl;
        PRINT_USAGE(ctx);
        {
            auto A = ctx.tensor({150, 1024}, bmengine::core::DataType::kFloat);
            std::cout << "[ptr] v: " << v.data() << " A: " << A.data() << std::endl << std::endl;
            PRINT_USAGE(ctx);
        }
        PRINT_USAGE(ctx);
    }
    PRINT_USAGE(ctx);
    {
        auto A = ctx.tensor({75, 1024}, bmengine::core::DataType::kFloat);
        PRINT_USAGE(ctx);
        work2(ctx, &A);
        work2(ctx, nullptr);
    }
    PRINT_USAGE(ctx);
    return 0;
}