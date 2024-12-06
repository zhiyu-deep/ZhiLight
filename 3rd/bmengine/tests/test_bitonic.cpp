#include <iostream>
#include "bmengine/core/core.h"
#include "bmengine/functions/all.h"
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>
#include <algorithm>

using namespace bmengine;

template<typename T>
T t_random();

template<>
double t_random() {
    return (double) rand() / ((double) RAND_MAX / 10);
}

template<>
float t_random() {
    return (float) rand() / ((float) RAND_MAX / 10);
}

template<>
half t_random() {
    return __float2half(t_random<float>());
}

template<>
int32_t t_random() {
    return rand();
}

template<>
int16_t t_random() {
    return rand() % 32768;
}

template<>
int8_t t_random() {
    return rand() % 128;
}

template<typename T>
core::DataType t_type() {
    if (std::is_same<T, int8_t>::value)
        return core::DataType::kInt8;
    if (std::is_same<T, int16_t>::value)
        return core::DataType::kInt16;
    if (std::is_same<T, int32_t>::value)
        return core::DataType::kInt32;
    if (std::is_same<T, float>::value)
        return core::DataType::kFloat;
    if (std::is_same<T, double>::value)
        return core::DataType::kDouble;
    if (std::is_same<T, half>::value)
        return core::DataType::kHalf;
    BM_EXCEPTION("Unknown type");
}

template<typename T>
bool run_one_test(const core::Context& ctx, size_t batch, size_t n, size_t top, bool print = false) {
    core::DataType dtype = t_type<T>();
    T* inp = new T[batch * n];
    T* out = new T[batch * top];
    int* idx = new int[batch * top];
    T* ans = new T[batch * n];
    for (int i = 0; i < batch * n; i++) {
        inp[i] = t_random<T>();
        ans[i] = inp[i];
    }
    core::Tensor tensor_inp = ctx.tensor({ batch, n }, dtype);
    core::Tensor tensor_out = ctx.tensor({ batch, top }, dtype);
    core::Tensor tensor_idx = ctx.tensor({ batch, top }, core::DataType::kInt32);
    tensor_inp.from_buffer(inp);

    functions::bitonic_topk(ctx, tensor_inp, tensor_out, tensor_idx);

    for (int i = 0; i < batch; ++i) {
        std::sort(ans + (i * n), ans + ((i + 1) * n));
    }
    tensor_out.to_buffer(out);
    tensor_idx.to_buffer(idx);

    bool ret = true;
    for (int i = 0; i < batch; i++) {
        if (print) {
            std::cout << "input" << std::endl;
            for (int j = 0; j < n; j++) {
                std::cout << std::setw(8) << std::setprecision(4) << inp[i * n + j] << " ";
            }
            std::cout << std::endl;
            std::cout << "output" << std::endl;
            for (int j = 0; j < top; j++) {
                std::cout << std::setw(8) << std::setprecision(4) << out[i * top + j] << " ";
            }
            std::cout << std::endl;
            std::cout << "ans" << std::endl;
            for (int j = 0; j < top; j++) {
                std::cout << std::setw(8) << std::setprecision(4) << ans[i * n + n - j - 1] << " ";
            }
            std::cout << std::endl;
            std::cout << "idx" << std::endl;
            for (int j = 0; j < top; j++) {
                std::cout << std::setw(8) << idx[i * top + j] << " ";
            }
            std::cout << std::endl;
        }
        bool correct = true;
        for (int j = 0; j < top; ++j) {
            if (out[i * top + j] != ans[i * n + n - j - 1]) {
                correct = false;
                break;
            }
            if (idx[i * top + j] < 0 || idx[i * top + j] >= n) {
                correct = false;
                break;
            }
            if (inp[idx[i * top + j] + i * n] != out[i * top + j]) {
                correct = false;
                break;
            }
        }
        if (print) {
            if (correct) {
                std::cout << "correct" << std::endl;
            } else {
                std::cout << "wrong" << std::endl;
            }
            std::cout << std::endl;
        }

        ret &= correct;
    }

    delete[] inp;
    delete[] out;
    delete[] idx;
    delete[] ans;
    return ret;
}

bool all_pass = true;

#define RUN_ONE_TEST_CASE(type, batch, n, top, idx, total)                                                                     \
    if (run_one_test<type>(ctx, batch, n, top)) {                                                                              \
        std::cout << "[" #type "] Test " << idx << "/" << total << " passed" << std::endl;                                     \
    } else {                                                                                                                   \
        std::cout << "[" #type "] Test " << idx << "/" << total << " failed" << std::endl;                                     \
        all_pass = false;                                                                                                      \
    }

#define RUN_TEST_CASE(type, total)                                                                                             \
    for (int i = 0; i < total; i++) {                                                                                          \
        int batch = 128;                                                                                                       \
        int n = t_random<int>() % 100000 + 10;                                                                                 \
        int top = t_random<int>() % 32 + 1;                                                                                    \
        RUN_ONE_TEST_CASE(type, batch, n, top, i + 1, total)                                                                   \
    }

int main() {
    srand(0);

    bmengine::core::Engine engine({
        { 0, 1ll * 1024 * 1024 * 1024 },
    });
    auto ctx = engine.create_context({ 0 });
    bmengine::core::WithDevice device(ctx, 0);

    RUN_TEST_CASE(int8_t, 10);
    RUN_TEST_CASE(int16_t, 10);
    RUN_TEST_CASE(int32_t, 10);
    RUN_TEST_CASE(half, 10);
    RUN_TEST_CASE(float, 10);
    RUN_TEST_CASE(double, 10);

    if (all_pass) {
        std::cout << "All tests passed" << std::endl;
    } else {
        std::cout << "Some tests failed" << std::endl;
    }
    return 0;
}