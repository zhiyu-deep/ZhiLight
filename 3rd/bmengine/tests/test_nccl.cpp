#include "bmengine/core/core.h"
#include "bmengine/c10d/c10d.h"
#include "bmengine/functions/all.h"
#include <thread>
#include <iostream>
#include <cuda_fp16.h>

int main() {
    std::vector<bmengine::core::DeviceConfiguration> devices;
    int gpu_num;
    BM_CUDART_ASSERT(cudaGetDeviceCount(&gpu_num));
    for (int i = 0; i < gpu_num; ++i) {
        devices.emplace_back(i, 1ll * 1024 * 1024 * 1024);
    }
    bmengine::core::Engine engine(devices);
    auto ctx = engine.create_context();
    std::cout << ctx.world_size() << std::endl;

    /*
      we compare an columnar-row splited matrices with normal gemm:
      [A1, A2] @ [B1,
                  B2]
      since we don't have a distributed tensor implementation, partations
      are allocated directly on devices.
    */
    std::vector<bmengine::core::Tensor> inputs;
    std::vector<bmengine::core::Tensor> weights;
    std::vector<bmengine::core::Tensor> outputs;
    for (int i = 0; i < ctx.world_size(); ++i) {
        bmengine::core::WithDevice device(ctx, i);
        auto input = ctx.tensor({2, 3}, bmengine::core::DataType::kHalf);
        auto W = ctx.tensor({3, 2}, bmengine::core::DataType::kHalf);
        auto o = ctx.tensor({2, 2}, bmengine::core::DataType::kHalf);
        bmengine::functions::ones_(ctx, input);
        bmengine::functions::ones_(ctx, W);
        bmengine::functions::zeros_(ctx, o);
        inputs.push_back(input);
        weights.push_back(W);
        outputs.push_back(o);
    }
    bmengine::c10d::NCCLGroupStart();
    for (int i = 0; i < ctx.world_size(); ++i) {
        std::cout << i << std::endl;
        bmengine::core::WithDevice device(ctx, i);
        bmengine::functions::Gemm gemm(ctx, bmengine::core::DataType::kHalf, false, false);
        auto res = gemm(ctx, inputs[i], weights[i]);
        std::cout << res << std::endl;
        bmengine::c10d::NCCLAllReduce(ctx, res, outputs[i], ncclSum);
    }
    bmengine::c10d::NCCLGroupEnd();
    for (int i = 0; i < ctx.world_size(); ++i) {
        std::cout << outputs[i] << std::endl;
    }

    bmengine::core::WithDevice device(ctx, 0);
    auto input = ctx.tensor({2, (size_t)3 * ctx.world_size()}, bmengine::core::DataType::kHalf);
    auto W = ctx.tensor({(size_t)3 * ctx.world_size(), 2}, bmengine::core::DataType::kHalf);
    bmengine::functions::ones_(ctx, input);
    bmengine::functions::ones_(ctx, W);
    bmengine::functions::Gemm gemm(ctx, bmengine::core::DataType::kHalf, false, false);
    auto res = gemm(ctx, input, W);
    std::cout << res << std::endl;
}