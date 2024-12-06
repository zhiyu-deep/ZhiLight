#include "nn/nn.h"
#include "nn/quant/int8/quant_kernel.h"
#include "bmengine/core/core.h"
#include "bmengine/functions/typecast.h"
// #include "bmengine/functions/quant.h"
#include "bmengine/logger/kernel_time_trace.hpp"
#include "bmengine/logger/std_log_op.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cuda.h>

using namespace bmengine::core;
using namespace bmengine::functions;
using namespace bmengine;
using std::vector;

__host__ void layernorm(
    const core::Tensor& input,  // (batch, seq_len, dim_model)
    const core::Tensor& weight, // (dim_model)
    core::Tensor* output,       // (batch, seq_len, dim_model)
    core::Tensor* scale_output, // (batch, seq_len)
    float eps,
    cudaStream_t stream);

template<DataType DT>
Tensor float_cast(const Context& ctx, const Tensor& t) {
    if (t.dtype() == DT) {
        return t;
    } else {
        return typecast(ctx, t, DT);
    }
}

template<DataType DT>
Tensor float_tensor(const Context& ctx, const std::vector<float>& data) {
    return float_cast<DT>(ctx, ctx.tensor_of(data));
}

template<DataType DT>
void test_layer_normal(const Context& ctx) {
    nn::LayerNorm ln(ctx, 10, true);
    size_t m = 2;
    size_t n = 10;
    std::vector<float> input_h(m * n, 2.);
std:
    iota(input_h.begin(), input_h.begin() + n, -5);
    std::vector<float> weight_h(n, 0.5);
    Tensor input = ctx.tensor_of(input_h, { m, n });
    Tensor weight = ctx.tensor_of(weight_h);
    input = float_cast<DT>(ctx, input);
    weight = float_cast<DT>(ctx, weight);

    *ln.parameters["weight"] = weight;

    std::cerr << "Input: " << input << std::endl;

    Tensor output = ln.forward(ctx, input);
    std::cerr << "Output: " << output << std::endl;

    Tensor quant_output = ln.forward(ctx, input);
    Tensor quant_scale = *quant_output.quant_scale.get();
    std::cerr << "Quant Output: " << quant_output << std::endl;
    std::cerr << "Quant scale: " << quant_scale << std::endl;

    Tensor quanted_int32 = typecast(ctx, quant_output, DataType::kInt32);
    Tensor scale_y = float_tensor<DT>(ctx, vector<float>(n, 1.));
    std::cerr << "scale_y: " << scale_y << std::endl;

    Tensor back = nn::quant_scale_back(ctx, quanted_int32, &quant_scale, &scale_y);
    std::cerr << "Quant Back: " << back << std::endl;
}

void test_quant_scale_back(const Context& ctx) {
    size_t m = 3;
    size_t n = 15360;
    std::vector<int> input_h(m * n, 1);
    std::vector<float> scale_x_h(m, 1.);
    std::vector<float> scale_y_h(n, 1.);
    auto input = ctx.tensor_of(input_h, { m, n });
    auto scale_x = ctx.tensor_of(scale_x_h);
    auto scale_y = ctx.tensor_of(scale_y_h);
    scale_x = typecast(ctx, scale_x, DataType::kHalf);
    scale_y = typecast(ctx, scale_y, DataType::kHalf);

    auto stream = ctx.current_stream()->ptr;
    cudaEvent_t start, stop;
    logger::createStartEvent(1, &start, &stop, stream);

    //    quant_scale_back(ctx, input, scale_x, scale_y);

    float elapsed_ms = logger::destroyDiffEvent(start, stop, stream);
    std::cout << "quant_scale_back take " << elapsed_ms << "ms, ret=" << input.shape() << std::endl;
}

int main() {
    bmengine::core::Engine engine({
        { 0, 1 * 1024 * 1024 * 1024 },
    });
    auto ctx = engine.create_context();
    auto with_dev = ctx.with_device(0);

    test_layer_normal<DataType::kFloat>(ctx);
    test_layer_normal<DataType::kHalf>(ctx);

    return 0;
}
