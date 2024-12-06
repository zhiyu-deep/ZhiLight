
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <tuple>
#include <thread>
#include <iostream>
#include <random>
#include <numeric>
#include <csignal>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "nn/nn.h"
#include "model/model.h"
#include "utils/exception.h"
#include "py_export/py_utils.h"
#include "bind_internal.h"
#include "internal_utils.h"
#include "layer_base.hpp"
#include "kvcache/transformer_buffer.h"
#include "kvcache/paged_kvcache.h"

namespace py = pybind11;
using kvcache::PageConfig;
using kvcache::PagedKVCache;

class PyAttention : public PyLayerBase<nn::Attention> {
private:
    model::ModelConfig model_config;

public:
    PyAttention(model::ModelConfig model_config, bool parallel = false)
        : PyLayerBase<nn::Attention>(model_config, 0, parallel), model_config(model_config) {};

    static PyAttention create(
        int dim_model,
        int num_heads,
        int dim_head,
        std::string pos_bias_type,
        int quant,
        bool scale_weights = false,
        bool weight_transposed = true,
        bool parallel = false) {
        model::ModelConfig model_config(
            "",
            0,
            dim_model,
            num_heads,
            dim_head,
            0,
            0,
            1e-6,
            -1,
            {},
            scale_weights,
            weight_transposed,
            0,
            1.0,
            1.0,
            bmengine::core::DataType::kHalf);
        auto layer = PyAttention(model_config, parallel);
        return layer;
    };

    at::Tensor forward(
        at::Tensor& input,
        at::Tensor& mask,
        at::Tensor& position,
        at::Tensor& seqlens_q,
        at::Tensor& seqlens_kv) {

        at::Tensor output;

        auto t_input = bind::aten_to_tensor(*ctx, input);
        auto t_mask = bind::aten_to_tensor(*ctx, mask);
        auto t_position = bind::aten_to_tensor(*ctx, position);
        auto t_seqlens_q = bind::aten_to_tensor(*ctx, seqlens_q);
        auto t_seqlens_kv = bind::aten_to_tensor(*ctx, seqlens_kv);

        // int len_buf = round_up(t_input.size(1), 32);
        int len_buf = t_mask.size(-1);
        PageConfig page_config { 128, 16 };
        PagedKVCache kv_cache(
            page_config,
            1,
            model_config.num_heads,
            model_config.dim_head,
            bmengine::core::DataType::kHalf,
            true);
        for (size_t i = 0; i < t_input.size(0); i++) {
            std::vector<int32_t> h_seq(t_input.size(1));
            std::iota(h_seq.begin(), h_seq.end(), i * t_input.size(1));
            kv_cache.add_sequence(*ctx, h_seq);
        }

        auto res = layer->forward(
            *ctx,
            t_input,
            t_mask,
            t_position,
            t_seqlens_q,
            t_seqlens_kv,
            &(kv_cache.key_cache(0)),
            &(kv_cache.value_cache(0)),
            kv_cache.block_table(0),
            nullptr,
            nullptr);

        output = bind::tensor_to_aten(*ctx, res);
        return output;
    };
};

namespace bind {
void define_layer_attention(py::module_& layers_m) {
    py::class_<PyAttention>(layers_m, "Attention")
        .def(py::init(&PyAttention::create))
        .def("load_state_dict", &PyAttention::load_state_dict)
        .def("named_parameters", &PyAttention::named_parameters)
        .def("forward", &PyAttention::forward);
}

}
