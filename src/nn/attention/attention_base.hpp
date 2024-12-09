#pragma once
#include "nn/attention/attention.h"
#include "nn/layernorm/layernorm.h"
#include "nn/linear/linear.h"
#include "nn/attention/attention_kernel.h"
#include "model/model_context.h"
#include <bmengine/core/core.h>

namespace nn {

class Attention::impl {
public:
    class NormalImpl;
    class MLAImpl;
    impl() = default;
    virtual ~impl() = default;
    impl(const impl&) = delete;
    impl(impl&&) = default;

    virtual core::Tensor dynamic_batch_forward(
        model::ModelContext& ctx,
        const core::Tensor& hidden_q,
        const core::Tensor& position_or_bias,
        core::Tensor *output)
    {
        throw std::runtime_error("Unsupported");
    }

    virtual void add_submodules(core::Layer* layer) {}
    virtual void on_load(const core::Context& ctx) {}

    static impl* create_mla_impl(const core::Context& ctx, const model::ModelConfig& cfg, model::QuantConfig quant);
};
}
