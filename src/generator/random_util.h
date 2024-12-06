#pragma once

#include <bmengine/core/context.h>
#include <bmengine/core/core.h>
#include <curand.h>
#include <unordered_map>
#include <vector>

namespace beam_utility {
using namespace bmengine;

void random_repetition_penalty(
    const core::Context& ctx,
    const std::vector<float>& penalty_factor,
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& batch_id,
    core::Tensor& logits);

void random_sampler_gpu(
    const core::Context& ctx,
    curandGenerator_t& gen,
    core::Tensor& probs,  // (..., n_classes)
    core::Tensor& select, // (...)
    float top_p = 1.0f,
    int top_k = 0,
    int num_samples = 1);

}
