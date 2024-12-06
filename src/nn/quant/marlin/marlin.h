#pragma one

#include "bmengine/core/core.h"

using namespace bmengine;

core::Tensor gptq_marlin_repack(
    const core::Context& ctx,
    core::Tensor& b_q_weight,
    core::Tensor& perm,
    size_t size_k,
    size_t size_n,
    int64_t num_bits);

core::Tensor gptq_marlin_gemm(
    const core::Context& ctx,
    const core::Tensor& a,
    core::Tensor& b_q_weight,
    core::Tensor& b_scales, core::Tensor& b_zeros,
    core::Tensor& g_idx, core::Tensor& perm,
    core::Tensor& workspace,
    // vllm::ScalarType* b_q_type,
    size_t size_m, size_t size_n, size_t size_k,
    bool is_k_full, bool has_zp,
    bool use_fp32_reduce);