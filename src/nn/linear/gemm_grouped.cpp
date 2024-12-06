#include "nn/linear/gemm_grouped.h"
#include <bmengine/logger/std_log_op.hpp>

namespace nn {

using namespace bmengine;
using bmengine::core::DataType;
using bmengine::core::Tensor;
using std::vector;

static void check_dtype(DataType dtype, const vector<Tensor>& tensors) {
    for (auto& t : tensors) {
        BM_ASSERT_EQ(dtype, t.dtype(), "");
    }
}

static vector<int> get_dims(const vector<Tensor>& tensors, int dim) {
    vector<int> results;
    for (auto& t : tensors) {
        BM_ASSERT_EQ(t.ndim(), 2, "Not 2D matrix");
        results.push_back(t.size(dim));
    }
    return std::move(results);
}

static vector<int> get_strides(const vector<Tensor>& tensors, int dim) {
    vector<int> results;
    for (auto& t : tensors) {
        results.push_back(t.stride(dim));
    }
    return std::move(results);
}

static Tensor from_host(const core::Context& ctx, const vector<void*>& ptrs) {
    Tensor a = ctx.tensor({ptrs.size()}, DataType::kDouble);
    auto stream = ctx.current_stream()->ptr;
    BM_CUDART_ASSERT(cudaMemcpyAsync(a.data(), ptrs.data(), a.nbytes(), cudaMemcpyHostToDevice, stream));
    return std::move(a);
}

static vector<void*> get_addresses(const core::Context& ctx, const vector<Tensor>& tensors) {
    vector<void*> ptrs;
    for (auto& t : tensors) {
        ptrs.push_back(t.data());
    }
    return std::move(ptrs);
}

// See https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Extensions/GemmGroupedBatchedEx/cublas_GemmGroupedBatchedEx_example.cu
void gemm_grouped(
    const core::Context& ctx,
    DataType dtype,
    const vector<Tensor>& inputs,   // B: (n, k)
    const vector<Tensor>& weights,  // A: (m, k)
    vector<Tensor>& results         // C: (n, m) !
) {
    BM_ASSERT(dtype == DataType::kHalf || dtype == DataType::kBFloat16,  "");
    BM_ASSERT(!inputs.empty(), "");
    BM_ASSERT_EQ(inputs.size(), weights.size(), "");
    if (results.empty()) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            results.push_back(ctx.tensor({inputs[i].size(0), weights[i].size(0)}, dtype));
        }
    }
    BM_ASSERT_EQ(inputs.size(), results.size(), "");
    check_dtype(dtype, inputs);
    check_dtype(dtype, weights);
    check_dtype(dtype, results);

    cudaDataType_t cuda_data_type = dtype == DataType::kHalf ? CUDA_R_16F : CUDA_R_16BF;
    int group_count = inputs.size();
    vector<int> group_size(group_count, 1);
    vector<cublasOperation_t> transa_array(group_count, CUBLAS_OP_T);  // CUBLAS_OP_T
    vector<cublasOperation_t> transb_array(group_count, CUBLAS_OP_N);
    vector<int> m_array = get_dims(weights, 0);
    vector<int> n_array = get_dims(inputs, 0);
    vector<int> k_array = get_dims(inputs, 1);
    vector<int> k_array1 = get_dims(weights, 1);
    vector<int> m_array1 = get_dims(results, 1);
    vector<int> n_array1 = get_dims(results, 0);
    BM_ASSERT_EQ(m_array, m_array1, "Weights and results 0-dim mismatch");
    BM_ASSERT_EQ(n_array, n_array1, "Input and results 'N' mismatch");
    BM_ASSERT_EQ(k_array, k_array1, "Input and weights last dim mismatch");
    vector<int> lda_array = get_strides(weights, 0);
    vector<int> ldb_array = get_strides(inputs, 0);
    vector<int> ldc_array = get_strides(results, 0);
    vector<float> alpha_array(group_count, 1);
    vector<float> beta_array(group_count, 0);

    vector<void*> a_array = get_addresses(ctx, weights);
    vector<void*> b_array = get_addresses(ctx, inputs);
    vector<void*> c_array = get_addresses(ctx, results);
    auto stream = ctx.current_stream()->ptr;
    Tensor d_a = from_host(ctx, a_array);
    Tensor d_b = from_host(ctx, b_array);
    Tensor d_c = from_host(ctx, c_array);

#if CUDART_VERSION >= 12050 && defined(ENABLE_GEMM_GROUPED)
    BM_CUBLAS_ASSERT(cublasGemmGroupedBatchedEx(
            ctx.cublas_handle(),
            transa_array.data(), transb_array.data(),
            m_array.data(), n_array.data(), k_array.data(),
            alpha_array.data(),
            (void **)d_a.data(), cuda_data_type, lda_array.data(),
            (void **)d_b.data(), cuda_data_type, ldb_array.data(),
            beta_array.data(),
            (void **)d_c.data(), cuda_data_type, ldc_array.data(),
            group_count, group_size.data(),
            CUBLAS_COMPUTE_32F));
    BM_CUDART_ASSERT(cudaGetLastError());
    BM_CUDART_ASSERT(cudaStreamSynchronize(stream));
#else
    throw std::runtime_error("Need cuda 12.5+");
#endif
}

}
