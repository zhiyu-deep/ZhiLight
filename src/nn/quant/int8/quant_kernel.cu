#include "nn/quant/int8/quant_kernel.h"
#include "nn/functions/activation.cuh"
#include <bmengine/core/core.h>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/logger/std_log_op.hpp>

namespace int8_op {

using namespace bmengine;
using bmengine::core::DataType;

// gridDim(M) blockDim(1024)
template<typename T, int Q_MAX=127>
static __global__ void BM_KERNEL(quant_calc_scale)(
    size_t k,
    const T* __restrict__ inp, // (M * K)
    float* __restrict__ scale, // (M)
    int8_t* __restrict__ out,  // (M * K)
    int q_zero
) {
    static constexpr float Q_MAX_F = Q_MAX;
    size_t offset = blockIdx.x * k;
    float abs_max = 0;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float v = inp[offset + i];
        v = v < 0 ? -v : v;
        abs_max = v > abs_max ? v : abs_max;
    }
    abs_max = functions::blockReduceMax<T>(abs_max);
    float block_scale = Q_MAX_F / abs_max;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        if (i < k) {
            float round_v = nearbyintf(float(inp[offset + i]) * block_scale);
            if (q_zero) {
                reinterpret_cast<uint8_t*>(out)[offset + i] = uint8_t(q_zero + round_v);
            } else {
                out[offset + i] = int8_t(round_v);
            }
        }
    }

    if (threadIdx.x == 0) {
        scale[blockIdx.x] = abs_max / Q_MAX_F;
    }
}

void quant_calc_scale(
    const core::Context& ctx,
    const core::Tensor& input,
    core::Tensor* output,
    core::Tensor* output_scale,
    int q_max,
    int q_zero) {
    BM_ASSERT(q_max == 127 || q_max == 16, "Unsupported q_max");
    if (q_zero) {
        BM_ASSERT(q_zero > q_max, "Too small zero point to make value unsigned");
    }
    int k = input.size(-1);
    int row = input.numel() / k;

    if (output->shape() != input.shape()) {
        *output = ctx.tensor(input.size(), DataType::kInt8, "", round_up(32 * k, 1024));
    }
    auto scale_shape = get_scale_shape(input);
    if (output_scale->shape() != scale_shape) {
        *output_scale = ctx.tensor(scale_shape, DataType::kFloat);
    }

    int threads = round_up_thread(k);
    dim3 gridDim(row);
    dim3 blockDim(threads);
    cudaStream_t stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
        auto kernel = BM_KERNEL(quant_calc_scale)<scalar_t, 127>;
        if (q_max == 16) {
            kernel = BM_KERNEL(quant_calc_scale)<scalar_t, 16>;
        }
        kernel<<<gridDim, blockDim, 0, stream>>>(
            k,
            input.data<scalar_t>(),
            output_scale->mutable_data<float>(),
            output->mutable_data<int8_t>(),
            q_zero);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

void set_quant_scale(core::Tensor& tensor, const core::Tensor& quant_scale) {
    tensor.quant_scale = std::make_shared<core::Tensor>();
    *tensor.quant_scale = quant_scale;
}

core::Tensor quant_calc_scale(
    const core::Context& ctx, const core::Tensor& input, int q_max, int q_zero
) {
    core::Tensor output, quant_scale;
    quant_calc_scale(ctx, input, &output, &quant_scale, q_max, q_zero);
    set_quant_scale(output, quant_scale);
    return output;
}

// gridDim (seq_len, 1, batch),     blockDim(1024, 1, 1)
template<typename T, typename T_SCALE>
static __global__ void BM_KERNEL(fuse_layernorm_rms_quant)(
    int dim_model,
    float eps,
    float scale,
    int input_stride,               // seq_len * dim_model
    const T* __restrict__ weight,   // (dim_model)
    const T* __restrict__ input,    // (batch, seq_len, dim_model)
    T* __restrict__ output,         // (batch, seq_len, dim_model)
    int8_t* __restrict__ out_int8,  // (batch, seq_len, dim_model)
    T_SCALE* __restrict__ out_scale // (batch, seq_len)
) {
    extern __shared__ float smem[]; // (dim_model)

    int offset = blockIdx.z * input_stride + blockIdx.x * dim_model;
    float local_sqr_sum = 0;
    float abs_max = 0;
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        float v = input[offset + i];
        local_sqr_sum += v * v;

        float v_w = v * float(weight[i]); // multiple weight here for quant
        smem[i] = v_w;
        float v_abs = v_w < 0 ? -v_w : v_w;
        abs_max = v_abs > abs_max ? v_abs : abs_max;
    }
    local_sqr_sum = functions::blockReduceSum<float>(local_sqr_sum);
    float reciprocal_sqrt = rsqrtf(local_sqr_sum / (float) dim_model + eps);

    abs_max = functions::blockReduceMax<T>(abs_max);
    //    if (threadIdx.x == 0) {
    //        printf("blockIdx.x=%d, reciprocal_sqrt=%.3f, abs_max=%.3f\n", blockIdx.x,
    //        reciprocal_sqrt, abs_max);
    //    }
    float block_scale = 127.0 / abs_max;
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        float v = smem[i] / scale;
        output[offset + i] = T(v * reciprocal_sqrt);
        out_int8[offset + i] = int8_t(nearbyintf(v * block_scale));
    }
    if (threadIdx.x == 0) {
        out_scale[blockIdx.z * gridDim.x + blockIdx.x] = T_SCALE(abs_max * reciprocal_sqrt / 127.);
        //        out_scale[blockIdx.z * gridDim.x + blockIdx.x] = T_SCALE(127. / abs_max /
        //        reciprocal_sqrt);
    }
}

void layernorm_quant(
    const core::Context& ctx,
    const core::Tensor& input,  // (batch, seq_len, dim_model)
    const core::Tensor& weight, // (dim_model)
    core::Tensor* output,       // (batch, seq_len, dim_model)
    core::Tensor* output_int8,  // (batch, seq_len, dim_model)
    core::Tensor* output_scale, // (batch, seq_len)
    float eps,
    float scale) {
    int batch = (input.ndim() == 2) ? 1 : input.size(0);
    int input_stride = (input.ndim() == 2) ? 0 : input.stride(0);
    int seq_len = input.size(-2);
    int dim_model = input.size(-1);

    if (output->shape() != input.shape()) {
        *output = ctx.tensor(input.size(), input.dtype());
    }
    if (output_int8->shape() != input.shape()) {
        // round_up tensor size
        *output_int8 = ctx.tensor(input.size(), DataType::kInt8, "", 32 * dim_model);
    }
    auto scale_shape = get_scale_shape(input);
    if (output_scale->shape() != scale_shape) {
        *output_scale = ctx.tensor(scale_shape, DataType::kFloat);
    }

    dim3 gridDim(seq_len, 1, batch);
    dim3 blockDim(round_up_thread(dim_model));
    cudaStream_t stream = ctx.current_stream()->ptr;

    if (output_scale->dtype() == core::DataType::kHalf) {
        BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
            BM_KERNEL(fuse_layernorm_rms_quant)<scalar_t, half>
                <<<gridDim, blockDim, dim_model * sizeof(float), stream>>>(
                    dim_model,
                    eps,
                    scale,
                    input_stride,
                    weight.data<scalar_t>(),
                    input.data<scalar_t>(),
                    output->mutable_data<scalar_t>(),
                    output_int8->mutable_data<int8_t>(),
                    output_scale->mutable_data<half>());
        });
    } else if (output_scale->dtype() == core::DataType::kBFloat16) {
        BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
            BM_KERNEL(fuse_layernorm_rms_quant)<scalar_t, nv_bfloat16>
                <<<gridDim, blockDim, dim_model * sizeof(float), stream>>>(
                    dim_model,
                    eps,
                    scale,
                    input_stride,
                    weight.data<scalar_t>(),
                    input.data<scalar_t>(),
                    output->mutable_data<scalar_t>(),
                    output_int8->mutable_data<int8_t>(),
                    output_scale->mutable_data<nv_bfloat16>());
        });
    } else {
        BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
            BM_KERNEL(fuse_layernorm_rms_quant)<scalar_t, float>
                <<<gridDim, blockDim, dim_model * sizeof(float), stream>>>(
                    dim_model,
                    eps,
                    scale,
                    input_stride,
                    weight.data<scalar_t>(),
                    input.data<scalar_t>(),
                    output->mutable_data<scalar_t>(),
                    output_int8->mutable_data<int8_t>(),
                    output_scale->mutable_data<float>());
        });
    }
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim(N / 1024, M)
// blockDim(1024)
template<typename T, typename T_Y=T>
static __global__ void KERNEL_quant_scale_back(
    int N,
    const int32_t* __restrict__ inp,   // (M, N)
    const float* __restrict__ scale_x, // (M,)
    const T_Y* __restrict__ scale_y,   // (N,)
    T* __restrict__ out) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < N) {
        int pos = r * N + c;
        //        printf("pos=%d, inp[pos]=%.3f, scale_x[r]=%.5f, scale_y[c]=%.5f\n", pos,
        //        float(inp[pos]), float((scale_x[r])), float(scale_y[c]));
        out[pos] = T(float(inp[pos]) * float(scale_x[r]) * float(scale_y[c]));
    }
}

core::Tensor quant_scale_back(
    const core::Context& ctx,
    const core::Tensor& input,   // (M, N)
    const core::Tensor* scale_x, // (M,)
    const core::Tensor* scale_y, // (N,)
    core::DataType out_type,
    core::Tensor* output
) {
    BM_ASSERT_EQ(input.dtype(), DataType::kInt32, "Wrong input dtype");
    BM_ASSERT_EQ(scale_x->dtype(), DataType::kFloat, "Wrong scale_x dtype");
    if (scale_y->dtype() != DataType::kFloat) {
        out_type = scale_y->dtype();
    }
    if (output) {
        BM_ASSERT_EQ(output->size(), input.size(), "Wrong output size()");
        BM_ASSERT_EQ(output->dtype(), out_type, "Wrong output dtype");
    }

    size_t n = input.size(-1);
    size_t m = input.numel() / n;

    BM_ASSERT_EQ(scale_x->numel(), m, "Wrong scale_x numel");
    BM_ASSERT_EQ(scale_y->numel(), n, "Wrong scale_y numel");

    int threads = round_up_thread(n);
    dim3 gridDim(round_up(n, threads) / threads, m);
    dim3 blockDim(threads);

    cudaStream_t stream = ctx.current_stream()->ptr;
    if (ctx.debug() >= 4) {
        std::cout << "m=" << m << ", n/threads=" << gridDim.x << ", threads=" << threads
                  << std::endl;
    }

    core::Tensor ret = output ? *output : ctx.tensor(input.size(), scale_y->dtype());
    BM_DTYPE_DISPATCH_FLOAT(out_type, {
        if (scale_y->dtype() == DataType::kFloat) {
            KERNEL_quant_scale_back<scalar_t, float><<<gridDim, blockDim, 0, stream>>>(
                n,
                input.data<int32_t>(),
                scale_x->data<float>(),
                scale_y->data<float>(),
                ret.mutable_data<scalar_t>());
        } else {
            KERNEL_quant_scale_back<scalar_t><<<gridDim, blockDim, 0, stream>>>(
                n,
                input.data<int32_t>(),
                scale_x->data<float>(),
                scale_y->data<scalar_t>(),
                ret.mutable_data<scalar_t>());
        }
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    if (output) {
        BM_ASSERT_EQ(ret.data<char>(), output->data<char>(), "");
    }
    return ret;
}

// gridDim(N / 1024, M)
// blockDim(1024)
template<typename T>
static __global__ void BM_KERNEL(quant_scale_back3)(
    int N,
    int dim_q,
    int dim_kv,
    const int32_t* __restrict__ inp,   // (M, N)
    const float* __restrict__ scale_x, // (M,)
    const T* __restrict__ scale_y,     // (N,)
    T* q,
    T* k,
    T* v) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < N) {
        int pos = r * N + c;
        T value = T(float(inp[pos]) * float(scale_x[r]) * float(scale_y[c]));
        if (c < dim_q) {
            q[r * dim_q + c] = value;
        } else if (c < dim_q + dim_kv) {
            k[r * dim_kv + c - dim_q] = value;
        } else {
            v[r * dim_kv + c - dim_q - dim_kv] = value;
        }
    }
}

void quant_scale_back3(
    const core::Context& ctx,
    const core::Tensor& input,   // (M, N)
    const core::Tensor* scale_x, // (M,)
    const core::Tensor* scale_y,  // (N,)
    int dim_q,
    int dim_kv,
    core::Tensor* q,
    core::Tensor* k,
    core::Tensor* v
) {
    BM_ASSERT_EQ(input.dtype(), DataType::kInt32, "Wrong input dtype");
    BM_ASSERT_EQ(scale_x->dtype(), DataType::kFloat, "Wrong scale_x dtype");
    BM_ASSERT_EQ(input.size(-1), dim_q + dim_kv * 2, "Wrong size");

    size_t n = input.size(-1);
    size_t m = input.numel() / n;

    BM_ASSERT_EQ(scale_x->numel(), m, "Wrong scale_x numel");
    BM_ASSERT_EQ(scale_y->numel(), n, "Wrong scale_y numel");

    int threads = round_up_thread(n);
    dim3 gridDim(round_up(n, threads) / threads, m);
    dim3 blockDim(threads);

    cudaStream_t stream = ctx.current_stream()->ptr;
    if (ctx.debug() >= 3) {
        std::cout << "m=" << m << ", n/threads=" << gridDim.x << ", threads=" << threads
                  << std::endl;
    }

    auto shape = input.shape();
    shape[shape.size() - 1] = dim_q;
    *q = ctx.tensor(shape, scale_y->dtype());
    shape[shape.size() - 1] = dim_kv;
    *k = ctx.tensor(shape, scale_y->dtype());
    *v = ctx.tensor(shape, scale_y->dtype());

    BM_DTYPE_DISPATCH_FLOAT(scale_y->dtype(), {
        BM_KERNEL(quant_scale_back3)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n, dim_q, dim_kv,
            input.data<int32_t>(),
            scale_x->data<float>(),
            scale_y->data<scalar_t>(),
            q->mutable_data<scalar_t>(),
            k->mutable_data<scalar_t>(),
            v->mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim (batch, len_kv, num_heads),  blockDim (dim_head, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(quant_back_copy_to_buffer)(
    int len_buf,
    int num_heads,
    int dim_head,
    int src_stride,
    int dst_stride,
    int place_stride,
    const int32_t* __restrict__ placement, // (batch, len_kv)
    const int32_t* __restrict__ src,       // (batch, len_kv, num_heads, dim_head)
    const float* __restrict__ scale_x,     // (batch, len_kv,)
    const T* __restrict__ scale_y,         // (num_heads * dim_head)
    T* __restrict__ dst                    // (batch, num_heads, len_buf, dim_head)
) {
    int batch_id = blockIdx.x;
    int pos_buf =
        (placement == nullptr) ? blockIdx.y : placement[batch_id * place_stride + blockIdx.y];
    // padded query when pos == -1, just ignore.
    int offset_src =
        batch_id * src_stride + (blockIdx.y * num_heads + blockIdx.z) * dim_head + threadIdx.x;
    int offset_dst =
        batch_id * dst_stride + (blockIdx.z * len_buf + pos_buf) * dim_head + threadIdx.x;
    if (pos_buf >= 0 && threadIdx.x < dim_head) {
        float x = float(scale_x[batch_id * gridDim.y + blockIdx.y]);
        float y = float(scale_y[blockIdx.z * dim_head + threadIdx.x]);
        dst[offset_dst] = T(float(src[offset_src]) * x * y);
    }
}

void quant_back_copy_to_buffer(
    const core::Context& ctx,
    int num_heads,
    int len_kv,
    int len_buf,
    int dim_head,
    const core::Tensor* placement,
    const core::Tensor& src,
    const core::Tensor* scale_x,
    const core::Tensor* scale_y,
    const core::Tensor& dst) {
    cudaStream_t stream = ctx.current_stream()->ptr;
    BM_ASSERT(scale_x != nullptr, "scale_x is null");
    BM_ASSERT(scale_y != nullptr, "scale_y is null");
    BM_ASSERT(scale_x->numel() > 0, "scale_x is empty");
    BM_ASSERT(scale_y->numel() > 0, "scale_y is empty");

    int batch = (src.ndim() == 3) ? 1 : src.size(0);
    int src_stride = (src.ndim() == 3) ? 0 : src.stride(0);
    int dst_stride = (dst.ndim() == 3) ? 0 : dst.stride(0);
    int place_stride = (placement == nullptr || placement->ndim() == 1) ? 0 : placement->stride(0);
    dim3 gridDim(batch, len_kv, num_heads);
    BM_ASSERT(dim_head < 1024, "Too big dim_head");
    dim3 blockDim(round_up(dim_head, 32));

    auto dtype = dst.dtype();
    BM_ASSERT_EQ(src.dtype(), DataType::kInt32, "src is not a quanted tensor");
    BM_ASSERT(
        (src.ndim() == 3 && dst.ndim() == 3) || (src.ndim() == 4 && dst.ndim() == 4),
        "src and dst must be 3/4-dimensional");

    BM_ASSERT_EQ(src.size(-3), len_kv, "dim mismatch");
    BM_ASSERT_EQ(src.size(-2), num_heads, "dim mismatch");
    BM_ASSERT_EQ(src.size(-1), dim_head, "dim mismatch");

    BM_ASSERT_EQ(dst.size(-3), num_heads, "dim mismatch");
    BM_ASSERT_EQ(dst.size(-2), len_buf, "dim mismatch");
    BM_ASSERT_EQ(dst.size(-1), dim_head, "dim mismatch");

    BM_DTYPE_DISPATCH_FLOAT(dtype, {
        BM_KERNEL(quant_back_copy_to_buffer)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            len_buf,
            num_heads,
            dim_head,
            src_stride,
            dst_stride,
            place_stride,
            (placement == nullptr ? nullptr : placement->data<int32_t>()),
            src.data<int32_t>(),
            scale_x->data<float>(),
            scale_y->data<scalar_t>(),
            dst.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim (batch_size, len_q, num_heads),  blockDim (dim_head)
template<typename T>
static __global__ void BM_KERNEL(quant_back_transpose)(
    const int32_t* __restrict__ input, // (batch, len_q, num_heads, dim_head)
    const float* __restrict__ scale_x, // (batch, len_q)
    const T* __restrict__ scale_y,     // (num_heads * dim_head)
    T* __restrict__ output             // (batch, num_heads, len_q, dim_head)
) {
    unsigned int dim_head = blockDim.x;
    size_t offset_src = ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * dim_head;
    size_t offset_dst = ((blockIdx.x * gridDim.z + blockIdx.z) * gridDim.y + blockIdx.y) * dim_head;

    float x = float(scale_x[blockIdx.x * gridDim.y + blockIdx.y]);
    float y = float(scale_y[blockIdx.z * dim_head + threadIdx.x]);

    output[offset_dst + threadIdx.x] = T(float(input[offset_src + threadIdx.x]) * x * y);
}

core::Tensor quant_back_transpose(
    const core::Context& ctx,
    const bmengine::core::Tensor& h_q, // (batch, len_q, num_heads, dim_head)
    const core::Tensor* scale_x, // (batch, len_q)
    const core::Tensor* scale_y) { // (num_heads * dim_head)
    BM_ASSERT_EQ(h_q.ndim(), 4, "input is not 4d")
    BM_ASSERT_EQ(h_q.dtype(), core::DataType::kInt32, "input is not int32")
    BM_ASSERT(scale_x != nullptr, "scale_x is null");
    BM_ASSERT(scale_y != nullptr, "scale_y is null");
    cudaStream_t stream = ctx.current_stream()->ptr;
    size_t batch_size = h_q.size(0);
    size_t len_q = h_q.size(1);
    size_t num_heads = h_q.size(2);
    size_t dim_head = h_q.size(3);

    BM_ASSERT_EQ(scale_x->numel(), batch_size * len_q, "scale_x is wrong size");
    BM_ASSERT_EQ(scale_y->numel(), num_heads * dim_head, "scale_y is wrong size");

    dim3 gridDim(batch_size, len_q, num_heads);
    dim3 blockDim(dim_head);

    auto shape = h_q.shape(); // (batch, num_heads, len_q, dim_head)
    shape[1] = num_heads;
    shape[2] = len_q;
    core::Tensor output = ctx.tensor(shape, scale_y->dtype());
    BM_DTYPE_DISPATCH_FLOAT(output.dtype(), {
        BM_KERNEL(quant_back_transpose)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            h_q.data<int32_t>(),
            scale_x->data<float>(),
            scale_y->data<scalar_t>(),
            output.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return output;
}

// gridDim(N / 1024, M, 1)
// blockDim(1024)
template<typename T>
static __global__ void BM_KERNEL(quant_back_element_add_scale)(
    int N,
    const int32_t* __restrict__ a,     // (M, N)
    const float* __restrict__ scale_x, // (M,)
    const T* __restrict__ scale_y,     // (N,)
    const T* __restrict__ b,           // (M, N)
    float scale,
    T* __restrict__ out) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < N) {
        int pos = r * N + c;
        float quant_back = float(a[pos]) * float(scale_x[r]) * float(scale_y[c]);
        out[pos] = T((quant_back + float(b[pos])) * scale);
    }
}

core::Tensor quant_back_element_add_scale(
    const core::Context& ctx,
    const core::Tensor& input,   // (M, N)
    const core::Tensor* scale_x, // (M,)
    const core::Tensor* scale_y, // (N,)
    const core::Tensor& input_b,
    float scale) {
    BM_ASSERT_EQ(input.dtype(), core::DataType::kInt32, "Wrong input dtype");

    size_t n = input.size(-1);
    size_t m = input.numel() / n;

    int threads = round_up_thread(n);
    dim3 gridDim(round_up(n, threads) / threads, m);
    dim3 blockDim(threads);

    cudaStream_t stream = ctx.current_stream()->ptr;

    core::Tensor ret = ctx.tensor(input.size(), input_b.dtype());
    BM_DTYPE_DISPATCH_FLOAT(input_b.dtype(), {
        BM_KERNEL(quant_back_element_add_scale)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n,
            input.data<int32_t>(),
            scale_x->data<float>(),
            scale_y->data<scalar_t>(),
            input_b.data<scalar_t>(),
            scale,
            ret.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return ret;
}

struct Act_gelu {
    static inline __device__ float apply(float x) { return nn::gelu(x); }
};

struct Act_silu {
    static inline __device__ float apply(float x) { return nn::silu(x); }
};

// gridDim(N / 1024, M, 1) blockDim(1024)
template<typename T, typename GATGE_ACT>
static __global__ void BM_KERNEL(quant_back_act_mul)(
    const int N,
    const int32_t* __restrict__ A,       // (M, N)
    const float* __restrict__ a_scale_x, // (M,)
    const T* __restrict__ a_scale_y,     // (N,)
    const int32_t* __restrict__ B,       // (M, N)
    const float* __restrict__ b_scale_x, // (M,)
    const T* __restrict__ b_scale_y,     // (N,)
    T* __restrict__ out) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < N) {
        int pos = r * N + c;
        float a_back = float(A[pos]) * float(a_scale_x[r]) * float(a_scale_y[c]);
        float b_back = float(B[pos]) * float(b_scale_x[r]) * float(b_scale_y[c]);
        float gate = GATGE_ACT::apply(a_back);
        out[pos] = T(b_back * gate);
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            //        printf("pos=%d, inp[pos]=%.f, scale_x[r]=%.5f, scale_y[c]=%.5f\n", pos,
            //        float(A[pos]), float((a_scale_x[r])), float(a_scale_y[c])); printf("pos=%d,
            //        inp[pos]=%.f, scale_x[r]=%.5f, scale_y[c]=%.5f, b_back=%.3f, gate=%.3f\n",
            //        pos, float(B[pos]), float((b_scale_x[r])), float(b_scale_y[c]), b_back, gate);
        }
    }
}

core::Tensor quant_back_act_mul(
    const core::Context& ctx,
    const core::Tensor& A,         // (M, N)
    const core::Tensor* a_scale_x, // (M,)
    const core::Tensor* a_scale_y, // (N,)
    const core::Tensor& B,         // (M, N)
    const core::Tensor* b_scale_x, // (M,)
    const core::Tensor* b_scale_y, // (N,)
    const std::string& act_type) {
    const size_t N = A.size(-1);
    const size_t M = A.numel() / N;

    BM_ASSERT(A.dtype() == core::DataType::kInt32, "Wrong input A dtype");
    BM_ASSERT(B.dtype() == core::DataType::kInt32, "Wrong input B dtype");
    BM_ASSERT(A.shape() == B.shape(), "Wrong input shape");
    BM_ASSERT(a_scale_x->shape() == b_scale_x->shape(), "Wrong scale shape");
    BM_ASSERT(a_scale_y->shape() == b_scale_y->shape(), "Wrong scale shape");
    BM_ASSERT(a_scale_x->numel() == M, "Wrong scale shape");
    BM_ASSERT(a_scale_y->numel() == N, "Wrong scale shape");

    int threads = round_up_thread(N);

    dim3 gridDim(round_up(N, threads) / threads, M);
    dim3 blockDim(threads);
    cudaStream_t stream = ctx.current_stream()->ptr;
    if (ctx.debug() >= 3) {
        std::cout << "M=" << M << ", N/threads=" << gridDim.x << ", threads=" << threads
                  << std::endl;
    }

    core::Tensor ret = ctx.tensor(A.shape(), a_scale_y->dtype());

    if (act_type == "gelu") {
        BM_DTYPE_DISPATCH_FLOAT(a_scale_y->dtype(), {
            BM_KERNEL(quant_back_act_mul)<scalar_t, Act_gelu><<<gridDim, blockDim, 0, stream>>>(
                N,
                A.data<int32_t>(),
                a_scale_x->data<float>(),
                a_scale_y->data<scalar_t>(),
                B.data<int32_t>(),
                b_scale_x->data<float>(),
                b_scale_y->data<scalar_t>(),
                ret.mutable_data<scalar_t>());
        });
    } else if (act_type == "silu") {
        BM_DTYPE_DISPATCH_FLOAT(a_scale_y->dtype(), {
            BM_KERNEL(quant_back_act_mul)<scalar_t, Act_silu><<<gridDim, blockDim, 0, stream>>>(
                N,
                A.data<int32_t>(),
                a_scale_x->data<float>(),
                a_scale_y->data<scalar_t>(),
                B.data<int32_t>(),
                b_scale_x->data<float>(),
                b_scale_y->data<scalar_t>(),
                ret.mutable_data<scalar_t>());
        });
    } else {
        throw std::runtime_error("No activation");
    }
    BM_CUDART_ASSERT(cudaGetLastError());
    return ret;
}

} // namespace nn
