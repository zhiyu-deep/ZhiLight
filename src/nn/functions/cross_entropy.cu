#include "nn/nn.h"
#include <bmengine/functions/reduce.cuh>
#include "bmengine/functions/index_select.h"

namespace nn {

// gridDim (batch, seq_len, 1),   blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(cross_entropy_2inp)(
    size_t n,
    size_t len_vocab,
    size_t len_ext,
    const T* __restrict__ logits,       // (batch, seq_len, len_vocab)
    const T* __restrict__ logits_ext,   // (seq_len, len_ext)
    const int32_t* __restrict__ labels, // (batch, seq_len)
    int32_t ignore_index,
    T* __restrict__ output,             // (batch, seq_len, n)
    float* __restrict__ loss            // (batch, seq_len)

) {
    size_t offset_n = (blockIdx.x * gridDim.y * n) + blockIdx.y * n;
    size_t offset_vocab = (blockIdx.x * gridDim.y * len_vocab) + blockIdx.y * len_vocab;
    size_t offset_ext = blockIdx.y * len_ext - len_vocab;
    size_t offset_label = (blockIdx.x * gridDim.y) + blockIdx.y;
    float local_max = -1e20;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val;
        if (i < len_vocab)
            val = logits[i + offset_vocab];
        else
            val = logits_ext[i + offset_ext];

        local_max = fmaxf(local_max, val);
    }
    local_max = functions::blockReduceMax<float>(local_max);
    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val;
        if (i < len_vocab)
            val = logits[i + offset_vocab];
        else
            val = logits_ext[i + offset_ext];

        local_sum += expf((float) val - local_max);
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val;
        if (i < len_vocab)
            val = logits[i + offset_vocab];
        else
            val = logits_ext[i + offset_ext];
        output[i + offset_n] = expf((float) val - local_max) / local_sum;
    }

    if (threadIdx.x == 0) {
        int32_t target = labels[offset_label];
        if (target != ignore_index) {
            T val;
            if (target < len_vocab)
                val = logits[target + offset_vocab];
            else
                val = logits_ext[target + offset_ext];
            loss[offset_label] = -(float) val + local_max + logf(local_sum);
        } else {
            loss[offset_label] = 0;
        }
    }
}

// gridDim (seq_len, 1, 1),   blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(greedy_matching)(
    size_t n,
    size_t len_vocab,
    size_t len_ext,
    const T* __restrict__ logits,       // (seq_len, len_vocab)
    const T* __restrict__ logits_ext,   // (seq_len, len_ext)
    const int32_t* __restrict__ labels, // (seq_len)
    int32_t ignore_index,
    int32_t* __restrict__ match         // (seq_len)
) {
    __shared__ float global_max;

    size_t offset_vocab = blockIdx.x * len_vocab;
    size_t offset_ext = blockIdx.x * len_ext - len_vocab;

    float local_max = -1e20;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val;
        if (i < len_vocab)
            val = logits[i + offset_vocab];
        else
            val = logits_ext[i + offset_ext];

        local_max = fmaxf(local_max, val);
    }
    local_max = functions::blockReduceMax<float>(local_max);
    if (threadIdx.x == 0) {
        global_max = local_max;
    }
    __syncthreads();
    local_max = global_max;

    int max_id = -1;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val;
        if (i < len_vocab)
            val = logits[i + offset_vocab];
        else
            val = logits_ext[i + offset_ext];

        if (float(val) == local_max)
            max_id = i;
    }
    max_id = functions::blockReduceMax<int32_t>(max_id);

    if (threadIdx.x == 0) {
        int32_t target = labels[blockIdx.x];
        if (target != ignore_index && target != 7) {
            match[blockIdx.x] = (target == max_id);
        } else {
            match[blockIdx.x] = 1;
        }
    }
}

// gridDim(1, 1, 1)     blockDim(1024, 1, 1)
static __global__ void BM_KERNEL(reduce_loss)(
    size_t seq_len,
    int32_t ignore_index,
    const float* __restrict__ loss,     // (seq_len)
    const int32_t* __restrict__ labels, // (seq_len)
    float* __restrict__ output          // (2)
) {
    float local_sum = 0;
    int32_t num_valid = 0;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        local_sum += loss[i];
        if (labels[i] != ignore_index)
            num_valid++;
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    num_valid = functions::blockReduceSum<int32_t>(num_valid);
    if (threadIdx.x == 0) {
        output[0] = local_sum / num_valid;
        output[1] = (float) num_valid;
    }
}

// gridDim(1, 1, 1)     blockDim(1024, 1, 1)
static __global__ void BM_KERNEL(reduce_min)(
    size_t seq_len,
    int32_t ignore_index,
    const int32_t* __restrict__ match,  // (seq_len)
    const int32_t* __restrict__ labels, // (seq_len)
    int32_t* __restrict__ output        // (1)
) {
    int32_t local_min = 1;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        local_min = match[i] < local_min
                      ? match[i]
                      : local_min; // if (labels[i] == ignore_index) assert match[i]==1;
    }
    local_min = -functions::blockReduceMax<int32_t>(-local_min); // Min
    if (threadIdx.x == 0) {
        output[0] = local_min;
    }
}

// gridDim(batch, 1, 1)     blockDim(1024, 1, 1)
static __global__ void BM_KERNEL(reduce_sum)(
    size_t seq_len,
    int32_t ignore_index,
    const float* __restrict__ loss,     // (batch, seq_len)
    const int32_t* __restrict__ labels, // (batch, seq_len)
    float* __restrict__ output          // (batch)
) {
    size_t seq_offset = blockIdx.x * seq_len;
    float local_sum = 0;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        local_sum += loss[i + seq_offset];
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = local_sum;
    }
}

// gridDim(n / 1024, 1, 1)      blockDim(1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(cross_entropy_backward)(
    size_t max_n,
    size_t len_vocab,
    size_t len_ext,
    float loss_scale,
    const T* __restrict__ probs,        // (..., len_vocab + len_ext)
    const int32_t* __restrict__ labels, // (...)
    int32_t ignore_index,
    T* __restrict__ output,             // (..., len_vocab)
    T* __restrict__ output_ext          // (..., len_ext)
) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < max_n) {
        size_t p = len_vocab + len_ext;
        size_t seq_pos = pos / p;
        size_t voc_pos = pos % p;

        int32_t target = labels[seq_pos];

        if (target == ignore_index) {
            if (voc_pos < len_vocab)
                output[seq_pos * len_vocab + voc_pos] = 0.;
            else
                output_ext[seq_pos * len_ext + (voc_pos - len_vocab)] = 0.;
        } else {
            if (target == voc_pos) {
                if (voc_pos < len_vocab)
                    output[seq_pos * len_vocab + voc_pos] =
                        (float) (probs[pos] - T(1.)) * loss_scale;
                else
                    output_ext[seq_pos * len_ext + (voc_pos - len_vocab)] =
                        (float) (probs[pos] - T(1.)) * loss_scale;
            } else {
                if (voc_pos < len_vocab)
                    output[seq_pos * len_vocab + voc_pos] = (float) probs[pos] * loss_scale;
                else
                    output_ext[seq_pos * len_ext + (voc_pos - len_vocab)] =
                        (float) probs[pos] * loss_scale;
            }
        }
    }
}

static __host__ std::tuple<float, core::Tensor, core::Tensor> cross_entropy_2inp(
    const core::Context& ctx,
    const core::Tensor& logits,     // (batch, seq_len, len_vocab)
    const core::Tensor& logits_ext, // (seq_len, len_ext)
    const core::Tensor& labels,     // (batch, seq_len)
    int32_t ignore_index,
    float loss_scale) {
    bool has_ext = logits_ext.numel() > 0;
    size_t batch = logits.ndim() < 3 ? 1 : logits.size(0);
    size_t seq_len = logits.size(-2);
    size_t len_vocab = logits.size(-1);
    size_t len_ext = has_ext ? logits_ext.size(1) : 0;
    size_t n = len_vocab + len_ext;

    int threads = min(round_up(n, 32), (size_t) 1024);
    dim3 gridDim(batch, seq_len, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    core::Tensor probs = ctx.tensor({ batch, seq_len, n }, logits.dtype());
    core::Tensor loss_array = ctx.tensor({ batch, seq_len }, core::DataType::kFloat);
    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(cross_entropy_2inp)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n,
            len_vocab,
            len_ext,
            logits.data<scalar_t>(),
            has_ext ? logits_ext.data<scalar_t>() : nullptr,
            labels.data<int32_t>(),
            ignore_index,
            probs.data<scalar_t>(),
            loss_array.data<float>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());

    core::Tensor loss = ctx.tensor({ 2 }, core::DataType::kFloat);
    gridDim = dim3(1, 1, 1);
    blockDim = dim3(threads, 1, 1);
    BM_KERNEL(reduce_loss)<<<gridDim, blockDim, 0, stream>>>(
        seq_len,
        ignore_index,
        loss_array.data<float>(),
        labels.data<int32_t>(),
        loss.data<float>());
    float tmp_array[2];
    loss.to_buffer(tmp_array);
    float ret_loss = tmp_array[0];
    float scale = loss_scale / tmp_array[1];

    // cross_entropy_backward
    core::Tensor grad_emb = ctx.tensor({ seq_len, len_vocab }, logits.dtype());
    core::Tensor grad_ext = core::Tensor();
    if (has_ext) {
        grad_ext = ctx.tensor({ seq_len, len_ext }, logits.dtype());
    }
    size_t numel = probs.numel();
    threads = min(round_up(numel, 32), (size_t) 1024);
    gridDim = dim3(round_up(numel, threads) / threads, 1, 1);
    blockDim = dim3(threads, 1, 1);

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(cross_entropy_backward)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            numel,
            len_vocab,
            len_ext,
            scale,
            probs.data<scalar_t>(),
            labels.data<int32_t>(),
            ignore_index,
            grad_emb.data<scalar_t>(),
            has_ext ? grad_ext.data<scalar_t>() : nullptr);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return std::make_tuple(ret_loss, grad_emb, grad_ext);
}

static __host__ int greedy_matching(
    const core::Context& ctx,
    const core::Tensor& logits,     // (seq_len, len_vocab)
    const core::Tensor& logits_ext, // (seq_len, len_ext)
    const core::Tensor& labels,     // (seq_len)
    int32_t ignore_index) {
    bool has_ext = logits_ext.numel() > 0;
    size_t seq_len = logits.size(0);
    size_t len_vocab = logits.size(1);
    size_t len_ext = has_ext ? logits_ext.size(1) : 0;
    size_t n = len_vocab + len_ext;

    int threads = min(round_up(n, 32), (size_t) 1024);
    dim3 gridDim(seq_len, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    core::Tensor match_array = ctx.tensor({ seq_len }, core::DataType::kInt32);
    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(greedy_matching)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n,
            len_vocab,
            len_ext,
            logits.data<scalar_t>(),
            has_ext ? logits_ext.data<scalar_t>() : nullptr,
            labels.data<int32_t>(),
            ignore_index,
            match_array.data<int32_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());

    core::Tensor greedy_match = ctx.tensor({ 1 }, core::DataType::kInt32);
    gridDim = dim3(1, 1, 1);
    blockDim = dim3(threads, 1, 1);
    BM_KERNEL(reduce_min)<<<gridDim, blockDim, 0, stream>>>(
        seq_len,
        ignore_index,
        match_array.data<int32_t>(),
        labels.data<int32_t>(),
        greedy_match.data<int32_t>());
    int tmp_array[1];
    greedy_match.to_buffer(tmp_array);
    int res = tmp_array[0];

    return res;
}

static __host__ std::tuple<core::Tensor, core::Tensor> log_probability(
    const core::Context& ctx,
    const core::Tensor& logits,     // (seq_len, len_vocab)
    const core::Tensor& logits_ext, // (seq_len, len_ext)
    const core::Tensor& labels,     // (seq_len)
    int32_t ignore_index) {
    bool has_ext = logits_ext.numel() > 0;
    size_t batch = logits.ndim() < 3 ? 1 : logits.size(0);
    size_t seq_len = logits.size(-2);
    size_t len_vocab = logits.size(-1);
    size_t len_ext = has_ext ? logits_ext.size(1) : 0;
    size_t n = len_vocab + len_ext;

    int threads = min(round_up(n, 32), (size_t) 1024);
    dim3 gridDim(batch, seq_len, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    core::Tensor probs = ctx.tensor({ batch, seq_len, n }, logits.dtype(), "probs");
    core::Tensor loss_array = ctx.tensor({ batch, seq_len }, core::DataType::kFloat, "loss_array");
    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(cross_entropy_2inp)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n,
            len_vocab,
            len_ext,
            logits.data<scalar_t>(),
            has_ext ? logits_ext.data<scalar_t>() : nullptr,
            labels.data<int32_t>(),
            ignore_index,
            probs.data<scalar_t>(),
            loss_array.data<float>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());

    core::Tensor log_prob = ctx.tensor({ batch }, core::DataType::kFloat);
    gridDim = dim3(batch, 1, 1);
    blockDim = dim3(threads, 1, 1);
    BM_KERNEL(reduce_sum)<<<gridDim, blockDim, 0, stream>>>(
        seq_len,
        ignore_index,
        loss_array.data<float>(),
        labels.data<int32_t>(),
        log_prob.data<float>());

    return std::make_tuple(log_prob, loss_array);
}

std::tuple<float, core::Tensor, core::Tensor> cross_entropy(
    const core::Context& ctx,
    const std::tuple<core::Tensor, core::Tensor>& logits_tuple,
    const core::Tensor& labels,
    int32_t ignore_index,
    float loss_scale) {
    auto& logits = std::get<0>(logits_tuple);
    auto& logits_ext = std::get<1>(logits_tuple);

    return cross_entropy_2inp(ctx, logits, logits_ext, labels, ignore_index, loss_scale);
}

std::tuple<float, core::Tensor> cross_entropy_raw(
    const core::Context& ctx,
    const core::Tensor& logits,
    const core::Tensor& labels,
    int32_t ignore_index,
    float loss_scale) {
    auto ret = cross_entropy_2inp(ctx, logits, core::Tensor(), labels, ignore_index, loss_scale);
    return std::make_tuple(std::get<0>(ret), std::get<1>(ret));
}

int greedy_match(
    const core::Context& ctx,
    const std::tuple<core::Tensor, core::Tensor>& logits_tuple,
    const core::Tensor& labels,
    int32_t ignore_index) {
    auto& logits = std::get<0>(logits_tuple);
    auto& logits_ext = std::get<1>(logits_tuple);

    return greedy_matching(ctx, logits, logits_ext, labels, ignore_index);
}

int greedy_match_raw(
    const core::Context& ctx,
    const core::Tensor& logits,
    const core::Tensor& labels,
    int32_t ignore_index) {
    return greedy_matching(ctx, logits, core::Tensor(), labels, ignore_index);
}

std::tuple<core::Tensor, core::Tensor> log_prob(
    const core::Context& ctx,
    const std::tuple<core::Tensor, core::Tensor>& logits_tuple,
    const core::Tensor& labels,
    int32_t ignore_index) {
    auto& logits = std::get<0>(logits_tuple);
    auto& logits_ext = std::get<1>(logits_tuple);

    return log_probability(ctx, logits, logits_ext, labels, ignore_index);
}

std::tuple<float, core::Tensor> log_prob_raw(
    const core::Context& ctx,
    const core::Tensor& logits,
    const core::Tensor& labels,
    int32_t ignore_index) {
    auto out = log_probability(ctx, logits, core::Tensor(), labels, ignore_index);
    float tmp_array[1];
    std::get<0>(out).to_buffer(tmp_array);
    float res = tmp_array[0];
    return std::make_tuple(res, std::get<1>(out));
}

} // namespace nn
