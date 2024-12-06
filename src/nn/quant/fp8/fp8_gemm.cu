// Author: Gaojunmin@zhihu.com

#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include <bmengine/logger/std_log_op.hpp>
#include <assert.h>

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

#define CHUNK_K 2  // BLOCK_K / MMA_K

#define WARP_SIZE 32

#define THREAD_COPY_BYTES 16

// 一行64字节
#define CHUNK_LINE_BYTES 64          // CHUNK_K * MMA_K * sizeof(half)
// 一个 warp 一次可以拷8行。
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES
// 一行需要 4个 lane
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 64  // CHUNK_K * MMA_K

#define C_SMEM_STRIDE 136  // BLOCK_N + 8 of int

#define BLOCK_STRIDE 16

// 两行共 128 字节
#define SMEM_BANK_ROWS 2  // 32 * 4 / (AB_SMEM_STRIDE * sizeof(half))

#define PERMUTED_OFFSET 16  // bytes
#define PERMUTED_COLS 8

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define MMA16832_FP8(E4M3, D0, D1, D2, D3, A0, A1, A2, A3, B0, B1)                           \
    if constexpr(E4M3)                                                                       \
    asm volatile(                                                                            \
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, " \
        "{%8,%9}, {%10,%11,%12,%13};\n"                                                      \
        : "=f"(D0), "=f"(D1), "=f"(D2), "=f"(D3)                                             \
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1),                              \
          "f"(D0), "f"(D1), "f"(D2), "f"(D3));                                               \
    else                                                                                     \
    asm volatile(                                                                            \
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, " \
        "{%8,%9}, {%10,%11,%12,%13};\n"                                                      \
        : "=f"(D0), "=f"(D1), "=f"(D2), "=f"(D3)                                             \
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1),                              \
          "f"(D0), "f"(D1), "f"(D2), "f"(D3))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define FP8_E4M3 1
#define FP8_E5M2 1

namespace nn::fp8 {

#if CUDART_VERSION >= 12040
typedef int32_t OUT_T;
template<
    int FP8_TYPE=FP8_E4M3,
    uint32_t BLOCK_M = 256,
    uint32_t BLOCK_N = 128,
    uint32_t BLOCK_K = 32,
    uint32_t NUM_WARPS_M = 4,
    uint32_t NUM_WARPS_N = 2,
    int K_STAGE = 3
>
__global__ void KERNEL_mma_fp8(const int8_t *__restrict__ A,
                               const int8_t *__restrict__ B,
                               half *__restrict__ C,
                               uint32_t M, uint32_t N, uint32_t K,
                               float scale) {
#if __CUDA_ARCH__ >= 890
    static constexpr uint32_t WARP_M = BLOCK_M / NUM_WARPS_M;
    static constexpr uint32_t WARP_N = BLOCK_N / NUM_WARPS_N;
    static constexpr uint32_t NUM_WARPS = NUM_WARPS_M * NUM_WARPS_N;
    static constexpr uint32_t WARP_TILES_M = WARP_M / MMA_M;
    static constexpr uint32_t WARP_TILES_N = WARP_N / MMA_N;

    float RC[WARP_TILES_M][WARP_TILES_N][4];
    uint32_t RA[WARP_TILES_M][4];
    uint32_t RB[WARP_TILES_N][2];

    const uint32_t block_tile_i = (blockIdx.z % 2) ? gridDim.y - blockIdx.y - 1 : blockIdx.y;
    const uint32_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x);

    if (block_tile_i * BLOCK_M >= M || block_tile_j * BLOCK_N >= N) {
        return;
    }

    extern __shared__ __align__(16) int8_t smem[][AB_SMEM_STRIDE];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;

    static constexpr uint32_t B_smem_idx_off = BLOCK_M;
    static constexpr uint32_t smem_stage_off = BLOCK_M + BLOCK_N;

#pragma unroll
    for (int i = 0; i < WARP_TILES_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
            RC[i][j][2] = 0;
            RC[i][j][3] = 0;
        }
    }

    const int8_t *A_warp_ptr = &A[block_tile_i * BLOCK_M * K] + BLOCK_M / NUM_WARPS * K * warp_id;
    const int8_t *B_warp_ptr = &B[block_tile_j * BLOCK_N * K] + BLOCK_N / NUM_WARPS * K * warp_id;

    static constexpr int A_smem_iters = BLOCK_M / (CHUNK_COPY_LINES_PER_WARP * NUM_WARPS);
    static constexpr int B_smem_iters = BLOCK_N / (CHUNK_COPY_LINES_PER_WARP * NUM_WARPS);

    uint32_t smem_store_idx = 0;
    uint32_t smem_load_idx = 0;

    uint32_t smem_store_off = 0;
    uint32_t smem_load_off = 0;

    uint32_t shared_addr0 = __cvta_generic_to_shared(&smem[0][0]);

    const uint32_t row = lane_id / CHUNK_COPY_LINE_LANES;
    const uint32_t col = lane_id % CHUNK_COPY_LINE_LANES;
    const uint32_t col_perm = (col + row / SMEM_BANK_ROWS) % CHUNK_COPY_LINE_LANES;
    auto fn_copy_A = [&](const int8_t* A_ptr) {

        int4* A_lane_ptr = (int4 *)(A_ptr + row * K) + col;

        uint32_t A_smem_idx = smem_store_off + BLOCK_M / NUM_WARPS * warp_id;  // warp 从哪一行开始
        A_smem_idx += row; // lane 从哪一行开始

        uint32_t shared_addr = shared_addr0 + A_smem_idx * CHUNK_LINE_BYTES +
                               col_perm * THREAD_COPY_BYTES;

#pragma unroll
        for (int i = 0; i < A_smem_iters; ++i) {
            CP_ASYNC_CG(shared_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((int8_t *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            shared_addr += CHUNK_COPY_LINES_PER_WARP * AB_SMEM_STRIDE;
        }
    };

    auto fn_copy_B = [&](const int8_t* B_ptr) {
        int4* B_lane_ptr = (int4 *)(B_ptr + row * K) + col;
        uint32_t B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_N / NUM_WARPS * warp_id;
        B_smem_idx += row;

        uint32_t shared_addr = shared_addr0 + B_smem_idx * CHUNK_LINE_BYTES +
                               col_perm * THREAD_COPY_BYTES;
#pragma unroll
        for (int i = 0; i < B_smem_iters; ++i) {

            CP_ASYNC_CG(shared_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((int8_t *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            shared_addr += CHUNK_COPY_LINES_PER_WARP * AB_SMEM_STRIDE;
        }
    };

    auto fn_copy_AB = [&](uint32_t k_offset) {
        smem_store_idx = (smem_store_idx + 1) % K_STAGE;
        smem_store_off = smem_store_idx * smem_stage_off;

        fn_copy_A(A_warp_ptr + k_offset);
        fn_copy_B(B_warp_ptr + k_offset);

        CP_ASYNC_COMMIT_GROUP();
    };

    // Copy 1
    fn_copy_A(A_warp_ptr);
    fn_copy_B(B_warp_ptr);

    CP_ASYNC_COMMIT_GROUP();

    // Copy 2
    fn_copy_AB(CHUNK_K * MMA_K);

    // Copy 3
    if (K_STAGE > 3) {
        fn_copy_AB(2 * CHUNK_K * MMA_K);
    }
    if (K_STAGE > 4) {
        fn_copy_AB(3 * CHUNK_K * MMA_K);
    }
    if (K_STAGE > 5) {
        fn_copy_AB(4 * CHUNK_K * MMA_K);
    }

    const uint32_t col_perm_offset_A = (lane_id / 16) * 16 + (lane_id % 16 /*% (PERMUTED_COLS * SMEM_BANK_ROWS)*/)
                                                             / SMEM_BANK_ROWS * PERMUTED_OFFSET;
    auto fn_load_A = [&](uint32_t offset=0) {
        // ldmatrix 一次加载 m8n8 个 b16 ；即一次加载 8 X 16Bytes；需要八个地址。
        // 即 LOAD_K = 8; MMA_K = 16; 所以矩阵要拆成两半/4个。左上下，右上下。
        // --每四个线程(lane_id % 8) 加载 4 * 4 = 16 Bytes; 即对应 m8n8 中的一行。
        // 地址：左半个矩阵16行地址；右半16行地址; 共 32
        // permute 之前是(lane_id / 16) * 16
        uint32_t lane_dim2 = (offset + col_perm_offset_A) % AB_SMEM_STRIDE;
        uint32_t A_smem_idx = smem_load_off + (warp_id / NUM_WARPS_N) * WARP_M;
        uint32_t A_smem_lane_addr = shared_addr0 + (A_smem_idx + lane_id % 16) * CHUNK_LINE_BYTES + lane_dim2;
#pragma unroll
        for (uint32_t i = 0; i < WARP_TILES_M; ++i) {

//            if (i == 1 && warp_id == 0 && lane_id < 16) {
//                printf("Load T=%d, A_smem=%d\n", threadIdx.x, (A_smem_lane_addr - shared_addr0));
//            }

            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                        A_smem_lane_addr + i * MMA_M * AB_SMEM_STRIDE);
        }
    };

    const uint32_t col_perm_offset_B = ((lane_id / 8) % 2) * 16 +
                                       (lane_id /*% 8*/ % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET;

    auto fn_load_B = [&](uint32_t offset=0) {
        uint32_t lane_dim2 = (offset + col_perm_offset_B) % AB_SMEM_STRIDE;
        uint32_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % NUM_WARPS_N) * WARP_N;
        uint32_t B_smem_lane_addr = shared_addr0 + (B_smem_idx + lane_id % 8) * CHUNK_LINE_BYTES + lane_dim2;
#pragma unroll
        for (uint32_t j = 0; j < WARP_TILES_N; ++j) {
            LDMATRIX_X2(RB[j][0], RB[j][1], B_smem_lane_addr + j * MMA_N * AB_SMEM_STRIDE);
        }
    };

    auto fn_load_AB = [&](int offset=0) {
        fn_load_A(offset);
        fn_load_B(offset);
    };

    auto fn_mma = [&]() {
#pragma unroll
        for (int i = 0; i < WARP_TILES_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILES_N; ++j) {
                int j_s = (i % 2) ? (WARP_TILES_N - j - 1) : j;
                
                MMA16832_FP8((FP8_TYPE == FP8_E4M3),
                             RC[i][j_s][0], RC[i][j_s][1], RC[i][j_s][2], RC[i][j_s][3],
                             RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                             RB[j_s][0], RB[j_s][1]);
            }
        }
    };

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
    fn_load_AB(0);

    auto fn_compute_before = [&]() {
        fn_mma();
        fn_load_AB(MMA_K);
    };
    auto fn_compute_after = [&](uint32_t tile_k = 0) {
        fn_mma();
        fn_load_AB(0);
    };
    /* --------------------------------------------------------------------- */
    /* -------------------------- MAIN-LOOP over K ------------------------- */
    /* --------------------------------------------------------------------- */
    const uint32_t K_tiles = K / MMA_K;
    for (uint32_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < K_tiles; tile_k += CHUNK_K) {
        fn_compute_before();

        fn_copy_AB(tile_k * MMA_K);

        smem_load_idx = (smem_load_idx + 1) % K_STAGE;
        smem_load_off = smem_load_idx * smem_stage_off;

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();

        fn_compute_after(tile_k);
    } // End of MAIN-LOOP

    // same as MAIN-LOOP without copy
#pragma unroll
    for (int stage = K_STAGE - 3; stage >= 0; stage--) {
        fn_compute_before();

        smem_load_idx = (smem_load_idx + 1) % K_STAGE;
        smem_load_off = smem_load_idx * smem_stage_off;

        if (stage == 0) {
            CP_ASYNC_WAIT_GROUP(0);
        } else if (stage == 1) {
            CP_ASYNC_WAIT_GROUP(1);
        } else if (stage == 2) {
            CP_ASYNC_WAIT_GROUP(2);
        } else {
            CP_ASYNC_WAIT_GROUP(3);
        }
        __syncthreads();

        fn_compute_after();
    } // End

#pragma unroll
    for (int k_step = 1; k_step < CHUNK_K; ++k_step) {
        fn_compute_before();
    }

    fn_mma();
//    if (threadIdx.x < 8) {
//        printf("T=%d, X=%u\n", threadIdx.x, RC[0][0][0]);
//        printf("T=%d, X=%u\n", threadIdx.x, RC[0][0][1]);
//    }

    // Write output
    half *smem_warp_tile_row_ptr = (half*)&smem[0][0] + (warp_id / NUM_WARPS_N) * C_SMEM_STRIDE * WARP_M
                                   + (lane_id / 4) * C_SMEM_STRIDE;
    const half *smem_warp_stream_ptr = (half*)&smem[0][0] + warp_id * MMA_M * 2 * C_SMEM_STRIDE;

    uint32_t col1 = (warp_id % NUM_WARPS_N) * WARP_N +
                    (lane_id % 4) * (sizeof(uint32_t) / sizeof(half));
#pragma unroll
    for (uint32_t i = 0; i < WARP_TILES_M; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < WARP_TILES_N; ++j) {
            // 上 8 行，4个lane 一行
            half *lane_ptr0 = smem_warp_tile_row_ptr + (i * MMA_M * C_SMEM_STRIDE + col1 + j * MMA_N);
            // 下 8 行
            half *lane_ptr1 = lane_ptr0 + 8 * C_SMEM_STRIDE;

            half2 up = __halves2half2(RC[i][j][0] * scale, RC[i][j][1] * scale);
            half2 down = __halves2half2(RC[i][j][2] * scale, RC[i][j][3] * scale);
            *((half2 *)(lane_ptr0)) = up;
            *((half2 *)(lane_ptr1)) = down;
        }
    }

    __syncthreads();

    static_assert(SMEM_BANK_ROWS == 2); // 下面的2都是这个
    static constexpr uint32_t WRITE_ITER = (BLOCK_M / NUM_WARPS / 2);
    const uint32_t gmem_idx = (block_tile_i * BLOCK_M + warp_id * WRITE_ITER * 2) * N + block_tile_j * BLOCK_N;
    smem_warp_stream_ptr = (half*)&smem[0][0] + (warp_id * WRITE_ITER * 2 * C_SMEM_STRIDE
                                          + (lane_id / 16) * C_SMEM_STRIDE + (lane_id % 16) * 8);
    const half *lane_C = &C[gmem_idx] + ((lane_id / 16) * N + (lane_id % 16) * 8);

    for (uint32_t i = 0; i < WRITE_ITER; ++i) {
        *((int4 *)(lane_C + i * 2 * N)) = *(int4 *)(smem_warp_stream_ptr + i * 2 * C_SMEM_STRIDE);
    }
#endif
}

#define BLOCK_M 256
#define BLOCK_N 128
#define BLOCK_K 64

#define BLOCK_WARPS_M 4  // BLOCK_M / WARP_M
#define BLOCK_WARPS_N 2  // BLOCK_N / WARP_N

#define K_STAGE 3

static size_t initMmaFp8() {
//    int dev_id = 0;
//    BM_CUDART_ASSERT(cudaGetDevice(&dev_id));
//
//    cudaDeviceProp dev_prop;
//    BM_CUDART_ASSERT(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t ab_smem_size = (BLOCK_M + BLOCK_N) * AB_SMEM_STRIDE * sizeof(int8_t) * K_STAGE;
    size_t out_smem_size = BLOCK_M * C_SMEM_STRIDE * sizeof(half);
    size_t smem_max_size = std::max(ab_smem_size, out_smem_size);

//    BM_ASSERT_LE(smem_max_size, dev_prop.sharedMemPerMultiprocessor, "");
//    auto kernel = KERNEL_mma_fp8<FP8_E4M3, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_WARPS_M, BLOCK_WARPS_N, K_STAGE>;
//    BM_CUDART_ASSERT(cudaGetLastError());
    return smem_max_size;
}

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void mma_fp8(cudaStream_t stream, int8_t *A, int8_t *B, half *C, float alpha, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaFp8();

    dim3 block(BLOCK_WARPS_M * BLOCK_WARPS_N * WARP_SIZE);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_M), div_ceil(N, BLOCK_N * BLOCK_STRIDE));
//    std::cout << "smem_max_size: " << smem_max_size << ", grid.y=" << grid.y << ", grid.z=" << grid.z << std::endl;

    auto kernel = KERNEL_mma_fp8<FP8_E4M3, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_WARPS_M, BLOCK_WARPS_N, K_STAGE>;
    BM_CUDART_ASSERT(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));
    kernel<<<grid, block, smem_max_size, stream>>>(A, B, C, M, N, K, alpha);
    BM_CUDART_ASSERT(cudaGetLastError());
}
#else
void mma_fp8(cudaStream_t stream, int8_t *A, int8_t *B, half *C, float alpha, size_t M, size_t N, size_t K) {
    BM_ASSERT(false, "");
}
#endif
}
