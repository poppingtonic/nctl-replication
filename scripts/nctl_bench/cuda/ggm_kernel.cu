/*
 * CUDA kernel for batched GGM forward + update.
 *
 * One thread-block per GGM node. Each block:
 * 1. Computes context index from hyperplanes @ z
 * 2. Loads active weight row into shared memory
 * 3. Computes logit(p_prev), dot product, sigmoid → prediction
 * 4. Gradient update: w += lr * error * logits
 *
 * Processes a BATCH of samples sequentially on-device without CPU roundtrip.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Numerical constants
#define EPS 1e-7f
#define MAX_WEIGHT 200.0f
#define LOGIT_CLIP 15.0f

__device__ __forceinline__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float d_logit(float p) {
    p = fmaxf(fminf(p, 1.0f - EPS), EPS);
    float v = logf(p / (1.0f - p));
    return fmaxf(fminf(v, LOGIT_CLIP), -LOGIT_CLIP);
}

/*
 * Process a batch of samples through one GGM layer.
 *
 * Each thread-block handles one node across ALL samples sequentially.
 * Within a block, threads cooperate on the K-dimensional reduction.
 *
 * Args:
 *   z_batch:        [B, D]     — side information for each sample
 *   p_prev_batch:   [B, K]     — input probabilities for each sample
 *   symbols_batch:  [B]        — target symbols (0 or 1)
 *   weights:        [N, C, K]  — GGM weights (modified in-place)
 *   hyperplanes:    [N, H, D]  — gating hyperplanes (read-only)
 *   hp_bias:        [N, H]     — gating bias (read-only)
 *   predictions:    [B, N]     — output: P(symbol) per sample per node
 *   lr:             scalar     — learning rate
 *   B, N, K, D, H, C:  dimensions
 */
__global__ void ggm_batch_forward_update(
    const float* __restrict__ z_batch,        // [B, D]
    const float* __restrict__ p_prev_batch,   // [B, K]
    const int*   __restrict__ symbols_batch,  // [B]
    float*       __restrict__ weights,        // [N, C, K]
    const float* __restrict__ hyperplanes,    // [N, H, D]
    const float* __restrict__ hp_bias,        // [N, H]
    float*       __restrict__ predictions,    // [B, N]
    float lr,
    int B, int N, int K, int D, int H, int C
) {
    int node = blockIdx.x;  // one block per node
    if (node >= N) return;

    // Shared memory for the active weight row [K] and logits [K]
    extern __shared__ float smem[];
    float* s_w = smem;            // [K]
    float* s_logits = smem + K;   // [K]

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Process each sample sequentially
    for (int b = 0; b < B; b++) {
        const float* z = z_batch + b * D;
        const float* p_prev = p_prev_batch + b * K;
        int symbol = symbols_batch[b];

        // --- Step 1: Compute context index ---
        int context = 0;
        for (int h = 0; h < H; h++) {
            float dot = 0.0f;
            // Parallel reduction across D dimensions
            for (int d = tid; d < D; d += num_threads) {
                dot += hyperplanes[node * H * D + h * D + d] * z[d];
            }
            // Warp reduction
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            // Thread 0 has the result
            if (tid == 0) {
                dot += hp_bias[node * H + h];
                if (dot >= 0.0f) context |= (1 << h);
            }
        }
        // Broadcast context to all threads
        context = __shfl_sync(0xffffffff, context, 0);

        // --- Step 2: Load active weight row and compute logits ---
        for (int k = tid; k < K; k += num_threads) {
            s_w[k] = weights[node * C * K + context * K + k];
            s_logits[k] = d_logit(p_prev[k]);
        }
        __syncthreads();

        // --- Step 3: Dot product w · logits ---
        float local_sum = 0.0f;
        for (int k = tid; k < K; k += num_threads) {
            local_sum += s_w[k] * s_logits[k];
        }
        // Warp reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }

        float p1, p_sym, error;
        if (tid == 0) {
            p1 = d_sigmoid(local_sum);
            p_sym = (symbol == 1) ? p1 : (1.0f - p1);
            predictions[b * N + node] = p_sym;
            error = (float)symbol - p1;
        }
        // Broadcast error to all threads
        error = __shfl_sync(0xffffffff, error, 0);

        // --- Step 4: Gradient update ---
        for (int k = tid; k < K; k += num_threads) {
            float delta = lr * error * s_logits[k];
            float new_w = s_w[k] + delta;
            new_w = fmaxf(fminf(new_w, MAX_WEIGHT), -MAX_WEIGHT);
            weights[node * C * K + context * K + k] = new_w;
        }
        __syncthreads();
    }
}

/*
 * Forward-only pass (no weight update). For evaluation.
 */
__global__ void ggm_batch_forward_only(
    const float* __restrict__ z_batch,
    const float* __restrict__ p_prev_batch,
    const float* __restrict__ weights,
    const float* __restrict__ hyperplanes,
    const float* __restrict__ hp_bias,
    float*       __restrict__ predictions,
    int B, int N, int K, int D, int H, int C
) {
    int node = blockIdx.x;
    if (node >= N) return;

    extern __shared__ float smem[];
    float* s_w = smem;
    float* s_logits = smem + K;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int b = 0; b < B; b++) {
        const float* z = z_batch + b * D;
        const float* p_prev = p_prev_batch + b * K;

        int context = 0;
        for (int h = 0; h < H; h++) {
            float dot = 0.0f;
            for (int d = tid; d < D; d += num_threads) {
                dot += hyperplanes[node * H * D + h * D + d] * z[d];
            }
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (tid == 0) {
                dot += hp_bias[node * H + h];
                if (dot >= 0.0f) context |= (1 << h);
            }
        }
        context = __shfl_sync(0xffffffff, context, 0);

        for (int k = tid; k < K; k += num_threads) {
            s_w[k] = weights[node * C * K + context * K + k];
            s_logits[k] = d_logit(p_prev[k]);
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int k = tid; k < K; k += num_threads) {
            local_sum += s_w[k] * s_logits[k];
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }

        if (tid == 0) {
            predictions[b * N + node] = d_sigmoid(local_sum);
        }
        __syncthreads();
    }
}


// --- PyTorch C++ bindings ---

torch::Tensor ggm_forward_update(
    torch::Tensor z_batch,       // [B, D]
    torch::Tensor p_prev_batch,  // [B, K]
    torch::Tensor symbols_batch, // [B] int
    torch::Tensor weights,       // [N, C, K]
    torch::Tensor hyperplanes,   // [N, H, D]
    torch::Tensor hp_bias,       // [N, H]
    float lr
) {
    int B = z_batch.size(0);
    int D = z_batch.size(1);
    int K = p_prev_batch.size(1);
    int N = weights.size(0);
    int C = weights.size(1);
    int H = hyperplanes.size(1);

    auto predictions = torch::empty({B, N}, z_batch.options());

    int threads = min(K, 256);  // one thread per weight dimension, up to 256
    int shared_mem = 2 * K * sizeof(float);  // w + logits

    ggm_batch_forward_update<<<N, threads, shared_mem>>>(
        z_batch.data_ptr<float>(),
        p_prev_batch.data_ptr<float>(),
        symbols_batch.data_ptr<int>(),
        weights.data_ptr<float>(),
        hyperplanes.data_ptr<float>(),
        hp_bias.data_ptr<float>(),
        predictions.data_ptr<float>(),
        lr, B, N, K, D, H, C
    );

    return predictions;  // [B, N]
}

torch::Tensor ggm_forward_only(
    torch::Tensor z_batch,
    torch::Tensor p_prev_batch,
    torch::Tensor weights,
    torch::Tensor hyperplanes,
    torch::Tensor hp_bias
) {
    int B = z_batch.size(0);
    int D = z_batch.size(1);
    int K = p_prev_batch.size(1);
    int N = weights.size(0);
    int C = weights.size(1);
    int H = hyperplanes.size(1);

    auto predictions = torch::empty({B, N}, z_batch.options());

    int threads = min(K, 256);
    int shared_mem = 2 * K * sizeof(float);

    ggm_batch_forward_only<<<N, threads, shared_mem>>>(
        z_batch.data_ptr<float>(),
        p_prev_batch.data_ptr<float>(),
        weights.data_ptr<float>(),
        hyperplanes.data_ptr<float>(),
        hp_bias.data_ptr<float>(),
        predictions.data_ptr<float>(),
        B, N, K, D, H, C
    );

    return predictions;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_update", &ggm_forward_update,
          "GGM batched forward + update (CUDA)");
    m.def("forward_only", &ggm_forward_only,
          "GGM batched forward only (CUDA)");
}
