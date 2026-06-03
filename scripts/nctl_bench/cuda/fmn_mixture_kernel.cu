/*
 * CUDA kernel for FMN posterior mixture prediction with pool-synced weight
 * updates.
 *
 * Active model slots for each node:
 *   - slots 0..pool_size-1: active copies of immutable pool snapshots
 *   - slot M-1: fresh/base GGM for the current segment
 *
 * Prediction uses the Bayesian conditional of the FMN segment mixture.  The
 * current-segment log likelihoods are posterior logits, with priors
 * 1/2 for the fresh model and 1/(2k) for each of k pool models.  All active
 * models are updated on each sample; immutable snapshots are refreshed only at
 * segment boundaries by Python.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define EPS 1e-7f
#define MAX_WEIGHT 200.0f
#define LOGIT_CLIP 15.0f
#define LOG_HALF -0.6931471805599453f

__device__ __forceinline__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float d_logit(float p) {
    p = fmaxf(fminf(p, 1.0f - EPS), EPS);
    float v = logf(p / (1.0f - p));
    return fmaxf(fminf(v, LOGIT_CLIP), -LOGIT_CLIP);
}

__device__ void reduce_block_sum(float* scratch, float value) {
    int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
}

__device__ int compute_context(
    int node,
    const float* __restrict__ z,
    const float* __restrict__ hyperplanes,
    const float* __restrict__ hp_bias,
    float* scratch,
    int D,
    int H
) {
    int tid = threadIdx.x;
    int context = 0;
    for (int h = 0; h < H; h++) {
        float dot = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) {
            dot += hyperplanes[node * H * D + h * D + d] * z[d];
        }
        reduce_block_sum(scratch, dot);
        if (tid == 0) {
            float total = scratch[0] + hp_bias[node * H + h];
            if (total >= 0.0f) context |= (1 << h);
        }
        __syncthreads();
    }
    if (tid == 0) scratch[0] = (float)context;
    __syncthreads();
    return (int)scratch[0];
}

__device__ float posterior_mixture_prediction(
    int node,
    int M,
    int k_pool,
    int fresh_idx,
    float posterior_temp,
    const float* __restrict__ model_preds,
    const float* __restrict__ segment_log_probs,
    float* __restrict__ posterior_weights
) {
    if (k_pool <= 0) return model_preds[fresh_idx];

    // Scale current-segment log-likelihood by posterior_temp to combat
    // posterior saturation in long segments.  Priors are NOT scaled —
    // FMN Eq. 5 priors are mixture
    // weights, not data evidence.  posterior_temp=1.0 reproduces the
    // standard Bayesian conditional; posterior_temp -> 0 collapses to
    // pure prior averaging.
    float max_logit = -INFINITY;
    float log_pool_prior = LOG_HALF - logf((float)k_pool);
    for (int m = 0; m < k_pool; m++) {
        float v = posterior_temp * segment_log_probs[node * M + m] + log_pool_prior;
        posterior_weights[m] = v;
        max_logit = fmaxf(max_logit, v);
    }
    float fresh_v = posterior_temp * segment_log_probs[node * M + fresh_idx] + LOG_HALF;
    posterior_weights[fresh_idx] = fresh_v;
    max_logit = fmaxf(max_logit, fresh_v);

    float denom = 0.0f;
    float pred = 0.0f;
    for (int m = 0; m < k_pool; m++) {
        float w = expf(posterior_weights[m] - max_logit);
        denom += w;
        pred += w * model_preds[m];
    }
    float fresh_w = expf(posterior_weights[fresh_idx] - max_logit);
    denom += fresh_w;
    pred += fresh_w * model_preds[fresh_idx];
    return pred / fmaxf(denom, 1e-30f);
}

__global__ void fmn_mixture_forward_update(
    const float* __restrict__ z_batch,
    const float* __restrict__ p_prev_batch,
    const int*   __restrict__ symbols,
    float*       __restrict__ mixture_weights,  // [N, M, C, K_in]
    const float* __restrict__ hyperplanes,
    const float* __restrict__ hp_bias,
    const int*   __restrict__ pool_sizes,
    float*       __restrict__ predictions,
    float*       __restrict__ segment_log_probs,
    float*       __restrict__ model_log_probs,
    float lr,
    float posterior_temp,
    int B, int N, int M, int C, int K_in, int D, int H
) {
    int node = blockIdx.x;
    if (node >= N) return;

    int tid = threadIdx.x;
    int k_pool = pool_sizes[node];
    int fresh_idx = M - 1;
    int node_offset = node * M * C * K_in;

    extern __shared__ float smem[];
    float* s_logits = smem;                       // [K_in]
    float* s_model_preds = smem + K_in;           // [M]
    float* s_posterior = s_model_preds + M;       // [M]
    float* s_reduce = s_posterior + M;            // [blockDim.x]

    for (int b = 0; b < B; b++) {
        const float* z = z_batch + b * D;
        const float* p_prev = p_prev_batch + b * K_in;
        int symbol = symbols[b];

        int context = compute_context(node, z, hyperplanes, hp_bias, s_reduce, D, H);

        for (int k = tid; k < K_in; k += blockDim.x) {
            s_logits[k] = d_logit(p_prev[k]);
        }
        __syncthreads();

        // Pool model predictions.
        for (int m = tid; m < k_pool; m += blockDim.x) {
            int w_base = node_offset + m * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[m] = d_sigmoid(dot);
        }
        // Fresh model prediction.  The fresh slot is fixed at M-1.
        if (tid == 0) {
            int w_base = node_offset + fresh_idx * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[fresh_idx] = d_sigmoid(dot);
        }
        __syncthreads();

        if (tid == 0) {
            float p_mixture = posterior_mixture_prediction(
                node, M, k_pool, fresh_idx, posterior_temp,
                s_model_preds, segment_log_probs, s_posterior
            );
            float p_sym = (symbol == 1) ? p_mixture : (1.0f - p_mixture);
            predictions[b * N + node] = p_sym;
        }
        __syncthreads();

        // Update pool slots and likelihood trackers.
        for (int m = 0; m < k_pool; m++) {
            float p_m = s_model_preds[m];
            float error_m = (float)symbol - p_m;
            int w_base = node_offset + m * C * K_in + context * K_in;
            for (int k = tid; k < K_in; k += blockDim.x) {
                float new_w = mixture_weights[w_base + k] + lr * error_m * s_logits[k];
                mixture_weights[w_base + k] = fmaxf(fminf(new_w, MAX_WEIGHT), -MAX_WEIGHT);
            }
            if (tid == 0) {
                float p_m_sym = (symbol == 1) ? p_m : (1.0f - p_m);
                float lp = logf(fmaxf(p_m_sym, 1e-30f));
                segment_log_probs[node * M + m] += lp;
                model_log_probs[node * M + m] += lp;
            }
            __syncthreads();
        }

        // Update fresh slot.
        float p_fresh = s_model_preds[fresh_idx];
        float error_fresh = (float)symbol - p_fresh;
        int fresh_base = node_offset + fresh_idx * C * K_in + context * K_in;
        for (int k = tid; k < K_in; k += blockDim.x) {
            float new_w = mixture_weights[fresh_base + k] + lr * error_fresh * s_logits[k];
            mixture_weights[fresh_base + k] = fmaxf(fminf(new_w, MAX_WEIGHT), -MAX_WEIGHT);
        }
        if (tid == 0) {
            float p_sym = (symbol == 1) ? p_fresh : (1.0f - p_fresh);
            float lp = logf(fmaxf(p_sym, 1e-30f));
            segment_log_probs[node * M + fresh_idx] += lp;
            model_log_probs[node * M + fresh_idx] += lp;
        }
        __syncthreads();
    }
}

__global__ void fmn_mixture_forward_only(
    const float* __restrict__ z_batch,
    const float* __restrict__ p_prev_batch,
    const float* __restrict__ mixture_weights,
    const float* __restrict__ hyperplanes,
    const float* __restrict__ hp_bias,
    const int*   __restrict__ pool_sizes,
    const float* __restrict__ segment_log_probs,
    float*       __restrict__ predictions,
    float posterior_temp,
    int B, int N, int M, int C, int K_in, int D, int H
) {
    int node = blockIdx.x;
    if (node >= N) return;

    int tid = threadIdx.x;
    int k_pool = pool_sizes[node];
    int fresh_idx = M - 1;
    int node_offset = node * M * C * K_in;

    extern __shared__ float smem[];
    float* s_logits = smem;
    float* s_model_preds = smem + K_in;
    float* s_posterior = s_model_preds + M;
    float* s_reduce = s_posterior + M;

    for (int b = 0; b < B; b++) {
        const float* z = z_batch + b * D;
        const float* p_prev = p_prev_batch + b * K_in;

        int context = compute_context(node, z, hyperplanes, hp_bias, s_reduce, D, H);

        for (int k = tid; k < K_in; k += blockDim.x) s_logits[k] = d_logit(p_prev[k]);
        __syncthreads();

        for (int m = tid; m < k_pool; m += blockDim.x) {
            int w_base = node_offset + m * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[m] = d_sigmoid(dot);
        }
        if (tid == 0) {
            int w_base = node_offset + fresh_idx * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[fresh_idx] = d_sigmoid(dot);
        }
        __syncthreads();

        if (tid == 0) {
            predictions[b * N + node] = posterior_mixture_prediction(
                node, M, k_pool, fresh_idx, posterior_temp,
                s_model_preds, segment_log_probs, s_posterior
            );
        }
        __syncthreads();
    }
}

static int choose_threads(int k) {
    int threads = 1;
    int target = k < 256 ? k : 256;
    while (threads < target) threads <<= 1;
    return threads;
}

torch::Tensor fmn_mixture_forward_update_py(
    torch::Tensor z_batch,
    torch::Tensor p_prev_batch,
    torch::Tensor symbols,
    torch::Tensor mixture_weights,
    torch::Tensor hyperplanes,
    torch::Tensor hp_bias,
    torch::Tensor pool_sizes,
    torch::Tensor segment_log_probs,
    torch::Tensor model_log_probs,
    float lr,
    float posterior_temp
) {
    int B = z_batch.size(0);
    int D = z_batch.size(1);
    int K_in = p_prev_batch.size(1);
    int N = mixture_weights.size(0);
    int M = mixture_weights.size(1);
    int C = mixture_weights.size(2);
    int H = hyperplanes.size(1);

    auto predictions = torch::empty({B, N}, z_batch.options());
    int threads = choose_threads(K_in);
    int smem = (K_in + M + M + threads) * sizeof(float);

    fmn_mixture_forward_update<<<N, threads, smem>>>(
        z_batch.data_ptr<float>(), p_prev_batch.data_ptr<float>(),
        symbols.data_ptr<int>(), mixture_weights.data_ptr<float>(),
        hyperplanes.data_ptr<float>(), hp_bias.data_ptr<float>(),
        pool_sizes.data_ptr<int>(), predictions.data_ptr<float>(),
        segment_log_probs.data_ptr<float>(), model_log_probs.data_ptr<float>(),
        lr, posterior_temp, B, N, M, C, K_in, D, H
    );
    return predictions;
}

torch::Tensor fmn_mixture_forward_only_py(
    torch::Tensor z_batch,
    torch::Tensor p_prev_batch,
    torch::Tensor mixture_weights,
    torch::Tensor hyperplanes,
    torch::Tensor hp_bias,
    torch::Tensor pool_sizes,
    torch::Tensor segment_log_probs,
    float posterior_temp
) {
    int B = z_batch.size(0);
    int D = z_batch.size(1);
    int K_in = p_prev_batch.size(1);
    int N = mixture_weights.size(0);
    int M = mixture_weights.size(1);
    int C = mixture_weights.size(2);
    int H = hyperplanes.size(1);

    auto predictions = torch::empty({B, N}, z_batch.options());
    int threads = choose_threads(K_in);
    int smem = (K_in + M + M + threads) * sizeof(float);

    fmn_mixture_forward_only<<<N, threads, smem>>>(
        z_batch.data_ptr<float>(), p_prev_batch.data_ptr<float>(),
        mixture_weights.data_ptr<float>(), hyperplanes.data_ptr<float>(),
        hp_bias.data_ptr<float>(), pool_sizes.data_ptr<int>(),
        segment_log_probs.data_ptr<float>(), predictions.data_ptr<float>(),
        posterior_temp, B, N, M, C, K_in, D, H
    );
    return predictions;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_update", &fmn_mixture_forward_update_py,
          "FMN posterior mixture forward + update all active models (CUDA)");
    m.def("forward_only", &fmn_mixture_forward_only_py,
          "FMN posterior mixture forward only (CUDA)");
}
