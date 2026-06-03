/*
 * CUDA kernel for FMN multi-level posterior-uniform (paper Eq. 5) prediction
 * with pool-synced weight updates --- Phase 5E fused per-level kernel.
 *
 * Replaces the Python `for level in range(L): cuda.forward_update(...)`
 * dispatch in `nctl_network.py` with a single launch over all L PTW
 * levels' per-level active state.  This kills launch overhead in the
 * dominant inner loop on the DP training path.
 *
 * Per-level active state layout:
 *   mixture_weights : [L, N, M, C, K_in]  one float per level/node/slot/context/feature
 *   pool_sizes      : [L, N]               int32 per-node pool occupancy at segment open
 *   segment_log_probs : [L, N, M]          float32 current-segment per-slot log-likelihood
 *   model_log_probs   : [L, N, M]          float32 lifetime diagnostic log-likelihood
 *
 * Hyperplanes / hp_bias / z / p_prev / symbols are all level-INDEPENDENT;
 * the same input drives every level's update.  Predictions are written to
 * a [L, B, N] tensor so Python's existing [L, B, N] cumsum DP path consumes
 * them directly with zero reshapes.
 *
 * Returns predictions == ν_j(x_obs | x_<t) computed with the paper-faithful
 * conditional ratio (Eq. 5): prior weights 1/2 fresh + (1/2)/(M-1) per pool
 * slot, multiplied by per-slot segment likelihoods (posterior_temp scales the
 * data evidence; posterior_temp=0 collapses to the pre-5I uniform mixture so
 * existing math tests remain bit-stable when the caller explicitly opts in).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define EPS 1e-7f
#define MAX_WEIGHT 200.0f
#define LOGIT_CLIP 15.0f

__device__ __forceinline__ float ml_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float ml_logit(float p) {
    p = fmaxf(fminf(p, 1.0f - EPS), EPS);
    float v = logf(p / (1.0f - p));
    return fmaxf(fminf(v, LOGIT_CLIP), -LOGIT_CLIP);
}

__device__ void ml_reduce_block_sum(float* scratch, float value) {
    int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
}

__device__ int ml_compute_context(
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
        ml_reduce_block_sum(scratch, dot);
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

/*
 * Unweighted uniform mixture: (1/2) * fresh + (1/2) * mean(pool).
 *
 * This is the posterior_temp=0 case of the paper Eq. 5 conditional, and also
 * the exact segment-start value when all per-slot log-likelihoods are equal.
 * The default DP dispatch path uses ml_paper_posterior_mixture below.
 */
__device__ float ml_paper_uniform_mixture(
    int k_pool,
    int fresh_idx,
    const float* __restrict__ model_preds
) {
    float fresh = model_preds[fresh_idx];
    if (k_pool <= 0) return fresh;
    float pool_sum = 0.0f;
    for (int m = 0; m < k_pool; m++) pool_sum += model_preds[m];
    float pool_mean = pool_sum / (float)k_pool;
    return 0.5f * fresh + 0.5f * pool_mean;
}

/*
 * Paper-faithful posterior-weighted ν_j(x_t=1|x_<t).  Eq. 5 of the FMN
 * paper defines ν_t(s) as a JOINT mixture; the conditional at step t is
 *
 *   ν_j(x_t|x_<t) = ν_j(x_<t, x_t) / ν_j(x_<t)
 *
 *      Σ_m  prior_m * exp(log ρ_m(x_<t)) * ρ_m(x_t|x_<t)
 *   = ────────────────────────────────────────────────────
 *           Σ_m  prior_m * exp(log ρ_m(x_<t))
 *
 * with paper-Eq.-5 priors
 *   prior_fresh = 1/2          (the base measure ρ)
 *   prior_m     = (1/2)/(M-1)  for each non-fresh pool slot (paper uses M\{ρ}).
 *
 * The current-segment log-likelihoods live in `segment_log_probs[level,n,m]`,
 * already accumulated incrementally by the kernel after every step.  When
 * `posterior_temp == 0.0f` this function collapses to the previous uniform
 * mixture (so the segment-start state -- log_probs all zero -- matches the
 * paper exactly).
 */
__device__ float ml_paper_posterior_mixture(
    int k_pool,
    int fresh_idx,
    float posterior_temp,
    const float* __restrict__ model_preds,
    const float* __restrict__ segment_log_probs_level_node,
    float* __restrict__ posterior_weights
) {
    if (k_pool <= 0) return model_preds[fresh_idx];
    const float LOG_HALF = -0.6931471805599453f;
    float log_pool_prior = LOG_HALF - logf((float)k_pool);

    float max_logit = -INFINITY;
    for (int m = 0; m < k_pool; m++) {
        float v = posterior_temp * segment_log_probs_level_node[m] + log_pool_prior;
        posterior_weights[m] = v;
        if (v > max_logit) max_logit = v;
    }
    float fresh_v = posterior_temp * segment_log_probs_level_node[fresh_idx] + LOG_HALF;
    posterior_weights[fresh_idx] = fresh_v;
    if (fresh_v > max_logit) max_logit = fresh_v;

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

__global__ void fmn_mixture_multilevel_forward_update(
    const float* __restrict__ z_batch,
    const float* __restrict__ p_prev_batch,
    const int*   __restrict__ symbols,
    float*       __restrict__ mixture_weights,    // [L, N, M, C, K_in]
    const float* __restrict__ hyperplanes,        // [N, H, D]   level-indep
    const float* __restrict__ hp_bias,            // [N, H]      level-indep
    const int*   __restrict__ pool_sizes,         // [L, N]
    float*       __restrict__ predictions,        // [L, B, N]
    float*       __restrict__ segment_log_probs,  // [L, N, M]
    float*       __restrict__ model_log_probs,    // [L, N, M]
    float lr,
    float posterior_temp,
    int B, int L, int N, int M, int C, int K_in, int D, int H
) {
    int gid = blockIdx.x;
    int level = gid / N;
    int node = gid - level * N;
    if (level >= L) return;
    if (node >= N) return;

    int tid = threadIdx.x;
    int k_pool = pool_sizes[level * N + node];
    int fresh_idx = M - 1;
    int level_node_offset = (level * N + node) * M * C * K_in;
    int level_seg_offset = (level * N + node) * M;

    extern __shared__ float smem[];
    float* s_logits = smem;                       // [K_in]
    float* s_model_preds = smem + K_in;           // [M]
    float* s_posterior = s_model_preds + M;       // [M]
    float* s_reduce = s_posterior + M;            // [blockDim.x]

    for (int b = 0; b < B; b++) {
        const float* z = z_batch + b * D;
        const float* p_prev = p_prev_batch + b * K_in;
        int symbol = symbols[b];

        // Context is level-independent (hyperplanes/hp_bias shared), but
        // we recompute per (level, node) block because the scratch space
        // is block-local.  Cheap relative to the K_in-wide weight updates.
        int context = ml_compute_context(node, z, hyperplanes, hp_bias, s_reduce, D, H);

        for (int k = tid; k < K_in; k += blockDim.x) {
            s_logits[k] = ml_logit(p_prev[k]);
        }
        __syncthreads();

        // Pool model predictions for this (level, node).
        for (int m = tid; m < k_pool; m += blockDim.x) {
            int w_base = level_node_offset + m * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[m] = ml_sigmoid(dot);
        }
        if (tid == 0) {
            int w_base = level_node_offset + fresh_idx * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[fresh_idx] = ml_sigmoid(dot);
        }
        __syncthreads();

        if (tid == 0) {
            float p_mixture = ml_paper_posterior_mixture(
                k_pool, fresh_idx, posterior_temp,
                s_model_preds,
                &segment_log_probs[level_seg_offset],
                s_posterior
            );
            float p_sym = (symbol == 1) ? p_mixture : (1.0f - p_mixture);
            // Predictions written in [L, B, N] order to match Python's
            // existing forward_update_batch_dp stack.
            predictions[(level * B + b) * N + node] = p_sym;
        }
        __syncthreads();

        // Update pool slots and likelihood trackers.
        for (int m = 0; m < k_pool; m++) {
            float p_m = s_model_preds[m];
            float error_m = (float)symbol - p_m;
            int w_base = level_node_offset + m * C * K_in + context * K_in;
            for (int k = tid; k < K_in; k += blockDim.x) {
                float new_w = mixture_weights[w_base + k] + lr * error_m * s_logits[k];
                mixture_weights[w_base + k] = fmaxf(fminf(new_w, MAX_WEIGHT), -MAX_WEIGHT);
            }
            if (tid == 0) {
                float p_m_sym = (symbol == 1) ? p_m : (1.0f - p_m);
                float lp = logf(fmaxf(p_m_sym, 1e-30f));
                segment_log_probs[level_seg_offset + m] += lp;
                model_log_probs[level_seg_offset + m] += lp;
            }
            __syncthreads();
        }

        // Update fresh slot.
        float p_fresh = s_model_preds[fresh_idx];
        float error_fresh = (float)symbol - p_fresh;
        int fresh_base = level_node_offset + fresh_idx * C * K_in + context * K_in;
        for (int k = tid; k < K_in; k += blockDim.x) {
            float new_w = mixture_weights[fresh_base + k] + lr * error_fresh * s_logits[k];
            mixture_weights[fresh_base + k] = fmaxf(fminf(new_w, MAX_WEIGHT), -MAX_WEIGHT);
        }
        if (tid == 0) {
            float p_sym = (symbol == 1) ? p_fresh : (1.0f - p_fresh);
            float lp = logf(fmaxf(p_sym, 1e-30f));
            segment_log_probs[level_seg_offset + fresh_idx] += lp;
            model_log_probs[level_seg_offset + fresh_idx] += lp;
        }
        __syncthreads();
    }
}

__global__ void fmn_mixture_multilevel_forward_only(
    const float* __restrict__ z_batch,
    const float* __restrict__ p_prev_batch,
    const float* __restrict__ mixture_weights,    // [L, N, M, C, K_in]
    const float* __restrict__ hyperplanes,
    const float* __restrict__ hp_bias,
    const int*   __restrict__ pool_sizes,         // [L, N]
    const float* __restrict__ segment_log_probs,  // [L, N, M] (read-only)
    float*       __restrict__ predictions,        // [L, B, N]
    float posterior_temp,
    int B, int L, int N, int M, int C, int K_in, int D, int H
) {
    int gid = blockIdx.x;
    int level = gid / N;
    int node = gid - level * N;
    if (level >= L) return;
    if (node >= N) return;

    int tid = threadIdx.x;
    int k_pool = pool_sizes[level * N + node];
    int fresh_idx = M - 1;
    int level_node_offset = (level * N + node) * M * C * K_in;
    int level_seg_offset = (level * N + node) * M;

    extern __shared__ float smem[];
    float* s_logits = smem;
    float* s_model_preds = smem + K_in;
    float* s_posterior = s_model_preds + M;       // [M]
    float* s_reduce = s_posterior + M;            // [blockDim.x]

    for (int b = 0; b < B; b++) {
        const float* z = z_batch + b * D;
        const float* p_prev = p_prev_batch + b * K_in;
        int context = ml_compute_context(node, z, hyperplanes, hp_bias, s_reduce, D, H);

        for (int k = tid; k < K_in; k += blockDim.x) s_logits[k] = ml_logit(p_prev[k]);
        __syncthreads();

        for (int m = tid; m < k_pool; m += blockDim.x) {
            int w_base = level_node_offset + m * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[m] = ml_sigmoid(dot);
        }
        if (tid == 0) {
            int w_base = level_node_offset + fresh_idx * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[fresh_idx] = ml_sigmoid(dot);
        }
        __syncthreads();

        if (tid == 0) {
            float p_mixture = ml_paper_posterior_mixture(
                k_pool, fresh_idx, posterior_temp,
                s_model_preds,
                &segment_log_probs[level_seg_offset],
                s_posterior
            );
            predictions[(level * B + b) * N + node] = p_mixture;
        }
        __syncthreads();
    }
}

__device__ void ml_apply_close_reset(
    int level,
    int node,
    int M,
    int C,
    int K_in,
    int k_pool,
    int fresh_idx,
    int level_node_offset,
    int level_seg_offset,
    int shared_node_pool_offset,
    float* __restrict__ mixture_weights,
    const float* __restrict__ pool_snapshots,
    float* __restrict__ segment_log_probs
) {
    int tid = threadIdx.x;
    // Restore active pool slots from immutable pool snapshots.  Each slot
    // has C * K_in entries; cooperate across the thread block.
    int slot_entries = C * K_in;
    int total_pool_entries = k_pool * slot_entries;
    for (int idx = tid; idx < total_pool_entries; idx += blockDim.x) {
        int m = idx / slot_entries;
        int rem = idx - m * slot_entries;
        mixture_weights[level_node_offset + m * slot_entries + rem] =
            pool_snapshots[shared_node_pool_offset + m * slot_entries + rem];
    }
    // Zero unused active slots [k_pool .. fresh_idx-1].
    int unused_start = k_pool;
    int unused_end = fresh_idx;  // exclusive; fresh slot preserved
    int unused_count = (unused_end - unused_start) * slot_entries;
    for (int idx = tid; idx < unused_count; idx += blockDim.x) {
        int m = idx / slot_entries;
        int rem = idx - m * slot_entries;
        mixture_weights[level_node_offset + (unused_start + m) * slot_entries + rem] = 0.0f;
    }
    // Zero current-segment evidence for every slot (including the fresh
    // slot, whose weights are kept but its segment_log_probs is reset
    // so the FMN posterior priors are honoured at segment open).
    for (int m = tid; m < M; m += blockDim.x) {
        segment_log_probs[level_seg_offset + m] = 0.0f;
    }
    __syncthreads();
}

__global__ void fmn_mixture_multilevel_forward_update_with_resets(
    const float* __restrict__ z_batch,
    const float* __restrict__ p_prev_batch,
    const int*   __restrict__ symbols,
    float*       __restrict__ mixture_weights,    // [L, N, M, C, K_in]
    const float* __restrict__ pool_snapshots,     // [N, pool_capacity, C, K_in]
    const float* __restrict__ hyperplanes,
    const float* __restrict__ hp_bias,
    const int*   __restrict__ pool_sizes,         // [L, N]
    const bool*  __restrict__ close_mask,         // [L, B]
    float*       __restrict__ predictions,        // [L, B, N]
    float*       __restrict__ segment_log_probs,  // [L, N, M]
    float*       __restrict__ model_log_probs,    // [L, N, M]
    float lr,
    float posterior_temp,
    int B, int L, int N, int M, int C, int K_in, int D, int H,
    int pool_capacity
) {
    int gid = blockIdx.x;
    int level = gid / N;
    int node = gid - level * N;
    if (level >= L) return;
    if (node >= N) return;

    int tid = threadIdx.x;
    int fresh_idx = M - 1;
    int level_node_offset = (level * N + node) * M * C * K_in;
    int level_seg_offset = (level * N + node) * M;
    int shared_node_pool_offset = node * pool_capacity * C * K_in;

    extern __shared__ float smem[];
    float* s_logits = smem;                       // [K_in]
    float* s_model_preds = smem + K_in;           // [M]
    float* s_posterior = s_model_preds + M;       // [M]
    float* s_reduce = s_posterior + M;            // [blockDim.x]

    for (int b = 0; b < B; b++) {
        // Apply close-only reset BEFORE this sample, if requested.
        if (close_mask[level * B + b]) {
            int k_pool_reset = pool_sizes[level * N + node];
            ml_apply_close_reset(
                level, node, M, C, K_in, k_pool_reset, fresh_idx,
                level_node_offset, level_seg_offset, shared_node_pool_offset,
                mixture_weights, pool_snapshots, segment_log_probs
            );
        }
        // Re-read k_pool after the optional reset (pool_sizes is invariant
        // for the sub-chunk, but reading it once per sample matches the
        // legacy kernel's semantics and keeps the code paths identical
        // when close_mask is all-False).
        int k_pool = pool_sizes[level * N + node];

        const float* z = z_batch + b * D;
        const float* p_prev = p_prev_batch + b * K_in;
        int symbol = symbols[b];

        int context = ml_compute_context(node, z, hyperplanes, hp_bias, s_reduce, D, H);

        for (int k = tid; k < K_in; k += blockDim.x) {
            s_logits[k] = ml_logit(p_prev[k]);
        }
        __syncthreads();

        for (int m = tid; m < k_pool; m += blockDim.x) {
            int w_base = level_node_offset + m * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[m] = ml_sigmoid(dot);
        }
        if (tid == 0) {
            int w_base = level_node_offset + fresh_idx * C * K_in + context * K_in;
            float dot = 0.0f;
            for (int k = 0; k < K_in; k++) dot += mixture_weights[w_base + k] * s_logits[k];
            s_model_preds[fresh_idx] = ml_sigmoid(dot);
        }
        __syncthreads();

        if (tid == 0) {
            float p_mixture = ml_paper_posterior_mixture(
                k_pool, fresh_idx, posterior_temp,
                s_model_preds,
                &segment_log_probs[level_seg_offset],
                s_posterior
            );
            float p_sym = (symbol == 1) ? p_mixture : (1.0f - p_mixture);
            predictions[(level * B + b) * N + node] = p_sym;
        }
        __syncthreads();

        for (int m = 0; m < k_pool; m++) {
            float p_m = s_model_preds[m];
            float error_m = (float)symbol - p_m;
            int w_base = level_node_offset + m * C * K_in + context * K_in;
            for (int k = tid; k < K_in; k += blockDim.x) {
                float new_w = mixture_weights[w_base + k] + lr * error_m * s_logits[k];
                mixture_weights[w_base + k] = fmaxf(fminf(new_w, MAX_WEIGHT), -MAX_WEIGHT);
            }
            if (tid == 0) {
                float p_m_sym = (symbol == 1) ? p_m : (1.0f - p_m);
                float lp = logf(fmaxf(p_m_sym, 1e-30f));
                segment_log_probs[level_seg_offset + m] += lp;
                model_log_probs[level_seg_offset + m] += lp;
            }
            __syncthreads();
        }

        float p_fresh = s_model_preds[fresh_idx];
        float error_fresh = (float)symbol - p_fresh;
        int fresh_base = level_node_offset + fresh_idx * C * K_in + context * K_in;
        for (int k = tid; k < K_in; k += blockDim.x) {
            float new_w = mixture_weights[fresh_base + k] + lr * error_fresh * s_logits[k];
            mixture_weights[fresh_base + k] = fmaxf(fminf(new_w, MAX_WEIGHT), -MAX_WEIGHT);
        }
        if (tid == 0) {
            float p_sym = (symbol == 1) ? p_fresh : (1.0f - p_fresh);
            float lp = logf(fmaxf(p_sym, 1e-30f));
            segment_log_probs[level_seg_offset + fresh_idx] += lp;
            model_log_probs[level_seg_offset + fresh_idx] += lp;
        }
        __syncthreads();
    }
}


/*
 * Phase 5H-2: in-kernel segmented PTW DP state advance.
 *
 * Walks the per-chunk event loop serially per output node and produces
 * per-segment seed tensors (seg_nu0, seg_w0, seg_b0).  Mutates the
 * caller's [L, N] DP state buffers in-place to the post-chunk values.
 *
 * Inputs:
 *   cum_padded       : [L, B+1, N] float64
 *                      cum_padded[j, k, n] = sum_{i=0..k-1} log_q[j, i, n]
 *                      (padded so cum_padded[:, 0, :] == 0)
 *   event_offsets    : [E]   int32  sorted (unique offsets)
 *   event_close_mask : [E,L] bool   level j closes at event e
 *   event_seg_idx    : [E]   int32  seg index whose seeds get recorded
 *                                   immediately BEFORE this event's state
 *                                   mutations (or -1 if the event sits at
 *                                   offset == previous seg_start)
 *   trailing_seg_idx : int32        seg index for the post-last-event
 *                                   trailing segment (or -1 if none)
 *   state_nu, state_w, state_b : [L, N] float64 in+out
 *   seg_nu0, seg_w0, seg_b0    : [S, L, N] float64 OUT
 *
 * Block layout: one block per node (gridDim.x == N), single thread per
 * block.  L is small (paper depth=15), so per-node work is ~E*L scalar
 * doubles in registers -- microseconds per block.  Memory bandwidth on
 * cum_padded is the dominant cost; the prior Python event loop paid
 * ~340 small CUDA dispatches per call to scatter the same data.
 *
 * Maintains the line-5 / line-8 ordering of Algorithm 1: b[i] <- w[i+1]
 * for i = (lowest closing level) - 1 fires BEFORE zeroing nu/w/b at the
 * closing levels.
 */
#define ML_SEG_MAX_L 32

__global__ void fmn_mixture_multilevel_segmented_dp_state_advance(
    const double* __restrict__ cum_padded,
    const int*    __restrict__ event_offsets,
    const bool*   __restrict__ event_close_mask,
    const int*    __restrict__ event_seg_idx,
    int trailing_seg_idx,
    double* __restrict__ state_nu,
    double* __restrict__ state_w,
    double* __restrict__ state_b,
    double* __restrict__ seg_nu0,
    double* __restrict__ seg_w0,
    double* __restrict__ seg_b0,
    int B, int L, int N, int E
) {
    int n = blockIdx.x;
    if (n >= N) return;
    if (threadIdx.x != 0) return;

    const double half = -0.6931471805599453;  // log(0.5)

    double s_nu[ML_SEG_MAX_L];
    double s_w[ML_SEG_MAX_L];
    double s_b[ML_SEG_MAX_L];

    // Load initial state.
    for (int j = 0; j < L; j++) {
        s_nu[j] = state_nu[j * N + n];
        s_w[j]  = state_w[j * N + n];
        s_b[j]  = state_b[j * N + n];
    }

    int seg_start = 0;

    for (int e = 0; e < E; e++) {
        int off = event_offsets[e];
        int rec = event_seg_idx[e];
        if (rec >= 0) {
            // Record current state as this segment's seed.
            for (int j = 0; j < L; j++) {
                seg_nu0[(rec * L + j) * N + n] = s_nu[j];
                seg_w0[(rec * L + j) * N + n]  = s_w[j];
                seg_b0[(rec * L + j) * N + n]  = s_b[j];
            }
            // Advance nu by [seg_start, off) cumsum.
            for (int j = 0; j < L; j++) {
                double diff = cum_padded[(j * (B + 1) + off) * N + n]
                            - cum_padded[(j * (B + 1) + seg_start) * N + n];
                s_nu[j] += diff;
            }
            // Recompute w bottom-up against the segment-constant b.
            s_w[L - 1] = s_nu[L - 1];
            for (int j = L - 2; j >= 0; j--) {
                double a  = half + s_nu[j];
                double c  = half + s_w[j + 1] + s_b[j];
                double mx = fmax(a, c);
                s_w[j]    = mx + log(exp(a - mx) + exp(c - mx));
            }
            seg_start = off;
        }
        // Apply close-event mutations: line 5 then line 8.
        int i_plus_one = -1;
        for (int j = 0; j < L; j++) {
            if (event_close_mask[e * L + j]) { i_plus_one = j; break; }
        }
        if (i_plus_one > 0) {
            s_b[i_plus_one - 1] = s_w[i_plus_one];
        }
        for (int j = 0; j < L; j++) {
            if (event_close_mask[e * L + j]) {
                s_nu[j] = 0.0;
                s_w[j]  = 0.0;
                s_b[j]  = 0.0;
            }
        }
    }

    // Trailing segment.
    if (trailing_seg_idx >= 0) {
        int rec = trailing_seg_idx;
        for (int j = 0; j < L; j++) {
            seg_nu0[(rec * L + j) * N + n] = s_nu[j];
            seg_w0[(rec * L + j) * N + n]  = s_w[j];
            seg_b0[(rec * L + j) * N + n]  = s_b[j];
        }
        // Advance through [seg_start, B).
        for (int j = 0; j < L; j++) {
            double diff = cum_padded[(j * (B + 1) + B) * N + n]
                        - cum_padded[(j * (B + 1) + seg_start) * N + n];
            s_nu[j] += diff;
        }
        s_w[L - 1] = s_nu[L - 1];
        for (int j = L - 2; j >= 0; j--) {
            double a  = half + s_nu[j];
            double c  = half + s_w[j + 1] + s_b[j];
            double mx = fmax(a, c);
            s_w[j]    = mx + log(exp(a - mx) + exp(c - mx));
        }
    }

    // Commit final state back to global.
    for (int j = 0; j < L; j++) {
        state_nu[j * N + n] = s_nu[j];
        state_w[j * N + n]  = s_w[j];
        state_b[j * N + n]  = s_b[j];
    }
}

static int ml_choose_threads(int k) {
    int threads = 1;
    int target = k < 256 ? k : 256;
    while (threads < target) threads <<= 1;
    return threads;
}

/*
 * Python entry points.  Mixture-weights tensor MUST be [L, N, M, C, K_in]
 * contiguous and float32 CUDA; pool_sizes MUST be [L, N] int32 CUDA;
 * segment_log_probs and model_log_probs MUST be [L, N, M] float32 CUDA.
 * Predictions are returned as [L, B, N] float32 CUDA.
 */
torch::Tensor fmn_mixture_multilevel_forward_update_py(
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
    int L = mixture_weights.size(0);
    int N = mixture_weights.size(1);
    int M = mixture_weights.size(2);
    int C = mixture_weights.size(3);
    int H = hyperplanes.size(1);

    auto predictions = torch::empty({L, B, N}, z_batch.options());
    int threads = ml_choose_threads(K_in);
    int smem = (K_in + 2 * M + threads) * sizeof(float);

    fmn_mixture_multilevel_forward_update<<<L * N, threads, smem>>>(
        z_batch.data_ptr<float>(), p_prev_batch.data_ptr<float>(),
        symbols.data_ptr<int>(), mixture_weights.data_ptr<float>(),
        hyperplanes.data_ptr<float>(), hp_bias.data_ptr<float>(),
        pool_sizes.data_ptr<int>(), predictions.data_ptr<float>(),
        segment_log_probs.data_ptr<float>(), model_log_probs.data_ptr<float>(),
        lr, posterior_temp, B, L, N, M, C, K_in, D, H
    );
    return predictions;
}

torch::Tensor fmn_mixture_multilevel_forward_only_py(
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
    int L = mixture_weights.size(0);
    int N = mixture_weights.size(1);
    int M = mixture_weights.size(2);
    int C = mixture_weights.size(3);
    int H = hyperplanes.size(1);

    auto predictions = torch::empty({L, B, N}, z_batch.options());
    int threads = ml_choose_threads(K_in);
    int smem = (K_in + 2 * M + threads) * sizeof(float);

    fmn_mixture_multilevel_forward_only<<<L * N, threads, smem>>>(
        z_batch.data_ptr<float>(), p_prev_batch.data_ptr<float>(),
        mixture_weights.data_ptr<float>(), hyperplanes.data_ptr<float>(),
        hp_bias.data_ptr<float>(), pool_sizes.data_ptr<int>(),
        segment_log_probs.data_ptr<float>(),
        predictions.data_ptr<float>(),
        posterior_temp,
        B, L, N, M, C, K_in, D, H
    );
    return predictions;
}

torch::Tensor fmn_mixture_multilevel_forward_update_with_resets_py(
    torch::Tensor z_batch,
    torch::Tensor p_prev_batch,
    torch::Tensor symbols,
    torch::Tensor mixture_weights,
    torch::Tensor pool_snapshots,
    torch::Tensor hyperplanes,
    torch::Tensor hp_bias,
    torch::Tensor pool_sizes,
    torch::Tensor close_mask,
    torch::Tensor segment_log_probs,
    torch::Tensor model_log_probs,
    float lr,
    float posterior_temp
) {
    int B = z_batch.size(0);
    int D = z_batch.size(1);
    int K_in = p_prev_batch.size(1);
    int L = mixture_weights.size(0);
    int N = mixture_weights.size(1);
    int M = mixture_weights.size(2);
    int C = mixture_weights.size(3);
    int H = hyperplanes.size(1);
    int pool_capacity = pool_snapshots.size(1);

    auto predictions = torch::empty({L, B, N}, z_batch.options());
    int threads = ml_choose_threads(K_in);
    int smem = (K_in + 2 * M + threads) * sizeof(float);

    fmn_mixture_multilevel_forward_update_with_resets<<<L * N, threads, smem>>>(
        z_batch.data_ptr<float>(), p_prev_batch.data_ptr<float>(),
        symbols.data_ptr<int>(), mixture_weights.data_ptr<float>(),
        pool_snapshots.data_ptr<float>(),
        hyperplanes.data_ptr<float>(), hp_bias.data_ptr<float>(),
        pool_sizes.data_ptr<int>(), close_mask.data_ptr<bool>(),
        predictions.data_ptr<float>(),
        segment_log_probs.data_ptr<float>(), model_log_probs.data_ptr<float>(),
        lr, posterior_temp, B, L, N, M, C, K_in, D, H, pool_capacity
    );
    return predictions;
}

std::vector<torch::Tensor> fmn_mixture_multilevel_segmented_dp_state_advance_py(
    torch::Tensor cum_padded,         // [L, B+1, N] float64 CUDA
    torch::Tensor event_offsets,      // [E] int32 CUDA
    torch::Tensor event_close_mask,   // [E, L] bool CUDA
    torch::Tensor event_seg_idx,      // [E] int32 CUDA
    int64_t trailing_seg_idx,
    torch::Tensor state_nu,           // [L, N] float64 CUDA in+out
    torch::Tensor state_w,            // [L, N] float64 CUDA in+out
    torch::Tensor state_b,            // [L, N] float64 CUDA in+out
    int64_t num_segments
) {
    TORCH_CHECK(cum_padded.dtype() == torch::kFloat64,
                "cum_padded must be float64");
    TORCH_CHECK(state_nu.dtype() == torch::kFloat64
             && state_w.dtype()  == torch::kFloat64
             && state_b.dtype()  == torch::kFloat64,
                "state_* must be float64");
    TORCH_CHECK(event_offsets.dtype() == torch::kInt32,
                "event_offsets must be int32");
    TORCH_CHECK(event_seg_idx.dtype() == torch::kInt32,
                "event_seg_idx must be int32");
    TORCH_CHECK(event_close_mask.dtype() == torch::kBool,
                "event_close_mask must be bool");

    int L = cum_padded.size(0);
    int B = cum_padded.size(1) - 1;
    int N = cum_padded.size(2);
    int E = event_offsets.size(0);
    int S = (int)num_segments;

    TORCH_CHECK(L <= ML_SEG_MAX_L,
                "active_level_count exceeds kernel ML_SEG_MAX_L=32");

    auto opts = state_nu.options();
    auto seg_nu0 = torch::empty({S, L, N}, opts);
    auto seg_w0  = torch::empty({S, L, N}, opts);
    auto seg_b0  = torch::empty({S, L, N}, opts);

    if (N > 0 && E > 0) {
        fmn_mixture_multilevel_segmented_dp_state_advance<<<N, 1>>>(
            cum_padded.data_ptr<double>(),
            event_offsets.data_ptr<int>(),
            event_close_mask.data_ptr<bool>(),
            event_seg_idx.data_ptr<int>(),
            (int)trailing_seg_idx,
            state_nu.data_ptr<double>(),
            state_w.data_ptr<double>(),
            state_b.data_ptr<double>(),
            seg_nu0.data_ptr<double>(),
            seg_w0.data_ptr<double>(),
            seg_b0.data_ptr<double>(),
            B, L, N, E
        );
    }
    return {seg_nu0, seg_w0, seg_b0};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_update", &fmn_mixture_multilevel_forward_update_py,
          "FMN multilevel paper-uniform forward + update all levels/active models (CUDA)");
    m.def("forward_only", &fmn_mixture_multilevel_forward_only_py,
          "FMN multilevel paper-uniform forward only (CUDA)");
    m.def("forward_update_with_resets", &fmn_mixture_multilevel_forward_update_with_resets_py,
          "FMN multilevel forward + update with in-kernel close-only resets (CUDA)");
    m.def("segmented_dp_state_advance", &fmn_mixture_multilevel_segmented_dp_state_advance_py,
          "Phase 5H-2: in-kernel segmented PTW DP state advance (CUDA)");
}
