"""NCTL Network using CUDA FMN/GGM mixture kernels.

The active path implements the NCTL paper's node-level idea: each GGM node is
wrapped by an FMN-style model pool.  A layer keeps immutable pool snapshots plus
active per-segment copies of those snapshots.  During a segment, all active
models are updated and their current-segment likelihoods define the Bayesian
posterior used for the next prediction.  At a segment boundary, the best active
model is stored back into the bounded pool and the next segment is re-opened
from the snapshots plus a fresh zero model.
"""

import math
import os
from functools import wraps
from pathlib import Path
from typing import Any

import torch

from nctl_bench._profile import prof

POOL_EVICT_POLICIES = {
    "fifo",
    "task-floor",
    "age-diversity",
    "age-bucket-floor",
    "age-diversity-oldest-floor",
}

# Default number of oldest snapshots protected by "age-diversity-oldest-floor".
DEFAULT_POOL_OLDEST_FLOOR = 2


def _profiled(section: str):
    """Decorator that wraps a bound method in ``with prof(section):``.

    Phase 5H-1 instrumentation hook.  When the global profiler is disabled
    (the default), ``prof()`` is a zero-overhead no-op so decorated methods
    pay nothing.  When enabled, the per-call latency is accumulated into
    ``section`` under the ``_profile`` singleton.
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with prof(section):
                return fn(*args, **kwargs)
        return wrapper
    return deco

# Load CUDA kernels from pre-compiled cache.
_CACHE_DIR = os.path.expanduser("~/.cache/torch_extensions/py310_cu128")
os.environ.setdefault("CUDA_HOME", "/usr")

_ggm_cuda = None
_fmn_cuda = None


def _load_cached(name: str):
    import importlib.util

    so = os.path.join(_CACHE_DIR, name, f"{name}.so")
    if not os.path.exists(so):
        raise RuntimeError(
            f"Pre-compiled kernel not found: {so}\n"
            f"Run: bash scripts/nctl_bench/cuda/build_kernels.sh"
        )
    spec = importlib.util.spec_from_file_location(name, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_ggm_cuda():
    global _ggm_cuda
    if _ggm_cuda is None:
        _ggm_cuda = _load_cached("ggm_cuda")
    return _ggm_cuda


def _get_fmn_cuda():
    global _fmn_cuda
    if _fmn_cuda is None:
        _fmn_cuda = _load_cached("fmn_mixture_cuda")
    return _fmn_cuda


_fmn_multilevel_cuda = None
_fmn_multilevel_cuda_missing = False


def _get_fmn_multilevel_cuda():
    """Optional Phase 5E fused per-level kernel.

    Returns the module if the .so is built, else None.  Callers MUST be
    prepared for None: the per-level loop fallback in the DP path is the
    authoritative correctness baseline.  Caching the missing-state means
    we only pay the import attempt once per process.
    """
    global _fmn_multilevel_cuda, _fmn_multilevel_cuda_missing
    if _fmn_multilevel_cuda is not None:
        return _fmn_multilevel_cuda
    if _fmn_multilevel_cuda_missing:
        return None
    try:
        _fmn_multilevel_cuda = _load_cached("fmn_mixture_multilevel_cuda")
    except RuntimeError:
        _fmn_multilevel_cuda_missing = True
        return None
    return _fmn_multilevel_cuda


def _fmn_multilevel_paper_posterior_reference(
    z_batch: torch.Tensor,
    p_prev_batch: torch.Tensor,
    symbols: torch.Tensor | None,
    mixture_weights: torch.Tensor,
    hyperplanes: torch.Tensor,
    hp_bias: torch.Tensor,
    pool_sizes: torch.Tensor,
    segment_log_probs: torch.Tensor,
    model_log_probs: torch.Tensor | None,
    lr: float,
    update: bool,
    posterior_temp: float = 0.0,
) -> torch.Tensor:
    """Pure-PyTorch reference for ``fmn_mixture_multilevel_kernel.cu``.

    Implements the FMN paper Eq. 5 conditional for every PTW level and node.
    With ``posterior_temp=1.0`` this is the Bayes conditional ratio induced by
    the joint mixture: prior 1/2 on the fresh base measure and `(1/2)/pool_size`
    on each remembered slot, multiplied by per-slot current-segment likelihood.
    With ``posterior_temp=0.0`` it deliberately collapses to the unweighted
    1/2 fresh + 1/2 pool-mean mixture for regression parity.  Optionally
    applies the per-slot weight updates that the kernel's training path does.
    Used for CPU regression tests of the Python wrapper independent of CUDA.

    Shapes:
        z_batch          : [B, D]
        p_prev_batch     : [B, K_in]
        symbols          : [B] int (required when ``update=True``)
        mixture_weights  : [L, N, M, C, K_in]
        hyperplanes      : [N, H, D]
        hp_bias          : [N, H]
        pool_sizes       : [L, N] int
        segment_log_probs: [L, N, M]
        model_log_probs  : [L, N, M] (required when ``update=True``)

    Returns predictions ``[L, B, N]`` matching the kernel's output:
        - update=True returns ν_j(x_obs | x_<t)
        - update=False returns ν_j(x_t=1 | x_<t)
    """
    EPS = 1e-7
    LOGIT_CLIP = 15.0
    MAX_WEIGHT = 200.0
    LOG_MIN = 1e-30
    L, N, M, C, K_in = mixture_weights.shape
    B, D = z_batch.shape
    H = hyperplanes.shape[1]
    fresh_idx = M - 1
    if update and (symbols is None or model_log_probs is None):
        raise ValueError("update=True requires symbols and model_log_probs")
    predictions = torch.zeros(L, B, N, dtype=z_batch.dtype, device=z_batch.device)
    for b in range(B):
        z = z_batch[b]               # [D]
        p_prev = p_prev_batch[b]     # [K_in]
        sym = int(symbols[b].item()) if (update and symbols is not None) else 1
        # Context (level-independent): one int per node.
        dots = (hyperplanes * z.view(1, 1, D)).sum(dim=-1) + hp_bias  # [N, H]
        context = torch.zeros(N, dtype=torch.long, device=z.device)
        for h in range(H):
            context |= ((dots[:, h] >= 0.0).long() << h)
        logits = torch.log(torch.clamp(p_prev, EPS, 1.0 - EPS) / torch.clamp(1.0 - p_prev, EPS, 1.0 - EPS))
        logits = torch.clamp(logits, -LOGIT_CLIP, LOGIT_CLIP)  # [K_in]
        for l in range(L):
            for n in range(N):
                ctx = int(context[n].item())
                k_pool = int(pool_sizes[l, n].item())
                # Per-slot raw predictions for this (level, node, context).
                slot_w = mixture_weights[l, n, :, ctx, :]  # [M, K_in]
                model_preds = torch.sigmoid(slot_w @ logits)  # [M]
                # Paper-faithful mixture: Bayesian when posterior_temp>0,
                # collapses to 1/2 fresh + 1/2 pool-mean at temp=0.
                p_mixture = _posterior_mixture_prob(
                    model_preds, segment_log_probs[l, n], k_pool, fresh_idx,
                    posterior_temp,
                )
                if update:
                    predictions[l, b, n] = p_mixture if sym == 1 else (1.0 - p_mixture)
                else:
                    predictions[l, b, n] = p_mixture
                if update:
                    # Update active pool slots and fresh slot.
                    error = float(sym) - model_preds  # [M]
                    # Only slots 0..k_pool-1 and fresh_idx are active.
                    active_mask = torch.zeros(M, dtype=torch.bool, device=z.device)
                    active_mask[:k_pool] = True
                    active_mask[fresh_idx] = True
                    delta = lr * error.unsqueeze(-1) * logits.unsqueeze(0)  # [M, K_in]
                    new_w = mixture_weights[l, n, :, ctx, :] + delta
                    new_w = torch.clamp(new_w, -MAX_WEIGHT, MAX_WEIGHT)
                    # Only write back active slots.
                    new_full = mixture_weights[l, n, :, ctx, :].clone()
                    new_full[active_mask] = new_w[active_mask]
                    mixture_weights[l, n, :, ctx, :] = new_full
                    # Accumulate log-likelihoods.
                    p_sym = torch.where(
                        torch.tensor(sym == 1, device=z.device),
                        model_preds,
                        1.0 - model_preds,
                    )
                    lp = torch.log(torch.clamp(p_sym, min=LOG_MIN))
                    segment_log_probs[l, n] = torch.where(
                        active_mask, segment_log_probs[l, n] + lp, segment_log_probs[l, n]
                    )
                    if model_log_probs is not None:
                        model_log_probs[l, n] = torch.where(
                            active_mask, model_log_probs[l, n] + lp, model_log_probs[l, n]
                        )
    return predictions


def _fmn_multilevel_paper_posterior_reference_with_resets(
    z_batch: torch.Tensor,
    p_prev_batch: torch.Tensor,
    symbols: torch.Tensor,
    mixture_weights: torch.Tensor,
    pool_snapshots: torch.Tensor,
    hyperplanes: torch.Tensor,
    hp_bias: torch.Tensor,
    pool_sizes: torch.Tensor,
    close_mask: torch.Tensor,
    segment_log_probs: torch.Tensor,
    model_log_probs: torch.Tensor,
    lr: float,
    posterior_temp: float = 0.0,
) -> torch.Tensor:
    """Pure-PyTorch reference for the Phase 5F ``forward_update_with_resets``
    multilevel kernel.

    Identical to ``_fmn_multilevel_paper_posterior_reference(update=True)``
    except that BEFORE processing sample ``b`` at level ``l`` we honour
    ``close_mask[l, b]``:

    * Restore active slots [0..k_pool-1] from ``pool_snapshots``.  The
      shared pool size is invariant across the sub-chunk (UPDATEMODELPOOL
      fires only between sub-chunks), so ``pool_sizes[l, n]`` already
      reflects it and we use that directly.
    * Zero unused slots [k_pool..M-2].
    * Zero ``segment_log_probs[l, n, :]`` for every slot.
    * Fresh slot weights (M-1) are preserved (FMN base measure ρ).
    * The fresh slot's segment_log_probs is reset along with the rest
      (paper FMN posterior priors honoured at segment open).

    Inputs:
        z_batch        : [B, D]
        p_prev_batch   : [B, K_in]
        symbols        : [B] int
        mixture_weights: [L, N, M, C, K_in]   (mutated)
        pool_snapshots : [N, pool_capacity, C, K_in]
        hyperplanes    : [N, H, D]
        hp_bias        : [N, H]
        pool_sizes     : [L, N] int
        close_mask     : [L, B] bool
        segment_log_probs / model_log_probs: [L, N, M]  (mutated)

    Returns predictions [L, B, N] of ν_j(x_obs | x_<t).
    """
    EPS = 1e-7
    LOGIT_CLIP = 15.0
    MAX_WEIGHT = 200.0
    LOG_MIN = 1e-30
    L, N, M, C, K_in = mixture_weights.shape
    B, D = z_batch.shape
    H = hyperplanes.shape[1]
    fresh_idx = M - 1
    pool_capacity = pool_snapshots.shape[1]
    if close_mask.shape != (L, B):
        raise ValueError(
            f"close_mask shape {tuple(close_mask.shape)} != ({L}, {B})"
        )
    predictions = torch.zeros(L, B, N, dtype=z_batch.dtype, device=z_batch.device)
    for b in range(B):
        # Apply per-(level, node) close-only resets at this offset.
        # Vectorised across nodes for each level.
        for l in range(L):
            if not bool(close_mask[l, b].item()):
                continue
            # Per-node reset (the kernel does this in cooperating threads).
            for n in range(N):
                k_pool = int(pool_sizes[l, n].item())
                # Slots [0..k_pool-1] <- pool_snapshots[n, 0..k_pool-1]
                if k_pool > 0:
                    mixture_weights[l, n, :k_pool] = pool_snapshots[n, :k_pool]
                # Slots [k_pool..fresh_idx-1] <- 0
                if pool_capacity > k_pool:
                    mixture_weights[l, n, k_pool:fresh_idx] = 0.0
                # All slots' segment evidence zeroed.
                segment_log_probs[l, n].zero_()
        # Now run the standard forward+update step for sample b.
        z = z_batch[b]
        p_prev = p_prev_batch[b]
        sym = int(symbols[b].item())
        dots = (hyperplanes * z.view(1, 1, D)).sum(dim=-1) + hp_bias
        context = torch.zeros(N, dtype=torch.long, device=z.device)
        for h in range(H):
            context |= ((dots[:, h] >= 0.0).long() << h)
        logits = torch.log(torch.clamp(p_prev, EPS, 1.0 - EPS) / torch.clamp(1.0 - p_prev, EPS, 1.0 - EPS))
        logits = torch.clamp(logits, -LOGIT_CLIP, LOGIT_CLIP)
        for l in range(L):
            for n in range(N):
                ctx = int(context[n].item())
                k_pool = int(pool_sizes[l, n].item())
                slot_w = mixture_weights[l, n, :, ctx, :]
                model_preds = torch.sigmoid(slot_w @ logits)
                p_mixture = _posterior_mixture_prob(
                    model_preds, segment_log_probs[l, n], k_pool, fresh_idx,
                    posterior_temp,
                )
                predictions[l, b, n] = p_mixture if sym == 1 else (1.0 - p_mixture)
                # Update active slots + fresh slot.
                error = float(sym) - model_preds
                active_mask = torch.zeros(M, dtype=torch.bool, device=z.device)
                active_mask[:k_pool] = True
                active_mask[fresh_idx] = True
                delta = lr * error.unsqueeze(-1) * logits.unsqueeze(0)
                new_w = mixture_weights[l, n, :, ctx, :] + delta
                new_w = torch.clamp(new_w, -MAX_WEIGHT, MAX_WEIGHT)
                new_full = mixture_weights[l, n, :, ctx, :].clone()
                new_full[active_mask] = new_w[active_mask]
                mixture_weights[l, n, :, ctx, :] = new_full
                p_sym = torch.where(
                    torch.tensor(sym == 1, device=z.device),
                    model_preds,
                    1.0 - model_preds,
                )
                lp = torch.log(torch.clamp(p_sym, min=LOG_MIN))
                segment_log_probs[l, n] = torch.where(
                    active_mask, segment_log_probs[l, n] + lp, segment_log_probs[l, n]
                )
                model_log_probs[l, n] = torch.where(
                    active_mask, model_log_probs[l, n] + lp, model_log_probs[l, n]
                )
    return predictions


def _posterior_mixture_prob(
    model_preds: torch.Tensor,
    segment_log_probs: torch.Tensor,
    pool_size: int,
    fresh_idx: int,
    posterior_temp: float = 1.0,
) -> torch.Tensor:
    """Bayesian predictive mixture for one node.

    FMN defines a segment model as a Bayesian mixture over the fresh/base model
    and remembered model states.  The conditional predictive probability is the
    posterior-weighted average of the active models' conditional predictions;
    the posterior uses likelihood accumulated on the *current segment*.

    Priors follow the FMN ν_t construction: when the pool is non-empty, the fresh
    model has prior 1/2 and the pool half is split uniformly over active pool
    models.  When the pool is empty, the fresh model receives all mass.

    ``posterior_temp`` scales the current-segment log-likelihood term in the
    posterior softmax (priors are not scaled).  ``posterior_temp=1.0``
    reproduces the standard Bayesian conditional; lower values combat the
    posterior saturation observed in long FMN segments where one model's
    accumulated log-likelihood dominates the mixture.
    """
    if pool_size <= 0:
        return model_preds[fresh_idx]

    active = list(range(pool_size)) + [fresh_idx]
    log_lik = segment_log_probs[active].clone() * posterior_temp
    log_lik[:pool_size] += math.log(0.5) - math.log(float(pool_size))
    log_lik[pool_size] += math.log(0.5)
    weights = torch.softmax(log_lik, dim=0)
    return (weights * model_preds[active]).sum()


def mscb(depth: int, t: int) -> int:
    """Most significant changed bit from Algorithm 1 of the FMN paper."""
    if t <= 1:
        return 0
    changed = (t - 1) ^ (t - 2)
    if changed == 0:
        return depth
    bit = changed.bit_length() - 1  # least-significant bit is position 0.
    return max(0, depth - 1 - bit)


def mscb_close_levels(depth: int, t: int, min_segment: int | None = None) -> list[int]:
    """PTW levels whose segments close before processing 1-based time ``t``.

    Algorithm 1 lines 4--8 close levels ``j = MSCB_d(t)+1 .. d`` for the PTW
    recursion *and* for ``UPDATEMODELPOOL`` invocations.  The paper's optional
    complexity-reducing ``2**c`` skip (lines 415--417 of the FMN paper) applies
    only to ``UPDATEMODELPOOL`` calls, not to the PTW close/reset itself.

    This function therefore returns the **full** paper close set with no
    filtering by default; pass ``min_segment`` only to recover the legacy
    update-eligible subset for back-compatibility with callers/tests that
    historically conflated the two.  New code should call this function with
    ``min_segment=None`` and then filter via :func:`mscb_update_eligible_levels`
    when an update-eligible subset is needed.
    """
    if t <= 1:
        return []
    i = mscb(depth, t)
    levels: list[int] = list(range(i + 1, depth + 1))
    if min_segment is None or int(min_segment) <= 1:
        return levels
    return mscb_update_eligible_levels(levels, depth, min_segment)


def mscb_update_eligible_levels(
    close_levels: list[int], depth: int, min_segment: int
) -> list[int]:
    """Filter close levels by the FMN paper's ``2**c`` UPDATEMODELPOOL skip.

    The paper (FMN §3.3, lines 415--417) allows skipping ``UPDATEMODELPOOL``
    when the closing segment length ``b - a + 1`` is below ``2**c`` for some
    ``c < d``.  Mapping ``min_segment = 2**c`` and segment length
    ``2**(depth - j)`` gives the level-wise condition below.

    Levels whose closing segment length falls below ``min_segment`` are still
    closed by the PTW recursion (and their per-level state reset) but are not
    eligible to call ``UPDATEMODELPOOL``.
    """
    threshold = max(1, int(min_segment))
    if threshold <= 1:
        return list(close_levels)
    return [j for j in close_levels if (1 << (int(depth) - int(j))) >= threshold]


def mscb_boundary_plan(
    current_index: int, batch_len: int, depth: int, min_segment: int = 1
) -> list[tuple[int, list[int], list[int]]]:
    """Return ``(offset, close_levels, update_levels)`` triples for a batch.

    ``current_index`` is the number of already processed samples.  An event at
    offset 0 must be applied before the first sample in this batch.

    ``close_levels`` lists every PTW level closing before this sample (paper
    Algorithm 1 lines 4--6); ``update_levels`` is the subset of those eligible
    to call ``UPDATEMODELPOOL`` under the ``2**c`` (``min_segment``) skip
    heuristic (paper §3.3 lines 415--417).  At ``min_segment <= 1`` the two
    lists are identical.
    """
    events: list[tuple[int, list[int], list[int]]] = []
    for offset in range(batch_len):
        t = current_index + offset + 1
        close_levels = mscb_close_levels(depth, t)
        if not close_levels:
            continue
        update_levels = mscb_update_eligible_levels(close_levels, depth, min_segment)
        events.append((offset, close_levels, update_levels))
    return events


def mscb_update_split_plan(
    current_index: int,
    batch_len: int,
    depth: int,
    min_segment: int = 1,
) -> "list[tuple[int, int, list[int], list[list[int]]]]":
    """Plan a chunk's processing as sub-chunks split only on UPDATEMODELPOOL events.

    Returns a list of ``(start, end, boundary_close_levels, boundary_update_levels, close_only_events)``:

    * ``start..end`` is a half-open range of *local* sample offsets inside
      the chunk (i.e. independent of ``current_index``).
    * ``boundary_close_levels`` is the full PTW close set firing AFTER
      sample ``end-1`` (or ``[]`` for the final sub-chunk, which has no
      trailing boundary event).
    * ``boundary_update_levels`` is the update-eligible subset of
      ``boundary_close_levels`` -- i.e. the levels at the boundary that
      must also invoke ``UPDATEMODELPOOL``.  ``segment_close`` consumes
      both.
    * ``close_only_events`` is the list of ``(local_offset, close_levels)``
      pairs for pure-close-only events (no UPDATEMODELPOOL) firing BEFORE
      each sample in the sub-chunk.  ``local_offset == 0`` means the close
      fires before the first sample of the sub-chunk.

    The contract is paper-faithful: every PTW close that
    ``mscb_close_levels`` would emit is still represented, just regrouped
    into batched sub-chunks plus per-sample close-only masks.
    """
    sub_chunks: list[tuple[int, int, list[int], list[int], list[tuple[int, list[int]]]]] = []
    start = 0
    close_only_events: list[tuple[int, list[int]]] = []
    for offset in range(batch_len):
        t = current_index + offset + 1
        close_levels = mscb_close_levels(depth, t)
        if not close_levels:
            continue
        update_levels = mscb_update_eligible_levels(close_levels, depth, min_segment)
        if update_levels:
            # Cut here: close-only mask covers samples start..offset-1
            # (events with offset == \`start\` fire before the first sample of
            # the new sub-chunk).  The trailing UPDATE event itself fires
            # AFTER sample offset-1 -- i.e. between this sub-chunk and the
            # next -- and is recorded in update_levels for the caller to
            # apply via the existing segment_close path.
            sub_chunks.append((start, offset, list(close_levels), list(update_levels), list(close_only_events)))
            start = offset
            close_only_events = []
        else:
            close_only_events.append((offset, list(close_levels)))
    sub_chunks.append((start, batch_len, [], [], list(close_only_events)))
    return sub_chunks


def build_close_mask(
    close_only_events: "list[tuple[int, list[int]]]",
    active_level_count: int,
    sub_chunk_len: int,
    device: "torch.device | str" = "cpu",
) -> torch.Tensor:
    """Materialise close_only_events into a [L, sub_chunk_len] bool tensor.

    ``close_mask[l, b] = True`` => level l\'s per-level FMN active state
    (mixture_weights[l], segment_log_probs[l], etc.) AND PTW DP state
    (level_log_nu[l], ptw_log_w[l], ptw_log_b[...]) must be reset BEFORE
    processing sample b of the sub-chunk.

    Levels above active_level_count - 1 are silently dropped (they
    correspond to depths that the layer does not maintain per-level state
    for at runtime).
    """
    mask = torch.zeros(active_level_count, sub_chunk_len, dtype=torch.bool, device=device)
    for offset, close_levels in close_only_events:
        if offset < 0 or offset >= sub_chunk_len:
            continue
        for level in close_levels:
            if 0 <= level < active_level_count:
                mask[level, offset] = True
    return mask


class CudaFmnMixtureLayer:
    """A layer using an FMN mixture of per-node GGM weight snapshots.

    Tensor layout:
      * ``pool_snapshots``: immutable remembered states, [N, pool_capacity, C, K]
      * ``mixture_weights``: active per-segment copies plus fresh model,
        [N, pool_capacity + 1, C, K]
      * fresh/base slot is always ``pool_capacity`` (M-1), independent of the
        current pool size.

    The CUDA kernel updates ``mixture_weights`` and ``segment_log_probs``.  Python
    handles sparse segment-boundary bookkeeping.
    """

    def __init__(
        self,
        num_nodes: int,
        num_inputs: int,
        input_dim: int,
        num_halfspaces: int,
        lr: float,
        pool_capacity: int,
        device: torch.device,
        seed: int = 42,
        posterior_temp: float = 1.0,
        pool_update_policy: str = "fifo",
        pool_alpha: float = math.inf,
        pool_beta: float = math.inf,
        pool_reservoir_size: int = 64,
        pool_evict_policy: str = "fifo",
        pool_oldest_floor: int = DEFAULT_POOL_OLDEST_FLOOR,
        active_state_mode: str = "flat",
        active_level_count: int | None = None,
        prediction_level: int | None = None,
        prediction_mode: str = "selected_level",
    ):
        self.N = num_nodes
        self.K = num_inputs
        self.D = input_dim
        self.H = num_halfspaces
        self.C = 2**num_halfspaces
        self.M = pool_capacity + 1
        self.lr = lr
        self.device = device
        self.pool_capacity = pool_capacity
        self.posterior_temp = float(posterior_temp)
        if pool_update_policy not in {"fifo", "paper"}:
            raise ValueError(f"unknown pool_update_policy: {pool_update_policy}")
        self.pool_update_policy = pool_update_policy
        self.pool_alpha = float(pool_alpha)
        self.pool_beta = float(pool_beta)
        self.pool_reservoir_size = max(0, int(pool_reservoir_size))
        # Pool eviction policy controls which slot is overwritten when the pool
        # is full.  "fifo" always drops the oldest slot.  "task-floor" is a
        # diagnostic that leaks benchmark task identity.  The age-* policies
        # are task-free follow-ups that preserve temporal coverage using
        # insertion indices only.
        if pool_evict_policy not in POOL_EVICT_POLICIES:
            raise ValueError(
                f"unknown pool_evict_policy: {pool_evict_policy!r}"
            )
        self.pool_evict_policy = pool_evict_policy
        # Number of oldest (lowest-slot) snapshots protected by the
        # "age-diversity-oldest-floor" policy; inert for every other policy.
        self.pool_oldest_floor = max(0, int(pool_oldest_floor))
        if active_state_mode not in {"flat", "per_level"}:
            raise ValueError(f"unknown active_state_mode: {active_state_mode}")
        if prediction_mode not in {"selected_level", "ptw_dp"}:
            raise ValueError(f"unknown prediction_mode: {prediction_mode}")
        self.active_state_mode = active_state_mode
        self.per_level_active = active_state_mode == "per_level"
        self.active_level_count = max(1, int(active_level_count or 1))
        self.prediction_level = int(prediction_level or 0)
        # Paper Algorithm 1 bottom-up w_j/b_j PTW dynamic programming.  When
        # enabled, ``forward_update_batch_dp`` and ``forward_only_batch_dp``
        # mix per-level segment-conditional predictions via the FMN recursion
        # instead of returning a single selected level's output.  Requires
        # per-level active state (per_level_active=True).
        self.prediction_mode = prediction_mode
        if prediction_mode == "ptw_dp" and not self.per_level_active:
            raise ValueError(
                "prediction_mode='ptw_dp' requires active_state_mode='per_level'"
            )

        gen = torch.Generator(device="cpu").manual_seed(seed)
        hp = torch.randn(num_nodes, num_halfspaces, input_dim, generator=gen)
        hp = hp / hp.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self.hyperplanes = hp.to(device)
        self.hp_bias = torch.zeros(num_nodes, num_halfspaces, device=device)

        self.pool_snapshots = torch.zeros(
            num_nodes, pool_capacity, self.C, num_inputs, device=device
        )
        self.pool_levels = torch.full(
            (num_nodes, pool_capacity), -1, device=device, dtype=torch.int32
        )
        # Diagnostic metadata for bounded-pool provenance.  These tensors do
        # not affect prediction or training; they make FIFO churn and task
        # retention visible in benchmark JSON outputs.
        self.pool_task_ids = torch.full(
            (num_nodes, pool_capacity), -1, device=device, dtype=torch.int32
        )
        self.pool_insert_indices = torch.full(
            (num_nodes, pool_capacity), -1, device=device, dtype=torch.int64
        )
        self.provenance_event_counts: dict[str, int] = {
            "append": 0,
            "evict": 0,
            "refine": 0,
            "skip": 0,
        }
        self.provenance_evicted_task_counts: dict[int, int] = {}

        self._reservoir_rng = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)
        self.segment_res_seen = 0
        self.segment_res_size = 0
        self.segment_res_z = torch.empty(self.pool_reservoir_size, self.D, device="cpu")
        self.segment_res_p_prev = torch.empty(self.pool_reservoir_size, self.K, device="cpu")
        self.segment_res_symbols = torch.empty(self.pool_reservoir_size, dtype=torch.int32, device="cpu")
        self.pool_res_sizes = torch.zeros(num_nodes, pool_capacity, dtype=torch.int32, device="cpu")
        self.pool_res_z = torch.empty(num_nodes, pool_capacity, self.pool_reservoir_size, self.D, device="cpu")
        self.pool_res_p_prev = torch.empty(num_nodes, pool_capacity, self.pool_reservoir_size, self.K, device="cpu")
        self.pool_res_symbols = torch.empty(num_nodes, pool_capacity, self.pool_reservoir_size, dtype=torch.int32, device="cpu")

        self.mixture_weights = torch.zeros(
            num_nodes, self.M, self.C, num_inputs, device=device
        )
        self.pool_sizes = torch.zeros(num_nodes, device=device, dtype=torch.int32)
        # Current-segment log likelihoods used as Bayesian posterior logits.
        self.segment_log_probs = torch.zeros(num_nodes, self.M, device=device)
        # Lifetime diagnostics only; not used for posterior weighting.
        self.model_log_probs = torch.zeros(num_nodes, self.M, device=device)

        self.level_mixture_weights = None
        self.level_pool_sizes = None
        self.level_segment_log_probs = None
        self.level_model_log_probs = None
        self.level_res_seen = None
        self.level_res_size = None
        self.level_res_z = None
        self.level_res_p_prev = None
        self.level_res_symbols = None
        # PTW DP state per node, per PTW level j in 0..active_level_count-1:
        #   level_log_nu[j, n] = log nu_{r_j}(x_{r_j:t})   accumulated segment
        #                       log-marginal of level j's open segment.
        #   ptw_log_w[j, n]    = log w_j(x_{1:t})           bottom-up DP value.
        #   ptw_log_b[j, n]    = log b_j                    line-5 cache.
        # All three reset to 0 at every level-j close (line 8 sets w_j, b_j,
        # nu_{r_j} <- 1); ptw_log_b[i] is set to ptw_log_w[i+1] right BEFORE
        # the deeper resets fire (line 5 of Algorithm 1).
        self.level_log_nu = None
        self.ptw_log_w = None
        self.ptw_log_b = None
        if self.per_level_active:
            self.enable_per_level_active_state(
                active_level_count=self.active_level_count,
                prediction_level=self.prediction_level,
            )

    @property
    def fresh_idx(self) -> int:
        return self.pool_capacity

    def enable_per_level_active_state(
        self, active_level_count: int, prediction_level: int | None = None
    ) -> None:
        """Allocate independent active FMN state for each retained PTW level.

        The global bounded pool remains shared, but each PTW level keeps its
        own adapted active copies, current-segment posterior evidence, pool-size
        view at segment open, and segment reservoir.  This is the first
        paper-faithful step beyond the earlier flat active-candidate
        approximation.
        """
        level_count = max(1, int(active_level_count))
        pred_level = level_count - 1 if prediction_level is None else int(prediction_level)
        pred_level = max(0, min(pred_level, level_count - 1))
        self.active_state_mode = "per_level"
        self.per_level_active = True
        self.active_level_count = level_count
        self.prediction_level = pred_level
        self.level_mixture_weights = self.mixture_weights.unsqueeze(0).repeat(
            level_count, 1, 1, 1, 1
        )
        self.level_pool_sizes = self.pool_sizes.unsqueeze(0).repeat(level_count, 1)
        self.level_segment_log_probs = self.segment_log_probs.unsqueeze(0).repeat(
            level_count, 1, 1
        )
        self.level_model_log_probs = self.model_log_probs.unsqueeze(0).repeat(
            level_count, 1, 1
        )
        self.level_res_seen = torch.zeros(level_count, dtype=torch.int64, device="cpu")
        self.level_res_size = torch.zeros(level_count, dtype=torch.int64, device="cpu")
        self.level_res_z = torch.empty(
            level_count, self.pool_reservoir_size, self.D, device="cpu"
        )
        self.level_res_p_prev = torch.empty(
            level_count, self.pool_reservoir_size, self.K, device="cpu"
        )
        self.level_res_symbols = torch.empty(
            level_count, self.pool_reservoir_size, dtype=torch.int32, device="cpu"
        )
        # PTW DP state.  Float64 for numerical headroom in the long-running
        # log-prob accumulation; the per-sample arithmetic is dominated by
        # other costs so the doubled memory bandwidth is not a concern.
        self.level_log_nu = torch.zeros(level_count, self.N, dtype=torch.float64, device=self.device)
        self.ptw_log_w = torch.zeros(level_count, self.N, dtype=torch.float64, device=self.device)
        self.ptw_log_b = torch.zeros(level_count, self.N, dtype=torch.float64, device=self.device)

    def _level_index(self, level: int) -> int:
        if not self.per_level_active:
            return 0
        return max(0, min(int(level), self.active_level_count - 1))

    def _active_tensors(self, level: int | None = None):
        if not self.per_level_active:
            return self.mixture_weights, self.pool_sizes, self.segment_log_probs, self.model_log_probs
        idx = self._level_index(self.prediction_level if level is None else level)
        return (
            self.level_mixture_weights[idx],
            self.level_pool_sizes[idx],
            self.level_segment_log_probs[idx],
            self.level_model_log_probs[idx],
        )

    def forward_update_batch(
        self,
        z_batch,
        p_prev_batch,
        symbols_batch,
        close_events: "list[tuple[int, list[int]]] | None" = None,
    ):
        """Dispatch on ``prediction_mode``.

        * ``selected_level`` (default) returns the kernel output of the
          single ``prediction_level``'s active state.  This is the legacy
          path; per-level mode still launches the kernel once per level
          internally so the per-level active state evolves correctly.
        * ``ptw_dp`` mixes per-level conditional predictions via the FMN
          paper's Algorithm 1 bottom-up w_j/b_j recursion and returns
          ``P_0(x_obs | x_<t)`` per sample.  Requires
          ``per_level_active=True``.

        ``close_events`` is forwarded to ``forward_update_batch_dp`` when
        ``prediction_mode='ptw_dp'``.  selected_level mode ignores it; the
        caller is responsible for invoking ``segment_close`` at the right
        boundaries in that mode.
        """
        if self.prediction_mode == "ptw_dp":
            return self.forward_update_batch_dp(
                z_batch, p_prev_batch, symbols_batch, close_events=close_events,
            )
        return self._forward_update_batch_selected_level(
            z_batch, p_prev_batch, symbols_batch
        )

    def _forward_update_batch_selected_level(self, z_batch, p_prev_batch, symbols_batch):
        cuda = _get_fmn_cuda()
        if self.per_level_active:
            out = None
            for level in range(self.active_level_count):
                weights, pool_sizes, seg_lp, model_lp = self._active_tensors(level)
                pred = cuda.forward_update(
                    z_batch,
                    p_prev_batch,
                    symbols_batch,
                    weights,
                    self.hyperplanes,
                    self.hp_bias,
                    pool_sizes,
                    seg_lp,
                    model_lp,
                    self.lr,
                    self.posterior_temp,
                )
                if level == self.prediction_level:
                    out = pred
            assert out is not None
            return out
        return cuda.forward_update(
            z_batch,
            p_prev_batch,
            symbols_batch,
            self.mixture_weights,
            self.hyperplanes,
            self.hp_bias,
            self.pool_sizes,
            self.segment_log_probs,
            self.model_log_probs,
            self.lr,
            self.posterior_temp,
        )

    def forward_only_batch(self, z_batch, p_prev_batch):
        if self.prediction_mode == "ptw_dp":
            return self.forward_only_batch_dp(z_batch, p_prev_batch)
        return self._forward_only_batch_selected_level(z_batch, p_prev_batch)

    def _forward_only_batch_selected_level(self, z_batch, p_prev_batch):
        cuda = _get_fmn_cuda()
        if self.per_level_active:
            weights, pool_sizes, seg_lp, _ = self._active_tensors(self.prediction_level)
            return cuda.forward_only(
                z_batch,
                p_prev_batch,
                weights,
                self.hyperplanes,
                self.hp_bias,
                pool_sizes,
                seg_lp,
                self.posterior_temp,
            )
        return cuda.forward_only(
            z_batch,
            p_prev_batch,
            self.mixture_weights,
            self.hyperplanes,
            self.hp_bias,
            self.pool_sizes,
            self.segment_log_probs,
            self.posterior_temp,
        )

    # ------------------------------------------------------------------
    # Algorithm 1 bottom-up w_j/b_j PTW dynamic programming (Phase 2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ptw_dp_combine(self, q1_per_level: torch.Tensor) -> torch.Tensor:
        """Combine per-level conditional q_j(x_t=1|x_<t) into P_0(x_t=1|x_<t).

        Uses the closed-form convex combination derived from Algorithm 1:

            P_d = q_d(1)
            P_j = pi_j * q_j(1) + (1 - pi_j) * P_{j+1}     for j < d

        with ``pi_j = exp(log(0.5) + level_log_nu[j] - ptw_log_w[j])`` evaluated
        on the DP state BEFORE observing x_t.

        ``q1_per_level`` may be ``[L, N]`` (single sample) or ``[L, B, N]``
        (a whole batch).  The DP state ``level_log_nu`` / ``ptw_log_w`` is
        per (level, node) and broadcasts cleanly over the batch axis, so the
        batched form is a pure broadcast --- no Python loop over samples.
        Returns the matching ``[N]`` / ``[B, N]`` shape.
        """
        assert self.level_log_nu is not None and self.ptw_log_w is not None
        L = self.active_level_count
        # log pi_j computed in float64 for numerical safety.
        log_pi = (
            math.log(0.5) + self.level_log_nu - self.ptw_log_w
        ).clamp(max=0.0)  # pi_j <= 1 by construction; defend against fp drift.
        pi = log_pi.exp().clamp(0.0, 1.0).to(q1_per_level.dtype)
        if q1_per_level.dim() == 3:
            # [L, B, N]: broadcast pi[L, N] -> pi[L, 1, N]
            pi_b = pi.unsqueeze(1)
            out = q1_per_level[L - 1]
            for j in range(L - 2, -1, -1):
                out = pi_b[j] * q1_per_level[j] + (1.0 - pi_b[j]) * out
            return out
        # [L, N]
        out = q1_per_level[L - 1]
        for j in range(L - 2, -1, -1):
            out = pi[j] * q1_per_level[j] + (1.0 - pi[j]) * out
        return out

    @torch.no_grad()
    def _ptw_dp_update_state(self, log_q_obs_per_level: torch.Tensor) -> None:
        """Update level_log_nu and ptw_log_w with ``log q_j(x_obs|x_<t)`` for
        one sample.  Implements Algorithm 1 lines 10--13 in log space.

        ``log_q_obs_per_level`` has shape ``[L, N]`` (one sample's per-level
        per-node observed-symbol log-prob).
        """
        assert self.level_log_nu is not None and self.ptw_log_w is not None
        assert self.ptw_log_b is not None
        L = self.active_level_count
        log_q = log_q_obs_per_level.to(self.level_log_nu.dtype)
        self.level_log_nu.add_(log_q)
        # Line 10: w_d <- nu_{r_d}(x_{r_d:t})
        self.ptw_log_w[L - 1].copy_(self.level_log_nu[L - 1])
        # Lines 11-13: bottom-up logaddexp.
        log_half = math.log(0.5)
        for j in range(L - 2, -1, -1):
            a = log_half + self.level_log_nu[j]
            b = log_half + self.ptw_log_w[j + 1] + self.ptw_log_b[j]
            torch.logaddexp(a, b, out=self.ptw_log_w[j])

    @torch.no_grad()
    @torch.no_grad()
    def _apply_ptw_dp_close_inline(self, close_levels: "list[int]") -> None:
        """Inline version of ``_apply_ptw_dp_close`` for use inside the DP
        recurrence.  Same line-5/line-8 logic, but does not gate on
        ``self.per_level_active`` (the caller already does, because this
        helper only runs in the DP path).
        """
        assert self.level_log_nu is not None
        assert self.ptw_log_w is not None
        assert self.ptw_log_b is not None
        if not close_levels:
            return
        sorted_levels = sorted({self._level_index(j) for j in close_levels})
        i_plus_one = sorted_levels[0]
        if i_plus_one > 0:
            i = i_plus_one - 1
            self.ptw_log_b[i].copy_(self.ptw_log_w[i_plus_one])
        idx = torch.tensor(sorted_levels, dtype=torch.long, device=self.device)
        self.level_log_nu.index_fill_(0, idx, 0.0)
        self.ptw_log_w.index_fill_(0, idx, 0.0)
        self.ptw_log_b.index_fill_(0, idx, 0.0)

    @torch.no_grad()
    @_profiled("layer.ptw_dp_apply_chunk_segmented")
    def _ptw_dp_apply_chunk_segmented(
        self,
        log_q_obs_chunk: torch.Tensor,
        q1_chunk: torch.Tensor,
        close_events: "list[tuple[int, list[int]]]",
    ) -> torch.Tensor:
        """Segmented variant of ``_ptw_dp_apply_chunk``.

        Splits the chunk into segments at each (offset, close_levels) event,
        applies line-5/line-8 to the DP state before each segment begins,
        and runs the chunk-wide DP combine in a SINGLE vectorised pass
        regardless of the number of close events.  All output samples come
        back in their original per-chunk order in an output tensor of
        shape [B, N].

        Phase 5G Commit C vectorisation.  The previous implementation
        dispatched one ``_ptw_dp_apply_chunk`` slice per segment plus one
        ``_apply_ptw_dp_close_inline`` per event, which at depth=15,
        chunk=1024 with ~340 events/sub-chunk drove ~180k tiny CUDA
        dispatches per seed (the dominant cost behind the Phase 5F D-
        step wedged sweep).  The vectorised body now:

        1. Pays ONE chunk-wide ``cumsum`` over ``log_q_obs_chunk``.
        2. Walks deduplicated event offsets Python-side ONCE, advancing
           per-level [L, N] scalar state (``nu_running``, ``w_running``,
           ``b_running``) and recording per-segment ``(nu0, w0, b)``
           triples.  Each iteration is O(L) small-tensor ops -- no
           per-event slice into ``log_q_obs`` and no nested
           ``_ptw_dp_apply_chunk`` call.
        3. Builds per-position [L, B, N] tensors of ``(nu0, w0, b)``
           via segment-range scatters, then runs the bottom-up DP combine
           ONCE across the whole [B, N] output via ``torch.logaddexp``
           on full-chunk tensors, with ``torch.where`` overrides at
           segment-start positions to honour the per-segment stored
           ``w0`` semantics (matches ``_ptw_dp_apply_chunk``'s sample-0
           override).
        4. Commits the post-chunk ``nu_running`` / ``w_running`` /
           ``b_running`` to ``self.level_log_nu`` / ``self.ptw_log_w`` /
           ``self.ptw_log_b``.

        The ``b`` cross-event dependency (``b_i \u2190 w_{i+1}`` at line 5
        reads the pre-close ``w`` mutated by both the previous segment's
        bottom-up combine and the previous event's close) is preserved
        sequentially in the Python event loop; only the [B, N]-shape
        compute is vectorised away from the loop.  See COOKBOOK Phase 5G
        Commit C for the deferred fully-vectorised-``b`` follow-up.

        Bit-identical to the legacy per-segment implementation on the four
        ``test_ptw_dp_apply_chunk_segmented_*`` math tests; the new
        ``test_ptw_dp_apply_chunk_segmented_perf_floor_*`` test pins the
        "zero inner ``_ptw_dp_apply_chunk`` calls" property.
        """
        assert self.level_log_nu is not None
        assert self.ptw_log_w is not None
        assert self.ptw_log_b is not None
        L = self.active_level_count
        B = int(log_q_obs_chunk.size(1))
        N = int(log_q_obs_chunk.size(-1))
        if B == 0:
            return torch.empty_like(q1_chunk[0])
        device = self.level_log_nu.device
        dtype = self.level_log_nu.dtype
        half = math.log(0.5)

        # Deduplicate events: at one offset, merge any duplicate close_levels.
        events_by_offset: dict[int, list[int]] = {}
        for offset, levels in close_events:
            if not 0 <= offset <= B:
                continue
            cur = events_by_offset.setdefault(offset, [])
            for j in levels:
                if j not in cur:
                    cur.append(j)
        sorted_offsets = sorted(events_by_offset.keys())

        # No events -> defer to the simple single-segment path.  This is
        # also the only ``self._ptw_dp_apply_chunk`` call in this method,
        # so the perf-floor test ("zero inner calls when events exist")
        # holds for every non-trivial close_events list.
        if not sorted_offsets:
            return self._ptw_dp_apply_chunk(log_q_obs_chunk, q1_chunk)

        # Chunk-wide cumsum, paid ONCE per chunk.  Padded so
        # ``cum_padded[:, 0]`` == 0 and ``cum_padded[:, k]`` is the
        # cumulative log_q sum *strictly before* position k.
        log_q = log_q_obs_chunk.to(dtype)  # [L, B, N]
        cum_inclusive = log_q.cumsum(dim=1)  # [L, B, N]
        zero_col = torch.zeros(L, 1, N, dtype=dtype, device=device)
        cum_padded = torch.cat([zero_col, cum_inclusive], dim=1)  # [L, B+1, N]

        # Phase 5H-2 fast path: when the multilevel .so exposes
        # ``segmented_dp_state_advance`` AND state lives on CUDA AND
        # ``L <= ML_SEG_MAX_L=32``, dispatch the per-node serial event
        # loop into a single CUDA launch.  The kernel mutates
        # ``self.level_log_nu`` / ``self.ptw_log_w`` / ``self.ptw_log_b``
        # in place and returns ``[S, L, N]`` seed tensors that the
        # existing gather/scatter combine consumes.  When the .so is
        # missing the symbol or tensors are on CPU, ``ml_kernel`` is
        # ``None`` and the CPU Python event loop below runs unchanged.
        ml_kernel = None
        ml = _get_fmn_multilevel_cuda()
        if (
            ml is not None
            and hasattr(ml, "segmented_dp_state_advance")
            and device.type == "cuda"
            and L <= 32
        ):
            ml_kernel = ml

        # Plan ``seg_records`` shape (deduplicate events, identify seg
        # starts/ends).  Used by BOTH the CUDA fast path and the CPU
        # Python loop.
        seg_starts: list[int] = []
        seg_ends: list[int] = []
        event_seg_idx: list[int] = []  # one per dedup'd event; -1 if
                                       # event sits at the current seg
                                       # start (no zero-length segment
                                       # recorded for it).
        cur_start = 0
        for off in sorted_offsets:
            if off > cur_start:
                event_seg_idx.append(len(seg_starts))
                seg_starts.append(cur_start)
                seg_ends.append(off)
                cur_start = off
            else:
                event_seg_idx.append(-1)
        trailing_seg_idx = -1
        if cur_start < B:
            trailing_seg_idx = len(seg_starts)
            seg_starts.append(cur_start)
            seg_ends.append(B)
        S = len(seg_starts)

        if ml_kernel is not None and S > 0:
            # Build per-event close mask once on CPU then move to device.
            with prof("layer.ptw_dp_seg_cuda_pack"):
                E = len(sorted_offsets)
                event_close_mask_cpu = torch.zeros(E, L, dtype=torch.bool)
                for e_i, off in enumerate(sorted_offsets):
                    for j in events_by_offset[off]:
                        if 0 <= j < L:
                            event_close_mask_cpu[e_i, j] = True
                event_offsets_dev = torch.tensor(
                    sorted_offsets, dtype=torch.int32, device=device
                )
                event_seg_idx_dev = torch.tensor(
                    event_seg_idx, dtype=torch.int32, device=device
                )
                event_close_mask_dev = event_close_mask_cpu.to(device, non_blocking=True).contiguous()
                cum_padded_c = cum_padded.contiguous()
            with prof("layer.ptw_dp_seg_cuda_kernel"):
                seg_nu0_all, seg_w0_all, seg_b0_all = ml_kernel.segmented_dp_state_advance(
                    cum_padded_c,
                    event_offsets_dev,
                    event_close_mask_dev,
                    event_seg_idx_dev,
                    int(trailing_seg_idx),
                    self.level_log_nu,
                    self.ptw_log_w,
                    self.ptw_log_b,
                    int(S),
                )
            # ``seg_*_all`` is [S, L, N]; permute to [L, S, N] for gather.
            seg_nu0_lsn = seg_nu0_all.permute(1, 0, 2).contiguous()
            seg_w0_lsn = seg_w0_all.permute(1, 0, 2).contiguous()
            seg_b0_lsn = seg_b0_all.permute(1, 0, 2).contiguous()
            # Build seg_idx_per_pos: [B] int64 mapping each position to
            # its segment record.  Same scatter the Python loop did.
            seg_idx_per_pos = torch.empty(B, dtype=torch.long, device=device)
            seg_start_per_pos = torch.empty(B, dtype=torch.long, device=device)
            is_seg_start = torch.zeros(B, dtype=torch.bool, device=device)
            for si, (s, e) in enumerate(zip(seg_starts, seg_ends)):
                seg_idx_per_pos[s:e] = si
                seg_start_per_pos[s:e] = s
                is_seg_start[s] = True
            # Gather per-position seeds from [L, S, N] -> [L, B, N].
            gather_idx = seg_idx_per_pos.view(1, B, 1).expand(L, B, N)
            nu0_per_pos = seg_nu0_lsn.gather(dim=1, index=gather_idx)
            w0_per_pos = seg_w0_lsn.gather(dim=1, index=gather_idx)
            b_per_pos = seg_b0_lsn.gather(dim=1, index=gather_idx)
            # Drop straight into the existing combine block by recording
            # the "we used the CUDA fast path" flag.
            _seg_fast_done = True
        else:
            _seg_fast_done = False

        if not _seg_fast_done:
            # Python event loop: advances scalar-shape per-level state and
            # records per-segment ``(nu0, w0, b)`` triples.  ``seg_records``
            # lists tuples ``(seg_start, seg_end, nu0, w0, b)`` covering
            # [0, B) exhaustively (an event at offset 0 shifts the first
            # segment's start state but does not create a zero-length
            # segment; an event at offset B applies at chunk end but does
            # not introduce a trailing segment).
            nu_running = self.level_log_nu.clone()  # [L, N]
            w_running = self.ptw_log_w.clone()  # [L, N]
            b_running = self.ptw_log_b.clone()  # [L, N]

            def _bottom_up_w(nu_now: torch.Tensor, b_now: torch.Tensor) -> torch.Tensor:
                """Recompute ``ptw_log_w`` bottom-up from ``nu`` and constant ``b``.

                Used after every segment advance so the next event's line-5 read
                (``b[i] <- w[i+1]``) sees a consistent ``w_running`` that
                reflects both the segment's accumulated ``nu`` and the
                segment-constant ``b``.
                """
                w_out = torch.empty_like(w_running)
                w_out[L - 1] = nu_now[L - 1]
                for j in range(L - 2, -1, -1):
                    w_out[j] = torch.logaddexp(
                        half + nu_now[j],
                        half + w_out[j + 1] + b_now[j],
                    )
                return w_out

            seg_records: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            seg_start = 0
            for off in sorted_offsets:
                if off > seg_start:
                    # Record the current state as this segment's (nu0, w0, b).
                    seg_records.append((
                        seg_start, off,
                        nu_running.clone(), w_running.clone(), b_running.clone(),
                    ))
                    # Advance running state through this segment:
                    # ``nu += sum_over_segment``; ``w`` becomes the bottom-up
                    # combine of the new ``nu`` with the segment's constant
                    # ``b``.  Both are needed for the NEXT event's line-5
                    # read (``b[i] <- w[i+1]`` pre-close).
                    seg_cum = cum_padded[:, off, :] - cum_padded[:, seg_start, :]
                    nu_running = nu_running + seg_cum
                    w_running = _bottom_up_w(nu_running, b_running)
                    seg_start = off
                # Apply close-event mutations: line 5 (b[i] <- w[i+1]) then
                # line 8 (zero nu / w / b at every closed level).
                sorted_levels = sorted({int(j) for j in events_by_offset[off]})
                i_plus_one = sorted_levels[0]
                if i_plus_one > 0:
                    i = i_plus_one - 1
                    b_running[i] = w_running[i_plus_one].clone()
                for j in sorted_levels:
                    nu_running[j] = 0.0
                    w_running[j] = 0.0
                    b_running[j] = 0.0

            # Trailing segment [seg_start, B), if any (no event at offset B
            # OR an event at offset B but seg_start < B because the previous
            # event opened a non-empty segment).
            if seg_start < B:
                seg_records.append((
                    seg_start, B,
                    nu_running.clone(), w_running.clone(), b_running.clone(),
                ))
                seg_cum = cum_padded[:, B, :] - cum_padded[:, seg_start, :]
                nu_running = nu_running + seg_cum
                w_running = _bottom_up_w(nu_running, b_running)

            if not seg_records:
                # Only reachable if every event sits at offset==B (so the
                # whole chunk is "before the first segment"), which is
                # degenerate for this API; commit any tail state and return.
                self.level_log_nu.copy_(nu_running)
                self.ptw_log_w.copy_(w_running)
                self.ptw_log_b.copy_(b_running)
                return torch.empty_like(q1_chunk[0])

        if not _seg_fast_done:
            # Build per-position [L, B, N] tensors from per-segment scalar
            # state via segment-range scatters.
            nu0_per_pos = torch.empty(L, B, N, dtype=dtype, device=device)
            w0_per_pos = torch.empty(L, B, N, dtype=dtype, device=device)
            b_per_pos = torch.empty(L, B, N, dtype=dtype, device=device)
            seg_start_per_pos = torch.empty(B, dtype=torch.long, device=device)
            is_seg_start = torch.zeros(B, dtype=torch.bool, device=device)
            for (s, e, nu0, w0, b0) in seg_records:
                nu0_per_pos[:, s:e, :] = nu0.unsqueeze(1)
                w0_per_pos[:, s:e, :] = w0.unsqueeze(1)
                b_per_pos[:, s:e, :] = b0.unsqueeze(1)
                seg_start_per_pos[s:e] = s
                is_seg_start[s] = True

        # ``nu_state[j, b, n]`` = ``nu0_per_pos[j, b, n] +
        # sum_{i=seg_start..b-1} log_q[j, i, n]``
        # = ``nu0_per_pos + cum_padded[:, b, :] - cum_padded[:, seg_start_per_pos[b], :]``
        seg_start_idx_l = seg_start_per_pos.view(1, B, 1).expand(L, B, N)
        cum_baseline = cum_padded.gather(dim=1, index=seg_start_idx_l)  # [L, B, N]
        cum_excl_seg = cum_padded[:, :B, :] - cum_baseline  # [L, B, N]
        nu_state = nu0_per_pos + cum_excl_seg  # [L, B, N]

        # Bottom-up combine for ``w_state`` over the whole chunk.  At
        # each segment-start position, override the bottom-up value with
        # the per-segment stored ``w0`` to match
        # ``_ptw_dp_apply_chunk``'s sample-0 stored-w override.
        w_levels: list[torch.Tensor] = [None] * L  # type: ignore[list-item]
        w_levels[L - 1] = nu_state[L - 1]
        for j in range(L - 2, -1, -1):
            a = half + nu_state[j]
            c = half + w_levels[j + 1] + b_per_pos[j]
            w_levels[j] = torch.logaddexp(a, c)
        for j in range(L):
            w_levels[j] = torch.where(
                is_seg_start.view(B, 1), w0_per_pos[j], w_levels[j]
            )

        log_pi_per_level = [
            (half + nu_state[j] - w_levels[j]).clamp(max=0.0) for j in range(L)
        ]
        pi_per_level = [
            lp.exp().clamp(0.0, 1.0).to(q1_chunk.dtype) for lp in log_pi_per_level
        ]
        out = q1_chunk[L - 1]
        for j in range(L - 2, -1, -1):
            out = pi_per_level[j] * q1_chunk[j] + (1.0 - pi_per_level[j]) * out

        # Commit post-chunk state.  CUDA fast path mutated the
        # per-level state in place inside the kernel; the Python event
        # loop accumulated it into nu_running/w_running/b_running.
        if not _seg_fast_done:
            self.level_log_nu.copy_(nu_running)
            self.ptw_log_w.copy_(w_running)
            self.ptw_log_b.copy_(b_running)
        return out

    @_profiled("layer.ptw_dp_apply_chunk")
    def _ptw_dp_apply_chunk(
        self,
        log_q_obs_chunk: torch.Tensor,
        q1_chunk: torch.Tensor,
        close_events: "list[tuple[int, list[int]]] | None" = None,
    ) -> torch.Tensor:
        """Vectorised per-chunk DP combine + state advance (Phase 5D).

        Replaces the per-sample ``_ptw_dp_combine`` / ``_ptw_dp_update_state``
        pair when the whole chunk's ``log q_j(x_obs|x_<t at b)`` and
        ``q_j(x_t=1|x_<t at b)`` are already available as ``[L, B, N]``
        tensors.

        The state BEFORE observing sample ``b`` is
        ``nu_b[j, n] = level_log_nu[j, n] + sum_{i=0..b-1} log_q_obs[j, i, n]``
        which is a single cumulative sum minus the inclusive entry at ``b``.
        ``w_b`` is then a non-recurrent bottom-up combine of ``nu_b``
        against the (chunk-constant) ``ptw_log_b``, so every level's update
        is one ``[B, N]`` op.  Net cost: O(L) PyTorch ops per chunk instead
        of O(B * L), which is the O(B * L) source of the Phase 2 perf
        regression on long chunks.

        Mutates ``self.level_log_nu`` and ``self.ptw_log_w`` to the
        post-chunk state.  Returns ``[B, N]`` of P_0(x_t=1|x_<t at b).

        ``close_events`` is an optional list of ``(local_offset, close_levels)``
        pairs.  When non-empty, the chunk is processed as a sequence of
        segments separated by those events; line-5/line-8 of Algorithm 1
        fire before the segment containing each event's offset begins.
        When None or empty, behaviour is identical to the legacy Phase 5D
        path -- the chunk is one segment.
        """
        if close_events:
            return self._ptw_dp_apply_chunk_segmented(
                log_q_obs_chunk, q1_chunk, close_events
            )
        assert self.level_log_nu is not None and self.ptw_log_w is not None
        assert self.ptw_log_b is not None
        L = self.active_level_count
        B = log_q_obs_chunk.size(1)
        if B == 0:
            return torch.empty_like(q1_chunk[0])
        nu0 = self.level_log_nu  # [L, N], float64 (state at sample-0 entry)
        w0 = self.ptw_log_w  # [L, N], float64 (stored consistent w at chunk entry)
        log_q = log_q_obs_chunk.to(nu0.dtype)  # [L, B, N]
        # cum_inclusive[j, b, n] = sum_{i=0..b} log_q[j, i, n]
        cum_inclusive = log_q.cumsum(dim=1)
        # state BEFORE sample b: nu0[j, n] + cum at b-1 = nu0 + cum_inclusive - log_q
        nu_state = nu0.unsqueeze(1) + cum_inclusive - log_q  # [L, B, N]
        half = math.log(0.5)
        # Bottom-up w_state from nu_state, with ptw_log_b held chunk-constant.
        # We materialise into a list to avoid out-of-place index_put on a
        # whole [L, B, N] tensor every iteration.
        # Crucial subtlety: immediately after `_apply_ptw_dp_close`, the
        # stored ptw_log_w[j] for non-closing levels remains the pre-close
        # value, which is NOT necessarily bottom-up(nu, ptw_log_b) -- the
        # close zeros nu[j]/b[j] for closing j without recomputing w for
        # the levels above.  The per-sample reference uses the stored w
        # at sample 0 and recomputes it via `_ptw_dp_update_state` from
        # sample 1 onwards.  Match exactly: use stored w0 at b=0, derived
        # w from nu_state at b>=1.
        w_levels: list[torch.Tensor] = [None] * L  # type: ignore[list-item]
        w_levels[L - 1] = nu_state[L - 1]
        for j in range(L - 2, -1, -1):
            a = half + nu_state[j]
            c = half + w_levels[j + 1] + self.ptw_log_b[j].unsqueeze(0)
            w_levels[j] = torch.logaddexp(a, c)
        # Override sample 0 with the chunk-entry stored w (paper-faithful).
        for j in range(L):
            w_levels[j] = w_levels[j].clone()
            w_levels[j][0] = w0[j]
        # pi[j, b, n] in q1's dtype for the combine.
        log_pi_per_level: list[torch.Tensor] = [
            (half + nu_state[j] - w_levels[j]).clamp(max=0.0)
            for j in range(L)
        ]
        pi_per_level = [
            lp.exp().clamp(0.0, 1.0).to(q1_chunk.dtype) for lp in log_pi_per_level
        ]
        # Bottom-up convex combination over levels for the WHOLE batch.
        out = q1_chunk[L - 1]
        for j in range(L - 2, -1, -1):
            out = pi_per_level[j] * q1_chunk[j] + (1.0 - pi_per_level[j]) * out
        # Commit post-chunk DP state to mutable buffers.
        self.level_log_nu.add_(cum_inclusive[:, -1, :])
        self.ptw_log_w[L - 1].copy_(self.level_log_nu[L - 1])
        for j in range(L - 2, -1, -1):
            torch.logaddexp(
                half + self.level_log_nu[j],
                half + self.ptw_log_w[j + 1] + self.ptw_log_b[j],
                out=self.ptw_log_w[j],
            )
        return out

    def _multilevel_state_views(self):
        """Return (mixture_weights_5d, pool_sizes_2d, segment_lp_3d,
        model_lp_3d) suitable for the multilevel kernel.

        Always shares storage with the underlying per-level state tensors,
        so in-place kernel mutations propagate without an explicit copy.
        """
        assert self.per_level_active
        assert self.level_mixture_weights is not None
        assert self.level_pool_sizes is not None
        assert self.level_segment_log_probs is not None
        assert self.level_model_log_probs is not None
        return (
            self.level_mixture_weights,
            self.level_pool_sizes,
            self.level_segment_log_probs,
            self.level_model_log_probs,
        )

    @torch.no_grad()
    @_profiled("layer.forward_only_batch_dp")
    def forward_only_batch_dp(self, z_batch, p_prev_batch):
        """Read-only DP prediction path.  Returns ``[B, N]`` of
        P_0(x_t=1|x_<t) for each sample, computed via Algorithm 1's bottom-up
        recursion over per-level conditional predictions.  Does not mutate any
        per-level active state or DP state.
        """
        if not self.per_level_active:
            raise RuntimeError(
                "forward_only_batch_dp requires per_level_active=True"
            )
        L = self.active_level_count
        ml = _get_fmn_multilevel_cuda()
        if ml is not None:
            # Phase 5E + 5I fast path: one CUDA launch covers every PTW
            # level.  Passes self.posterior_temp so the kernel uses the
            # paper-faithful conditional ratio Eq.5 (posterior_temp=0
            # collapses to the unweighted 1/2 fresh + 1/2 pool-mean
            # legacy mixture; posterior_temp=1.0 is the strict paper).
            mw, ps, slp, _ = self._multilevel_state_views()
            q1 = ml.forward_only(
                z_batch, p_prev_batch, mw, self.hyperplanes, self.hp_bias,
                ps, slp, float(self.posterior_temp),
            )
        else:
            cuda = _get_fmn_cuda()
            # Per-level loop fallback.  Passes self.posterior_temp so
            # the kernel returns the paper-faithful posterior-weighted
            # ν_j when the .so does not expose the multilevel kernel.
            per_level = []
            for level in range(L):
                weights, pool_sizes, seg_lp, _ = self._active_tensors(level)
                pred = cuda.forward_only(
                    z_batch,
                    p_prev_batch,
                    weights,
                    self.hyperplanes,
                    self.hp_bias,
                    pool_sizes,
                    seg_lp,
                    float(self.posterior_temp),
                )
                per_level.append(pred)
            q1 = torch.stack(per_level, dim=0)  # [L, B, N]
        # Read-only path: DP state is unchanged across the batch, so one
        # vectorised combine over [L, B, N] is exact.  The per-sample
        # Python loop here was pure overhead.
        return self._ptw_dp_combine(q1)

    @torch.no_grad()
    @_profiled("layer.forward_update_batch_dp")
    def forward_update_batch_dp(
        self,
        z_batch,
        p_prev_batch,
        symbols_batch,
        close_events: "list[tuple[int, list[int]]] | None" = None,
    ):
        """DP training path.  Returns ``[B, N]`` of P_0(x_obs|x_<t) for each
        sample after applying Algorithm 1's bottom-up recursion AND advancing
        both the per-level CUDA active state (via the kernel's per-sample
        update) and the Python-side DP state (level_log_nu, ptw_log_w).

        ``close_events`` (Phase 5F D-step 4): optional list of
        ``(local_offset, close_levels)`` pairs for pure-close-only events
        that fire BEFORE the listed offset inside this batch.  When
        provided, the multilevel kernel applies per-level FMN active-state
        resets in-kernel via ``forward_update_with_resets`` (if the .so
        exports it) and the DP combine uses the segmented variant of
        ``_ptw_dp_apply_chunk``.  When ``None`` or empty, behaviour is
        identical to the legacy Phase 5E path: one kernel launch, one
        single-segment DP combine.
        """
        if not self.per_level_active:
            raise RuntimeError(
                "forward_update_batch_dp requires per_level_active=True"
            )
        L = self.active_level_count
        B = int(z_batch.size(0))
        symbols_batch = self._coerce_symbols_to_device(symbols_batch, z_batch)
        ml = _get_fmn_multilevel_cuda()
        has_with_resets = ml is not None and hasattr(ml, "forward_update_with_resets")
        if ml is not None and close_events and has_with_resets:
            # Phase 5F fast path: build [L, B] close_mask and dispatch the
            # in-kernel reset variant in one CUDA launch.
            mw, ps, slp, mlp = self._multilevel_state_views()
            with prof("layer.forward_update_batch_dp.build_close_mask"):
                close_mask = build_close_mask(close_events, L, B, device=z_batch.device)
            with prof("layer.forward_update_batch_dp.ml_forward_update_with_resets"):
                q_obs = ml.forward_update_with_resets(
                    z_batch, p_prev_batch, symbols_batch,
                    mw, self.pool_snapshots, self.hyperplanes, self.hp_bias, ps,
                    close_mask, slp, mlp, self.lr,
                    float(self.posterior_temp),
                )
        elif ml is not None:
            # Phase 5E fast path (no close_events or pre-5F .so): one CUDA
            # launch covers all L levels' forward+update.  In-place
            # mutations to mixture_weights, segment_log_probs,
            # model_log_probs propagate through the 5D/3D views since those
            # alias the per-level state tensors.
            mw, ps, slp, mlp = self._multilevel_state_views()
            with prof("layer.forward_update_batch_dp.ml_forward_update"):
                q_obs = ml.forward_update(
                    z_batch, p_prev_batch, symbols_batch,
                    mw, self.hyperplanes, self.hp_bias, ps, slp, mlp, self.lr,
                    float(self.posterior_temp),
                )
            # If we were asked to honour close_events but the .so predates
            # 5F, we still need to apply per-level active-state resets in
            # Python at the right offsets.  This is a correctness-only
            # fallback; it does NOT recover the perf win, but it keeps the
            # numerics paper-faithful so downstream layers see the right
            # state.  The per-event reset for active state lives on
            # ``segment_close``; the caller (train_chunk) is responsible
            # for invoking it.  We log nothing here -- the path is rare.
        else:
            cuda = _get_fmn_cuda()
            # Per-level loop fallback (legacy path).  posterior_temp=0 so the
            # kernel returns paper-uniform ν_j.  Per-slot weight updates
            # inside the kernel are independent of posterior_temp.
            with prof("layer.forward_update_batch_dp.level_loop_fallback"):
                per_level = []
                for level in range(L):
                    weights, pool_sizes, seg_lp, model_lp = self._active_tensors(level)
                    pred = cuda.forward_update(
                        z_batch,
                        p_prev_batch,
                        symbols_batch,
                        weights,
                        self.hyperplanes,
                        self.hp_bias,
                        pool_sizes,
                        seg_lp,
                        model_lp,
                        self.lr,
                        float(self.posterior_temp),
                    )
                    per_level.append(pred)
                q_obs = torch.stack(per_level, dim=0)  # [L, B, N]
        with prof("layer.forward_update_batch_dp.recover_q_log"):
            sym = symbols_batch.to(torch.bool)  # [B] device-resident
            sym_f = sym.to(q_obs.dtype).view(B, 1)  # [B, 1] for branchless select
            # Recover q_j(1|x_<t at b) and log q_obs once for the whole batch.
            q1 = torch.where(sym.view(1, B, 1), q_obs, 1.0 - q_obs)
            log_q_obs_all = torch.log(q_obs.clamp(min=1e-30))  # [L, B, N]
        # Phase 5D: fused batched DP combine + state advance.  When
        # close_events is non-empty, _ptw_dp_apply_chunk dispatches to the
        # segmented variant (D-step 3) that resets the PTW DP state
        # in-line at each event.
        with prof("layer.forward_update_batch_dp.ptw_apply"):
            p1_batch = self._ptw_dp_apply_chunk(
                log_q_obs_all, q1, close_events=close_events,
            )
        # Branchless P_0(x_obs) = sym * p1 + (1 - sym) * (1 - p1).  Single
        # vectorised select replaces the previous per-sample assignment.
        with prof("layer.forward_update_batch_dp.final_select"):
            return sym_f * p1_batch + (1.0 - sym_f) * (1.0 - p1_batch)

    @staticmethod
    def _coerce_symbols_to_device(symbols: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Place ``symbols`` on ``reference.device`` and cast to int32 for the
        CUDA kernel binding.  Centralised here so the DP path matches the
        invariant ``NctlNetwork._coerce_symbols`` upholds for the legacy path.
        """
        if symbols.dtype != torch.int32 or symbols.device != reference.device:
            symbols = symbols.to(device=reference.device, dtype=torch.int32)
        return symbols.contiguous()

    @torch.no_grad()
    @_profiled("layer.observe_segment_batch")
    def observe_segment_batch(self, z_batch, p_prev_batch, symbols_batch) -> None:
        """Reservoir-sample layer inputs for the currently open segment."""
        if self.pool_reservoir_size <= 0:
            return
        z_cpu = z_batch.detach().to("cpu")
        p_cpu = p_prev_batch.detach().to("cpu")
        s_cpu = symbols_batch.detach().to("cpu", dtype=torch.int32)
        if self.per_level_active:
            for level in range(self.active_level_count):
                self._observe_segment_batch_for_level(level, z_cpu, p_cpu, s_cpu)
            return
        self._observe_segment_batch_flat(z_cpu, p_cpu, s_cpu)

    def _observe_into_reservoir(
        self,
        z_store: torch.Tensor,
        p_store: torch.Tensor,
        symbol_store: torch.Tensor,
        seen: int,
        size: int,
        z_cpu: torch.Tensor,
        p_cpu: torch.Tensor,
        s_cpu: torch.Tensor,
    ) -> tuple[int, int]:
        for i in range(int(z_cpu.size(0))):
            if size < self.pool_reservoir_size:
                slot = size
                size += 1
            else:
                slot = int(torch.randint(0, seen + 1, (1,), generator=self._reservoir_rng).item())
                if slot >= self.pool_reservoir_size:
                    seen += 1
                    continue
            z_store[slot].copy_(z_cpu[i])
            p_store[slot].copy_(p_cpu[i])
            symbol_store[slot] = s_cpu[i]
            seen += 1
        return seen, size

    def _observe_segment_batch_flat(self, z_cpu, p_cpu, s_cpu) -> None:
        self.segment_res_seen, self.segment_res_size = self._observe_into_reservoir(
            self.segment_res_z,
            self.segment_res_p_prev,
            self.segment_res_symbols,
            int(self.segment_res_seen),
            int(self.segment_res_size),
            z_cpu,
            p_cpu,
            s_cpu,
        )

    def _observe_segment_batch_for_level(self, level: int, z_cpu, p_cpu, s_cpu) -> None:
        assert self.level_res_seen is not None
        assert self.level_res_size is not None
        assert self.level_res_z is not None
        assert self.level_res_p_prev is not None
        assert self.level_res_symbols is not None
        seen, size = self._observe_into_reservoir(
            self.level_res_z[level],
            self.level_res_p_prev[level],
            self.level_res_symbols[level],
            int(self.level_res_seen[level].item()),
            int(self.level_res_size[level].item()),
            z_cpu,
            p_cpu,
            s_cpu,
        )
        self.level_res_seen[level] = seen
        self.level_res_size[level] = size

    def _reset_segment_reservoir(self, level: int | None = None) -> None:
        if self.per_level_active and level is not None:
            assert self.level_res_seen is not None
            assert self.level_res_size is not None
            idx = self._level_index(level)
            self.level_res_seen[idx] = 0
            self.level_res_size[idx] = 0
            return
        self.segment_res_seen = 0
        self.segment_res_size = 0

    def _segment_reservoir(self, level: int | None = None):
        if self.per_level_active and level is not None:
            idx = self._level_index(level)
            assert self.level_res_size is not None
            assert self.level_res_z is not None
            assert self.level_res_p_prev is not None
            assert self.level_res_symbols is not None
            size = int(self.level_res_size[idx].item())
            if size <= 0:
                return None
            return (
                self.level_res_z[idx, :size],
                self.level_res_p_prev[idx, :size],
                self.level_res_symbols[idx, :size],
            )
        size = self.segment_res_size
        if size <= 0:
            return None
        return (
            self.segment_res_z[:size],
            self.segment_res_p_prev[:size],
            self.segment_res_symbols[:size],
        )

    def _pool_reservoir(self, n: int, slot: int):
        size = int(self.pool_res_sizes[n, slot].item())
        if size <= 0:
            return None
        return (
            self.pool_res_z[n, slot, :size],
            self.pool_res_p_prev[n, slot, :size],
            self.pool_res_symbols[n, slot, :size],
        )

    def _store_segment_reservoir(self, n: int, slot: int, level: int | None = None) -> None:
        res = self._segment_reservoir(level)
        size = 0 if res is None else int(res[0].size(0))
        self.pool_res_sizes[n, slot] = size
        if res is not None and size > 0:
            z, p_prev, symbols = res
            self.pool_res_z[n, slot, :size].copy_(z[:size])
            self.pool_res_p_prev[n, slot, :size].copy_(p_prev[:size])
            self.pool_res_symbols[n, slot, :size].copy_(symbols[:size])

    @_profiled("layer.merge_pool_and_segment_reservoirs")
    def _merge_pool_and_segment_reservoirs(self, n: int, slot: int, level: int | None = None) -> None:
        if self.pool_reservoir_size <= 0:
            return
        old_size = int(self.pool_res_sizes[n, slot].item())
        seg_res = self._segment_reservoir(level)
        new_size = 0 if seg_res is None else int(seg_res[0].size(0))
        if old_size <= 0:
            self._store_segment_reservoir(n, slot, level)
            return
        if new_size <= 0 or seg_res is None:
            return
        seg_z, seg_p_prev, seg_symbols = seg_res
        z = torch.cat([self.pool_res_z[n, slot, :old_size], seg_z[:new_size]], dim=0)
        p_prev = torch.cat([self.pool_res_p_prev[n, slot, :old_size], seg_p_prev[:new_size]], dim=0)
        symbols = torch.cat([self.pool_res_symbols[n, slot, :old_size], seg_symbols[:new_size]], dim=0)
        total = int(z.size(0))
        keep = min(self.pool_reservoir_size, total)
        if keep < total:
            perm = torch.randperm(total, generator=self._reservoir_rng)[:keep]
            z = z[perm]
            p_prev = p_prev[perm]
            symbols = symbols[perm]
        self.pool_res_sizes[n, slot] = keep
        self.pool_res_z[n, slot, :keep].copy_(z[:keep])
        self.pool_res_p_prev[n, slot, :keep].copy_(p_prev[:keep])
        self.pool_res_symbols[n, slot, :keep].copy_(symbols[:keep])

    def _predict_node_with_weights(
        self,
        node: int,
        weights: torch.Tensor,
        z_cpu: torch.Tensor,
        p_prev_cpu: torch.Tensor,
    ) -> torch.Tensor:
        device = weights.device
        z = z_cpu.to(device=device, dtype=torch.float32)
        p_prev = p_prev_cpu.to(device=device, dtype=torch.float32)
        hp = self.hyperplanes[node]
        bias = self.hp_bias[node]
        gates = ((z @ hp.T) + bias) >= 0
        powers = (2 ** torch.arange(self.H, device=device, dtype=torch.long)).view(1, -1)
        contexts = (gates.to(torch.long) * powers).sum(dim=1)
        logits = torch.logit(p_prev.clamp(1e-7, 1.0 - 1e-7)).clamp(-15.0, 15.0)
        selected = weights[contexts]
        return torch.sigmoid((selected * logits).sum(dim=1))

    @_profiled("layer.log_prob_for_weights")
    def _log_prob_for_weights(
        self,
        node: int,
        weights: torch.Tensor,
        reservoir,
    ) -> float:
        if reservoir is None:
            return float("-inf")
        z_cpu, p_prev_cpu, symbols_cpu = reservoir
        if int(z_cpu.size(0)) <= 0:
            return float("-inf")
        pred = self._predict_node_with_weights(node, weights, z_cpu, p_prev_cpu)
        symbols = symbols_cpu.to(device=pred.device, dtype=torch.bool)
        psym = torch.where(symbols, pred, 1.0 - pred).clamp(min=1e-30)
        return float(torch.log(psym).sum().item())

    @_profiled("layer.log_prob_uniform_pool_mixture")
    def _log_prob_uniform_pool_mixture(self, node: int, k: int, reservoir) -> float:
        """Paper-faithful joint log ξ(s) of the uniform Bayesian mixture
        over the current pool, evaluated on a reservoir of segment data.

        With a uniform prior 1/|M| over the |M|=k pool slots,

            log ξ(s) = log[(1/k) Σ_m ρ_m(s)]
                     = -log(k) + logsumexp_m( log ρ_m(s) )

        where log ρ_m(s) is the JOINT log-likelihood under slot m, i.e. the
        sum of per-symbol log probabilities at slot m's frozen weights.
        This is the exact value the paper's β skip threshold compares
        against the best newly-trained model's log probability on the
        same reservoir; per-step averaging would understate it by
        Jensen's inequality and effectively disable β.
        """
        if k <= 0 or reservoir is None:
            return float("-inf")
        z_cpu, p_prev_cpu, symbols_cpu = reservoir
        per_slot_joint_logp: list[float] = []
        for slot in range(k):
            pred = self._predict_node_with_weights(
                node, self.pool_snapshots[node, slot], z_cpu, p_prev_cpu,
            )
            symbols = symbols_cpu.to(device=pred.device, dtype=torch.bool)
            psym = torch.where(symbols, pred, 1.0 - pred).clamp(min=1e-30)
            per_slot_joint_logp.append(float(torch.log(psym).sum().item()))
        if not per_slot_joint_logp:
            return float("-inf")
        max_lp = max(per_slot_joint_logp)
        if max_lp == float("-inf"):
            return float("-inf")
        # logsumexp_m(x_m) = max + log Σ exp(x_m - max)
        log_sum_exp = max_lp + math.log(
            sum(math.exp(lp - max_lp) for lp in per_slot_joint_logp)
        )
        return float(log_sum_exp - math.log(float(k)))

    @_profiled("layer.choose_paper_pool_action")
    def _choose_paper_pool_action(
        self,
        node: int,
        k: int,
        best_slot: int,
        best_weights: torch.Tensor,
        level: int | None = None,
    ) -> tuple[str, int | None]:
        seg_res = self._segment_reservoir(level)
        if self.pool_update_policy != "paper" or seg_res is None:
            return "add", None

        if best_slot < k:
            old_res = self._pool_reservoir(node, best_slot)
            if old_res is not None:
                old_lp = self._log_prob_for_weights(
                    node, self.pool_snapshots[node, best_slot], old_res
                )
                new_lp = self._log_prob_for_weights(node, best_weights, old_res)
                if new_lp - old_lp > self.pool_alpha:
                    return "refine", best_slot

        mix_lp = self._log_prob_uniform_pool_mixture(node, k, seg_res)
        best_lp = self._log_prob_for_weights(node, best_weights, seg_res)
        if mix_lp - best_lp > self.pool_beta:
            return "skip", None
        return "add", None

    def _record_pool_slot(
        self,
        node: int,
        slot: int,
        weights: torch.Tensor,
        score: torch.Tensor,
        deepest_level: int,
        task_id: int,
        global_index: int,
    ) -> None:
        self.pool_snapshots[node, slot].copy_(weights)
        self.pool_levels[node, slot] = int(deepest_level)
        self.pool_task_ids[node, slot] = task_id
        self.pool_insert_indices[node, slot] = global_index
        self.model_log_probs[node, slot] = score

    def _choose_age_diversity_evict_slot(self, node: int) -> int:
        """Evict the temporally most redundant slot without using task ids.

        ``pool_insert_indices`` are benchmark-agnostic segment close positions.
        For each slot, find the distance to its nearest temporal neighbour and
        evict the slot with the smallest such distance.  Ties resolve to the
        oldest slot, preserving deterministic FIFO-like behavior when temporal
        density is equal.
        """
        insert_indices = [int(v) for v in self.pool_insert_indices[node].tolist()]
        active = [(slot, idx) for slot, idx in enumerate(insert_indices) if idx >= 0]
        if len(active) < 2:
            # Full pools normally have valid metadata for every slot; if a
            # legacy or hand-built state does not, make the fallback exactly
            # FIFO instead of inventing an age-diversity decision.
            return 0

        best_slot = 0
        best_key: tuple[int, int] | None = None
        for slot, idx in active:
            nearest = min(
                abs(idx - other_idx)
                for other_slot, other_idx in active
                if other_slot != slot
            )
            key = (nearest, slot)
            if best_key is None or key < best_key:
                best_key = key
                best_slot = slot
        return best_slot

    def _choose_age_diversity_oldest_floor_evict_slot(self, node: int) -> int:
        """age-diversity, but protect the ``pool_oldest_floor`` oldest slots.

        Slots are FIFO-ordered (slot 0 = oldest), so the oldest floor is the
        leading slots ``[0:pool_oldest_floor]``.  Among the remaining slots we
        evict the temporally most redundant one (smallest nearest-neighbour gap
        measured against every surviving snapshot, including protected ones).
        Ties resolve to the oldest eligible slot.  This keeps the earliest
        task's snapshots from being flushed -- the residual weakness of plain
        age-diversity, where the oldest task starves to a handful of slots --
        without consulting task ids.  Falls back to FIFO when the floor leaves
        nothing eligible to evict.
        """
        insert_indices = [int(v) for v in self.pool_insert_indices[node].tolist()]
        active = [(slot, idx) for slot, idx in enumerate(insert_indices) if idx >= 0]
        if len(active) < 2:
            return 0
        eligible = [
            (slot, idx) for slot, idx in active if slot >= self.pool_oldest_floor
        ]
        if not eligible:
            # Floor covers every surviving slot; nothing is safe to drop under
            # the floor invariant, so fall back to FIFO oldest eviction.
            return 0

        best_slot = eligible[0][0]
        best_key: tuple[int, int] | None = None
        for slot, idx in eligible:
            nearest = min(
                abs(idx - other_idx)
                for other_slot, other_idx in active
                if other_slot != slot
            )
            key = (nearest, slot)
            if best_key is None or key < best_key:
                best_key = key
                best_slot = slot
        return best_slot

    def _choose_age_bucket_floor_evict_slot(self, node: int) -> int:
        """Protect log-age buckets, then evict FIFO within overfull buckets.

        The bucket is ``floor(log2(current_index - insert_index + 1))``.  This
        mirrors the FMN/PTW binary temporal hierarchy: old snapshots occupy
        coarse buckets, recent snapshots occupy fine buckets.  We walk slots in
        FIFO order and evict the first slot whose bucket has another survivor.
        If every occupied bucket is a singleton, eviction falls back to FIFO.
        """
        insert_indices = [int(v) for v in self.pool_insert_indices[node].tolist()]
        active = [(slot, idx) for slot, idx in enumerate(insert_indices) if idx >= 0]
        if len(active) < 2:
            return 0

        current_index = max(idx for _, idx in active) + 1

        def bucket_for(idx: int) -> int:
            age = max(1, current_index - idx)
            return int(math.log2(age))

        bucket_counts: dict[int, int] = {}
        for _, idx in active:
            bucket = bucket_for(idx)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        for slot, idx in active:
            if bucket_counts[bucket_for(idx)] >= 2:
                return slot
        return 0

    def _choose_evict_slot(self, node: int) -> int:
        """Return the slot index to overwrite when the pool is full.

        For policy="fifo" this is always 0 (oldest by insertion order), which
        matches the pre-refactor behaviour exactly.  For policy="task-floor"
        (opt-in diagnostic) we walk slots oldest->newest and skip any whose
        task_id is the sole surviving representative of that task on the node.
        For age-* policies we evict by insertion-index structure only, keeping
        the retention lever task-free.
        """
        if self.pool_evict_policy == "fifo":
            return 0
        if self.pool_evict_policy == "age-diversity":
            return self._choose_age_diversity_evict_slot(node)
        if self.pool_evict_policy == "age-diversity-oldest-floor":
            return self._choose_age_diversity_oldest_floor_evict_slot(node)
        if self.pool_evict_policy == "age-bucket-floor":
            return self._choose_age_bucket_floor_evict_slot(node)
        task_ids = self.pool_task_ids[node].tolist()
        counts: dict[int, int] = {}
        for tid in task_ids:
            counts[tid] = counts.get(tid, 0) + 1
        for slot, tid in enumerate(task_ids):
            if counts.get(tid, 0) >= 2:
                return slot
        return 0

    def _shift_slots_down_for_evict(self, node: int, evict_slot: int) -> None:
        """Shift slots `[evict_slot+1:]` one step left into `[evict_slot:-1]`.

        Leaves slots `[0:evict_slot]` untouched.  For evict_slot=0 this
        produces the exact pre-refactor block-shift (the FIFO path), so
        the 30.9 acceptance result is preserved bit-identically.  For
        evict_slot>0 only the slots above the evicted index move; the
        slot at `pool_capacity - 1` is always available for the new
        snapshot after the shift returns.
        """
        if self.pool_capacity <= 1 or evict_slot >= self.pool_capacity - 1:
            return
        s = evict_slot
        self.pool_snapshots[node, s:-1].copy_(
            self.pool_snapshots[node, s + 1:].clone()
        )
        self.pool_levels[node, s:-1].copy_(self.pool_levels[node, s + 1:].clone())
        self.pool_task_ids[node, s:-1].copy_(self.pool_task_ids[node, s + 1:].clone())
        self.pool_insert_indices[node, s:-1].copy_(
            self.pool_insert_indices[node, s + 1:].clone()
        )
        self.pool_res_sizes[node, s:-1].copy_(self.pool_res_sizes[node, s + 1:].clone())
        self.pool_res_z[node, s:-1].copy_(self.pool_res_z[node, s + 1:].clone())
        self.pool_res_p_prev[node, s:-1].copy_(
            self.pool_res_p_prev[node, s + 1:].clone()
        )
        self.pool_res_symbols[node, s:-1].copy_(
            self.pool_res_symbols[node, s + 1:].clone()
        )
        self.model_log_probs[node, s : self.pool_capacity - 1].copy_(
            self.model_log_probs[node, s + 1 : self.pool_capacity].clone()
        )

    @_profiled("layer.insert_pool_snapshot_fifo")
    def _insert_pool_snapshot_fifo(
        self,
        node: int,
        weights: torch.Tensor,
        score: torch.Tensor,
        deepest_level: int,
        task_id: int,
        global_index: int,
        level: int | None = None,
    ) -> None:
        k = int(self.pool_sizes[node].item())
        if k < self.pool_capacity:
            insert = k
            self.pool_sizes[node] = k + 1
            self.provenance_event_counts["append"] += 1
        else:
            evict_slot = self._choose_evict_slot(node)
            evicted_task = int(self.pool_task_ids[node, evict_slot].item())
            self.provenance_evicted_task_counts[evicted_task] = (
                self.provenance_evicted_task_counts.get(evicted_task, 0) + 1
            )
            self.provenance_event_counts["evict"] += 1
            self._shift_slots_down_for_evict(node, evict_slot)
            insert = self.pool_capacity - 1

        self._record_pool_slot(
            node, insert, weights, score, deepest_level, task_id, global_index
        )
        self._store_segment_reservoir(node, insert, level)

    def _apply_pool_update(
        self,
        node: int,
        k: int,
        best_slot: int,
        best_weights: torch.Tensor,
        best_score: torch.Tensor,
        deepest_level: int,
        task_id: int,
        global_index: int,
        level: int | None = None,
    ) -> None:
        if self.pool_capacity <= 0:
            return
        action, slot = self._choose_paper_pool_action(node, k, best_slot, best_weights, level)
        if action == "skip":
            self.provenance_event_counts["skip"] += 1
            return
        if action == "refine":
            assert slot is not None
            self.provenance_event_counts["refine"] += 1
            self._record_pool_slot(
                node, slot, best_weights, best_score, deepest_level, task_id, global_index
            )
            self._merge_pool_and_segment_reservoirs(node, slot, level)
            return
        self._insert_pool_snapshot_fifo(
            node, best_weights, best_score, deepest_level, task_id, global_index, level
        )

    @torch.no_grad()
    @_profiled("layer.reset_active_models_from_snapshots")
    def _reset_active_models_from_snapshots(self, n: int, level: int | None = None) -> None:
        """Open a new segment for node ``n`` (and optionally one PTW level).

        Pool slots are restored from their immutable snapshots; the fresh
        slot's weights persist (it is the continuously-updated FMN base
        measure ρ).  All active slots' current-segment posterior is reset
        because FMN Eq. 5 mixes models by their *current-segment* likelihood
        only.  Stale all-time likelihood in the fresh slot would skew the
        next segment's priors and suppress adaptation at segment open.
        """
        if self.per_level_active and level is not None:
            idx = self._level_index(level)
            assert self.level_mixture_weights is not None
            assert self.level_pool_sizes is not None
            assert self.level_segment_log_probs is not None
            k = int(self.pool_sizes[n].item())
            fresh_idx = self.fresh_idx
            if k > 0:
                self.level_mixture_weights[idx, n, :k].copy_(self.pool_snapshots[n, :k])
            for slot in range(k, fresh_idx):
                self.level_mixture_weights[idx, n, slot].zero_()
            self.level_pool_sizes[idx, n] = k
            self.level_segment_log_probs[idx, n].zero_()
            return

        k = int(self.pool_sizes[n].item())
        fresh_idx = self.fresh_idx
        if k > 0:
            self.mixture_weights[n, :k].copy_(self.pool_snapshots[n, :k])
        for slot in range(k, fresh_idx):
            self.mixture_weights[n, slot].zero_()
        # Fresh weights persist (base measure rho); ALL active slots restart
        # with zero current-segment evidence so the FMN posterior priors
        # (1/2 fresh, 1/(2k) per pool model) are honoured at segment open.
        self.segment_log_probs[n].zero_()

    @torch.no_grad()
    @_profiled("layer.reset_level_active_state_all_nodes")
    def _reset_level_active_state_all_nodes(self, level: int) -> None:
        """Vectorised per-level reset across all nodes (close-only path).

        For PTW levels that close but are *not* eligible for
        ``UPDATEMODELPOOL`` (paper's ``2**c`` skip), we only need to restore
        the per-level active state from the shared pool snapshots and zero the
        per-level segment evidence and reservoir.  This avoids the per-node
        Python loop that the update path requires.

        No-op when per-level active state is not allocated (``per_level_active``
        is ``False``).
        """
        if not self.per_level_active:
            return
        assert self.level_mixture_weights is not None
        assert self.level_pool_sizes is not None
        assert self.level_segment_log_probs is not None
        idx = self._level_index(level)
        # Copy current pool snapshots into every node's per-level active slot
        # state.  Slots beyond each node's pool size are zeroed below; the
        # fresh slot (index ``self.fresh_idx``) is intentionally untouched so
        # the FMN base measure rho keeps learning across closes.
        # `pool_snapshots`: [N, pool_capacity, C, K]
        # `level_mixture_weights[idx]`: [N, pool_capacity + 1, C, K]
        self.level_mixture_weights[idx, :, :self.pool_capacity].copy_(
            self.pool_snapshots
        )
        # Zero unused slots beyond per-node pool sizes; the fresh slot weights
        # are intentionally preserved (FMN base measure rho keeps learning).
        # Branchless on-device mask: a per-close \`.any().item()\` here forces
        # a CUDA->host sync, which serialises Phase 1's 12,664 close events
        # per task.  An unconditional masked\_fill of the [N, pool_capacity]
        # block is one fused kernel; when the mask is all-False the cost is
        # a tiny no-op launch, but no host sync.
        if self.pool_capacity > 0:
            sizes = self.pool_sizes  # [N]
            slot_range = torch.arange(self.pool_capacity, device=sizes.device)
            unused = slot_range.unsqueeze(0) >= sizes.unsqueeze(1)  # [N, pool_capacity]
            # Broadcast to [N, pool_capacity, C, K] and zero the slice in one op.
            mask = unused.view(self.N, self.pool_capacity, 1, 1)
            self.level_mixture_weights[idx, :, :self.pool_capacity].masked_fill_(mask, 0.0)
        # Per-node pool-size view follows the shared pool size at segment open.
        self.level_pool_sizes[idx].copy_(self.pool_sizes)
        # Zero current-segment evidence for every (node, slot).
        self.level_segment_log_probs[idx].zero_()
        # Drop the per-level segment reservoir as well.
        self._reset_segment_reservoir(idx)

    @torch.no_grad()
    def _apply_ptw_dp_close(self, close_levels: list[int]) -> None:
        """Apply Algorithm 1 line 5 (b_i <- w_{i+1}) and line 8 resets to the
        PTW DP state for every closing level.

        * Line 5 happens FIRST, capturing the pre-reset ``ptw_log_w[i+1]``
          where ``i = MSCB_d(t)``.  We recover ``i`` from ``close_levels[0] -
          1`` because :func:`mscb_close_levels` returns the contiguous range
          ``[MSCB_d(t)+1 .. d]`` in ascending order.
        * Line 8 then zeros ``level_log_nu[j]``, ``ptw_log_w[j]`` and
          ``ptw_log_b[j]`` for every ``j`` in ``close_levels`` (paper sets
          w_j, b_j, nu_{r_j} <- 1, which is 0 in log space).

        No-op when the PTW DP state is not allocated
        (``per_level_active=False``).
        """
        if not self.per_level_active:
            return
        assert self.level_log_nu is not None
        assert self.ptw_log_w is not None
        assert self.ptw_log_b is not None
        if not close_levels:
            return
        # Line 5: b_i <- w_{i+1}, recovered from the closing-level range.
        sorted_levels = sorted({self._level_index(j) for j in close_levels})
        i_plus_one = sorted_levels[0]
        if i_plus_one > 0:
            i = i_plus_one - 1
            self.ptw_log_b[i].copy_(self.ptw_log_w[i_plus_one])
        # Line 8: reset every closing level's DP triple.
        idx = torch.tensor(sorted_levels, dtype=torch.long, device=self.device)
        self.level_log_nu.index_fill_(0, idx, 0.0)
        self.ptw_log_w.index_fill_(0, idx, 0.0)
        self.ptw_log_b.index_fill_(0, idx, 0.0)

    @torch.no_grad()
    @_profiled("layer.segment_close")
    def segment_close(
        self,
        level: int | None = None,
        levels: list[int] | None = None,
        update_levels: list[int] | None = None,
        task_id: int | None = None,
        global_index: int | None = None,
    ):
        """Close PTW segments and optionally invoke ``UPDATEMODELPOOL``.

        Algorithm 1 lines 4--8 of the FMN paper close the PTW state for every
        level ``j in (MSCB_d(t)+1 .. d)`` *and* (when eligible under §3.3's
        ``2**c`` skip) call ``UPDATEMODELPOOL`` on each of them.  In paper
        mode this method honours both: ``levels`` (or ``level``) lists every
        level whose per-level active state must be reset, and ``update_levels``
        is the subset that is also eligible to update the bounded pool.  When
        ``update_levels`` is ``None`` (paper-faithful at ``min_segment <= 1``)
        all closing levels are also update-eligible.

        Non-paper/FIFO mode retains the historical diagnostic collapse of
        simultaneous PTW closes to one snapshot tagged with the deepest level.
        """
        if levels is None:
            tag = -1 if level is None else int(level)
            levels = [tag]
        if not levels:
            return
        close_levels = [int(l) for l in levels]
        if update_levels is None:
            update_levels_list = list(close_levels)
        else:
            update_levels_list = [int(l) for l in update_levels]
        # Apply Algorithm 1 line 5 + line 8 to the PTW DP state BEFORE the
        # pool/active-state work, so any pool-update path reads the post-reset
        # DP triples and pre-reset b_i.  This is a no-op when per_level_active
        # is False or close_levels is empty.
        self._apply_ptw_dp_close(close_levels)
        if self.pool_update_policy != "paper":
            # Historical FIFO mode: pick the deepest UPDATE-ELIGIBLE level as
            # the single remembered snapshot for the entire close event, OR
            # skip the close entirely if no level is update-eligible under the
            # ``2**c`` (``min_segment``) skip.  This preserves the legacy
            # diagnostic where small ``min_segment`` filters made the pool
            # insertion sparse on its own.  An earlier draft collapsed to
            # ``max(close_levels)`` and treated it as update-eligible, which
            # fired UPDATEMODELPOOL on every sample at depth=15 (a 256x
            # regression for the historical fifo + min_segment=512 configs).
            if not update_levels_list:
                return
            close_levels = [max(update_levels_list)]
            update_levels_list = list(close_levels)

        update_set = set(update_levels_list)
        provenance_task = -1 if task_id is None else int(task_id)
        provenance_index = -1 if global_index is None else int(global_index)

        # Paper Algorithm 1 lines 6--8 process closing levels in ascending
        # order: each level's reset sees the pool size AFTER all shallower
        # updates from the same close event.  The ``2**c`` skip retains a
        # contiguous prefix of levels as update-eligible, so doing the
        # eligible per-level updates first and the (deeper) close-only
        # resets second matches the paper ordering and lets the close-only
        # path see the post-update shared pool size.
        fresh_idx = self.fresh_idx
        for n in range(self.N):
            if self.per_level_active:
                for closing_level in close_levels:
                    if closing_level not in update_set:
                        continue  # handled after the per-node update loop.
                    level_idx = self._level_index(closing_level)
                    assert self.level_mixture_weights is not None
                    assert self.level_pool_sizes is not None
                    assert self.level_segment_log_probs is not None
                    k0 = int(self.level_pool_sizes[level_idx, n].item())
                    active_slots = list(range(k0)) + [fresh_idx]
                    scores = self.level_segment_log_probs[level_idx, n, active_slots]
                    best_slot = active_slots[int(torch.argmax(scores).item())]
                    best_weights = self.level_mixture_weights[
                        level_idx, n, best_slot
                    ].detach().clone()
                    best_score = self.level_segment_log_probs[
                        level_idx, n, best_slot
                    ].detach().clone()
                    self._apply_pool_update(
                        n,
                        int(self.pool_sizes[n].item()),
                        best_slot,
                        best_weights,
                        best_score,
                        closing_level,
                        provenance_task,
                        provenance_index,
                        level_idx,
                    )
                    self._reset_active_models_from_snapshots(n, level_idx)
                    self._reset_segment_reservoir(level_idx)
                continue

            k0 = int(self.pool_sizes[n].item())
            active_slots = list(range(k0)) + [fresh_idx]
            scores = self.segment_log_probs[n, active_slots]
            best_slot = active_slots[int(torch.argmax(scores).item())]
            best_weights = self.mixture_weights[n, best_slot].detach().clone()
            best_score = self.segment_log_probs[n, best_slot].detach().clone()

            for closing_level in close_levels:
                if closing_level not in update_set:
                    continue  # flat mode: still observe paper update skip.
                k = int(self.pool_sizes[n].item())
                self._apply_pool_update(
                    n,
                    k,
                    best_slot,
                    best_weights,
                    best_score,
                    closing_level,
                    provenance_task,
                    provenance_index,
                )
            self._reset_active_models_from_snapshots(n)

        if not self.per_level_active:
            self._reset_segment_reservoir()
            return

        # Close-only levels (those skipped by the ``2**c`` filter) reset after
        # all eligible updates so their per-level pool-size view reflects the
        # post-update shared pool size.  This matches paper Algorithm 1's
        # ascending-level processing because update_levels is always a
        # contiguous prefix of close_levels.
        for closing_level in close_levels:
            if closing_level in update_set:
                continue
            self._reset_level_active_state_all_nodes(closing_level)

    # ----- Snapshot / restore (Phase 5I-D) -----
    # Lists below MUST stay aligned with __init__ + _enable_per_level_active.
    # Immutable tensors (hyperplanes, hp_bias) and pure hyperparameters are
    # intentionally excluded; nothing under training mutates them.
    _SNAPSHOT_DEVICE_TENSORS = (
        "pool_snapshots",
        "pool_levels",
        "pool_task_ids",
        "pool_insert_indices",
        "mixture_weights",
        "pool_sizes",
        "segment_log_probs",
        "model_log_probs",
        "level_mixture_weights",
        "level_pool_sizes",
        "level_segment_log_probs",
        "level_model_log_probs",
        "level_log_nu",
        "ptw_log_w",
        "ptw_log_b",
    )
    _SNAPSHOT_CPU_TENSORS = (
        "segment_res_z",
        "segment_res_p_prev",
        "segment_res_symbols",
        "pool_res_sizes",
        "pool_res_z",
        "pool_res_p_prev",
        "pool_res_symbols",
        "level_res_seen",
        "level_res_size",
        "level_res_z",
        "level_res_p_prev",
        "level_res_symbols",
    )
    _SNAPSHOT_SCALARS = (
        "segment_res_seen",
        "segment_res_size",
    )

    def snapshot_state(self) -> dict[str, object]:
        """Return a CPU-resident snapshot of every mutable attribute.

        Used by NctlNetwork.snapshot_state to support the in-place
        evaluate_task path without doubling GPU memory.  The snapshot is
        a dict mapping attribute names to either CPU tensors (cloned) or
        plain Python scalars / dicts.  None-valued attributes are stored
        as None so restore can preserve the absent state.
        """
        snap: dict[str, object] = {}
        for name in self._SNAPSHOT_DEVICE_TENSORS:
            t = getattr(self, name, None)
            snap[name] = None if t is None else t.detach().to("cpu", copy=True)
        for name in self._SNAPSHOT_CPU_TENSORS:
            t = getattr(self, name, None)
            snap[name] = None if t is None else t.detach().clone()
        for name in self._SNAPSHOT_SCALARS:
            snap[name] = getattr(self, name)
        snap["provenance_event_counts"] = dict(self.provenance_event_counts)
        snap["provenance_evicted_task_counts"] = dict(self.provenance_evicted_task_counts)
        snap["reservoir_rng_state"] = self._reservoir_rng.get_state().clone()
        return snap

    def restore_state(self, snap: dict[str, object]) -> None:
        """Restore in-place from a snapshot produced by snapshot_state.

        Tensor restores use ``.copy_(...)`` so the original storage is
        reused; no new GPU allocations occur on the restore path.
        """
        for name in self._SNAPSHOT_DEVICE_TENSORS:
            cur = getattr(self, name, None)
            saved = snap.get(name)
            if cur is None and saved is None:
                continue
            if cur is None or saved is None:
                # Topology change between snapshot and restore is a bug.
                raise RuntimeError(
                    f"restore_state: tensor '{name}' presence mismatch "
                    f"(current={cur is None}, snapshot={saved is None})"
                )
            cur.copy_(saved.to(cur.device, non_blocking=True))
        for name in self._SNAPSHOT_CPU_TENSORS:
            cur = getattr(self, name, None)
            saved = snap.get(name)
            if cur is None and saved is None:
                continue
            if cur is None or saved is None:
                raise RuntimeError(
                    f"restore_state: cpu tensor '{name}' presence mismatch"
                )
            cur.copy_(saved)
        for name in self._SNAPSHOT_SCALARS:
            setattr(self, name, snap[name])
        self.provenance_event_counts = dict(snap["provenance_event_counts"])
        self.provenance_evicted_task_counts = dict(snap["provenance_evicted_task_counts"])
        self._reservoir_rng.set_state(snap["reservoir_rng_state"])

    def clone(self):
        c = CudaFmnMixtureLayer.__new__(CudaFmnMixtureLayer)
        c.N, c.K, c.D, c.H, c.C, c.M, c.lr = (
            self.N,
            self.K,
            self.D,
            self.H,
            self.C,
            self.M,
            self.lr,
        )
        c.device = self.device
        c.pool_capacity = self.pool_capacity
        c.posterior_temp = self.posterior_temp
        c.pool_update_policy = self.pool_update_policy
        c.pool_alpha = self.pool_alpha
        c.pool_beta = self.pool_beta
        c.pool_reservoir_size = self.pool_reservoir_size
        c.pool_evict_policy = self.pool_evict_policy
        c.pool_oldest_floor = self.pool_oldest_floor
        c.active_state_mode = self.active_state_mode
        c.per_level_active = self.per_level_active
        c.active_level_count = self.active_level_count
        c.prediction_level = self.prediction_level
        c.hyperplanes = self.hyperplanes
        c.hp_bias = self.hp_bias
        c.pool_snapshots = self.pool_snapshots.clone()
        c.pool_levels = self.pool_levels.clone()
        c.pool_task_ids = self.pool_task_ids.clone()
        c.pool_insert_indices = self.pool_insert_indices.clone()
        c.provenance_event_counts = dict(self.provenance_event_counts)
        c.provenance_evicted_task_counts = dict(self.provenance_evicted_task_counts)
        c._reservoir_rng = torch.Generator(device="cpu")
        c._reservoir_rng.set_state(self._reservoir_rng.get_state())
        c.segment_res_seen = self.segment_res_seen
        c.segment_res_size = self.segment_res_size
        c.segment_res_z = self.segment_res_z.clone()
        c.segment_res_p_prev = self.segment_res_p_prev.clone()
        c.segment_res_symbols = self.segment_res_symbols.clone()
        c.pool_res_sizes = self.pool_res_sizes.clone()
        c.pool_res_z = self.pool_res_z.clone()
        c.pool_res_p_prev = self.pool_res_p_prev.clone()
        c.pool_res_symbols = self.pool_res_symbols.clone()
        c.mixture_weights = self.mixture_weights.clone()
        c.pool_sizes = self.pool_sizes.clone()
        c.segment_log_probs = self.segment_log_probs.clone()
        c.model_log_probs = self.model_log_probs.clone()
        c.level_mixture_weights = (
            None if self.level_mixture_weights is None else self.level_mixture_weights.clone()
        )
        c.level_pool_sizes = None if self.level_pool_sizes is None else self.level_pool_sizes.clone()
        c.level_segment_log_probs = (
            None if self.level_segment_log_probs is None else self.level_segment_log_probs.clone()
        )
        c.level_model_log_probs = (
            None if self.level_model_log_probs is None else self.level_model_log_probs.clone()
        )
        c.level_res_seen = None if self.level_res_seen is None else self.level_res_seen.clone()
        c.level_res_size = None if self.level_res_size is None else self.level_res_size.clone()
        c.level_res_z = None if self.level_res_z is None else self.level_res_z.clone()
        c.level_res_p_prev = None if self.level_res_p_prev is None else self.level_res_p_prev.clone()
        c.level_res_symbols = None if self.level_res_symbols is None else self.level_res_symbols.clone()
        c.prediction_mode = self.prediction_mode
        c.level_log_nu = None if self.level_log_nu is None else self.level_log_nu.clone()
        c.ptw_log_w = None if self.ptw_log_w is None else self.ptw_log_w.clone()
        c.ptw_log_b = None if self.ptw_log_b is None else self.ptw_log_b.clone()
        return c

    def state_dict(self) -> dict[str, Any]:
        return {
            "N": self.N,
            "K": self.K,
            "D": self.D,
            "H": self.H,
            "C": self.C,
            "M": self.M,
            "lr": self.lr,
            "pool_capacity": self.pool_capacity,
            "posterior_temp": self.posterior_temp,
            "pool_update_policy": self.pool_update_policy,
            "pool_alpha": self.pool_alpha,
            "pool_beta": self.pool_beta,
            "pool_reservoir_size": self.pool_reservoir_size,
            "pool_evict_policy": self.pool_evict_policy,
            "pool_oldest_floor": self.pool_oldest_floor,
            "active_state_mode": self.active_state_mode,
            "active_level_count": self.active_level_count,
            "prediction_level": self.prediction_level,
            "hyperplanes": self.hyperplanes.detach().cpu(),
            "hp_bias": self.hp_bias.detach().cpu(),
            "pool_snapshots": self.pool_snapshots.detach().cpu(),
            "pool_levels": self.pool_levels.detach().cpu(),
            "pool_task_ids": self.pool_task_ids.detach().cpu(),
            "pool_insert_indices": self.pool_insert_indices.detach().cpu(),
            "provenance_event_counts": dict(self.provenance_event_counts),
            "provenance_evicted_task_counts": dict(self.provenance_evicted_task_counts),
            "pool_res_sizes": self.pool_res_sizes.detach().cpu(),
            "pool_res_z": self.pool_res_z.detach().cpu(),
            "pool_res_p_prev": self.pool_res_p_prev.detach().cpu(),
            "pool_res_symbols": self.pool_res_symbols.detach().cpu(),
            "reservoir_rng_state": self._reservoir_rng.get_state(),
            "mixture_weights": self.mixture_weights.detach().cpu(),
            "pool_sizes": self.pool_sizes.detach().cpu(),
            "segment_log_probs": self.segment_log_probs.detach().cpu(),
            "model_log_probs": self.model_log_probs.detach().cpu(),
            "level_mixture_weights": None if self.level_mixture_weights is None else self.level_mixture_weights.detach().cpu(),
            "level_pool_sizes": None if self.level_pool_sizes is None else self.level_pool_sizes.detach().cpu(),
            "level_segment_log_probs": None if self.level_segment_log_probs is None else self.level_segment_log_probs.detach().cpu(),
            "level_model_log_probs": None if self.level_model_log_probs is None else self.level_model_log_probs.detach().cpu(),
            "level_res_seen": None if self.level_res_seen is None else self.level_res_seen.detach().cpu(),
            "level_res_size": None if self.level_res_size is None else self.level_res_size.detach().cpu(),
            "level_res_z": None if self.level_res_z is None else self.level_res_z.detach().cpu(),
            "level_res_p_prev": None if self.level_res_p_prev is None else self.level_res_p_prev.detach().cpu(),
            "level_res_symbols": None if self.level_res_symbols is None else self.level_res_symbols.detach().cpu(),
            "prediction_mode": self.prediction_mode,
            "level_log_nu": None if self.level_log_nu is None else self.level_log_nu.detach().cpu(),
            "ptw_log_w": None if self.ptw_log_w is None else self.ptw_log_w.detach().cpu(),
            "ptw_log_b": None if self.ptw_log_b is None else self.ptw_log_b.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], device: torch.device):
        c = cls.__new__(cls)
        c.N = int(state["N"])
        c.K = int(state["K"])
        c.D = int(state["D"])
        c.H = int(state["H"])
        c.C = int(state["C"])
        c.M = int(state["M"])
        c.lr = float(state["lr"])
        c.pool_capacity = int(state["pool_capacity"])
        c.posterior_temp = float(state.get("posterior_temp", 1.0))
        c.pool_update_policy = str(state.get("pool_update_policy", "fifo"))
        c.pool_alpha = float(state.get("pool_alpha", math.inf))
        c.pool_beta = float(state.get("pool_beta", math.inf))
        c.pool_reservoir_size = int(state.get("pool_reservoir_size", 0))
        c.pool_evict_policy = str(state.get("pool_evict_policy", "fifo"))
        if c.pool_evict_policy not in POOL_EVICT_POLICIES:
            raise ValueError(
                f"unknown pool_evict_policy in state_dict: {c.pool_evict_policy!r}"
            )
        c.pool_oldest_floor = int(
            state.get("pool_oldest_floor", DEFAULT_POOL_OLDEST_FLOOR)
        )
        c.active_state_mode = str(state.get("active_state_mode", "flat"))
        c.per_level_active = c.active_state_mode == "per_level"
        c.active_level_count = int(state.get("active_level_count", 1))
        c.prediction_level = int(state.get("prediction_level", max(0, c.active_level_count - 1)))
        c.prediction_mode = str(state.get("prediction_mode", "selected_level"))
        if c.prediction_mode not in {"selected_level", "ptw_dp"}:
            raise ValueError(f"unknown prediction_mode in state_dict: {c.prediction_mode}")
        if c.prediction_mode == "ptw_dp" and not c.per_level_active:
            raise ValueError(
                "state_dict has prediction_mode='ptw_dp' but active_state_mode is not 'per_level'"
            )
        c.device = device
        c.hyperplanes = state["hyperplanes"].to(device)
        c.hp_bias = state["hp_bias"].to(device)
        c.pool_snapshots = state["pool_snapshots"].to(device)
        c.pool_levels = state.get(
            "pool_levels",
            torch.full(
                (c.N, c.pool_capacity), -1, dtype=torch.int32
            ),
        ).to(device)
        c.pool_task_ids = state.get(
            "pool_task_ids",
            torch.full((c.N, c.pool_capacity), -1, dtype=torch.int32),
        ).to(device)
        c.pool_insert_indices = state.get(
            "pool_insert_indices",
            torch.full((c.N, c.pool_capacity), -1, dtype=torch.int64),
        ).to(device)
        c.provenance_event_counts = dict(
            state.get(
                "provenance_event_counts",
                {"append": 0, "evict": 0, "refine": 0, "skip": 0},
            )
        )
        c.provenance_evicted_task_counts = {
            int(k): int(v)
            for k, v in state.get("provenance_evicted_task_counts", {}).items()
        }
        c._reservoir_rng = torch.Generator(device="cpu")
        if "reservoir_rng_state" in state:
            c._reservoir_rng.set_state(state["reservoir_rng_state"])
        c.segment_res_seen = 0
        c.segment_res_size = 0
        c.segment_res_z = torch.empty(c.pool_reservoir_size, c.D, device="cpu")
        c.segment_res_p_prev = torch.empty(c.pool_reservoir_size, c.K, device="cpu")
        c.segment_res_symbols = torch.empty(c.pool_reservoir_size, dtype=torch.int32, device="cpu")
        c.pool_res_sizes = state.get(
            "pool_res_sizes",
            torch.zeros(c.N, c.pool_capacity, dtype=torch.int32),
        ).to("cpu")
        c.pool_res_z = state.get(
            "pool_res_z",
            torch.empty(c.N, c.pool_capacity, c.pool_reservoir_size, c.D),
        ).to("cpu")
        c.pool_res_p_prev = state.get(
            "pool_res_p_prev",
            torch.empty(c.N, c.pool_capacity, c.pool_reservoir_size, c.K),
        ).to("cpu")
        c.pool_res_symbols = state.get(
            "pool_res_symbols",
            torch.empty(c.N, c.pool_capacity, c.pool_reservoir_size, dtype=torch.int32),
        ).to("cpu")
        c.mixture_weights = state["mixture_weights"].to(device)
        c.pool_sizes = state["pool_sizes"].to(device)
        c.segment_log_probs = state["segment_log_probs"].to(device)
        c.model_log_probs = state["model_log_probs"].to(device)
        if c.per_level_active:
            c.level_mixture_weights = state["level_mixture_weights"].to(device)
            c.level_pool_sizes = state["level_pool_sizes"].to(device)
            c.level_segment_log_probs = state["level_segment_log_probs"].to(device)
            c.level_model_log_probs = state["level_model_log_probs"].to(device)
            c.level_res_seen = state["level_res_seen"].to("cpu")
            c.level_res_size = state["level_res_size"].to("cpu")
            c.level_res_z = state["level_res_z"].to("cpu")
            c.level_res_p_prev = state["level_res_p_prev"].to("cpu")
            c.level_res_symbols = state["level_res_symbols"].to("cpu")
            # DP state: back-compat default is freshly zeroed tensors when
            # the snapshot pre-dates Phase 2.
            zero = torch.zeros(c.active_level_count, c.N, dtype=torch.float64, device=device)
            c.level_log_nu = state.get("level_log_nu", zero).to(device) if state.get("level_log_nu") is not None else zero.clone()
            c.ptw_log_w  = state.get("ptw_log_w",  zero).to(device) if state.get("ptw_log_w")  is not None else zero.clone()
            c.ptw_log_b  = state.get("ptw_log_b",  zero).to(device) if state.get("ptw_log_b")  is not None else zero.clone()
        else:
            c.level_mixture_weights = None
            c.level_pool_sizes = None
            c.level_segment_log_probs = None
            c.level_model_log_probs = None
            c.level_res_seen = None
            c.level_res_size = None
            c.level_res_z = None
            c.level_res_p_prev = None
            c.level_res_symbols = None
            c.level_log_nu = None
            c.ptw_log_w = None
            c.ptw_log_b = None
        return c


    def provenance_summary(self) -> dict[str, Any]:
        """Return JSON-friendly bounded-pool provenance diagnostics."""
        pool_sizes = self.pool_sizes.detach().cpu().to(torch.int64)
        task_ids = self.pool_task_ids.detach().cpu().to(torch.int64)
        levels = self.pool_levels.detach().cpu().to(torch.int64)
        insert_indices = self.pool_insert_indices.detach().cpu().to(torch.int64)

        task_hist: dict[str, int] = {}
        level_hist: dict[str, int] = {}
        slot_task_hist: list[dict[str, int]] = [
            {} for _ in range(self.pool_capacity)
        ]
        for n in range(self.N):
            k = int(pool_sizes[n].item())
            for slot in range(k):
                task = int(task_ids[n, slot].item())
                level = int(levels[n, slot].item())
                task_key = str(task)
                level_key = str(level)
                task_hist[task_key] = task_hist.get(task_key, 0) + 1
                level_hist[level_key] = level_hist.get(level_key, 0) + 1
                slot_task_hist[slot][task_key] = slot_task_hist[slot].get(task_key, 0) + 1

        return {
            "pool_size_total": int(pool_sizes.sum().item()),
            "pool_capacity_total": int(self.N * self.pool_capacity),
            "pool_task_histogram": task_hist,
            "pool_level_histogram": level_hist,
            "slot_task_histograms": slot_task_hist,
            "pool_task_ids": task_ids.tolist(),
            "pool_insert_indices": insert_indices.tolist(),
            "reservoir_size": int(self.pool_reservoir_size),
            "active_state_mode": self.active_state_mode,
            "active_level_count": int(self.active_level_count),
            "prediction_level": int(self.prediction_level),
            "pool_reservoir_sizes": self.pool_res_sizes.to(torch.int64).tolist(),
            "event_counts": {k: int(v) for k, v in self.provenance_event_counts.items()},
            "evicted_task_counts": {
                str(k): int(v) for k, v in self.provenance_evicted_task_counts.items()
            },
        }


class NctlNetwork:
    """NCTL with CUDA FMN mixture layers.

    Chunk processing is split using Algorithm 1's MSCB carry pattern, with the
    FMN paper's optional ``min_segment`` skip heuristic to avoid remembering
    very short segments.  By default this uses the historical flat active-state
    approximation; ``active_state_mode=per_level`` gives each retained PTW level
    independent active mixture state while keeping the global model pool shared.
    """

    def __init__(
        self,
        layer_sizes,
        input_dim,
        num_halfspaces,
        lr,
        pool_capacity=8,
        min_segment=32,
        ptw_depth=15,
        posterior_temp=1.0,
        pool_update_policy="fifo",
        pool_alpha=math.inf,
        pool_beta=math.inf,
        pool_reservoir_size=64,
        pool_evict_policy: str = "fifo",
        pool_oldest_floor: int = DEFAULT_POOL_OLDEST_FLOOR,
        output_pool_capacity: int | None = None,
        active_state_mode: str = "flat",
        prediction_mode: str = "selected_level",
        device=None,
        seed=42,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.input_dim = input_dim
        self.layer_sizes = list(layer_sizes)
        self.num_halfspaces = num_halfspaces
        self.lr = lr
        self.pool_capacity = int(pool_capacity)
        self.output_pool_capacity = (
            int(output_pool_capacity)
            if output_pool_capacity is not None
            else self.pool_capacity
        )
        if self.pool_capacity <= 0:
            raise ValueError("pool_capacity must be positive")
        if self.output_pool_capacity <= 0:
            raise ValueError("output_pool_capacity must be positive")
        self.layer_pool_capacities = [self.pool_capacity] * len(self.layer_sizes)
        if self.layer_pool_capacities:
            self.layer_pool_capacities[-1] = self.output_pool_capacity
        self.min_segment = min_segment
        self.ptw_depth = ptw_depth
        self.posterior_temp = float(posterior_temp)
        self.pool_update_policy = str(pool_update_policy)
        self.pool_alpha = float(pool_alpha)
        self.pool_beta = float(pool_beta)
        self.pool_reservoir_size = int(pool_reservoir_size)
        if pool_evict_policy not in POOL_EVICT_POLICIES:
            raise ValueError(
                f"unknown pool_evict_policy: {pool_evict_policy!r}"
            )
        self.pool_evict_policy = pool_evict_policy
        self.pool_oldest_floor = max(0, int(pool_oldest_floor))
        if active_state_mode not in {"flat", "per_level"}:
            raise ValueError(f"unknown active_state_mode: {active_state_mode}")
        if prediction_mode not in {"selected_level", "ptw_dp"}:
            raise ValueError(f"unknown prediction_mode: {prediction_mode}")
        if prediction_mode == "ptw_dp" and active_state_mode != "per_level":
            raise ValueError(
                "prediction_mode='ptw_dp' requires active_state_mode='per_level'"
            )
        self.active_state_mode = active_state_mode
        self.prediction_mode = prediction_mode
        # Paper Algorithm 1 maintains PTW state for every level 0..d, and
        # the 2**c skip applies only to UPDATEMODELPOOL.  We therefore allocate
        # per-level active state for all depth+1 levels; min_segment continues
        # to gate UPDATEMODELPOOL via mscb_update_eligible_levels.  The
        # ``_active_level_count_for_min_segment`` name is retained as the
        # full-depth helper for compatibility with serialised state.
        self.active_level_count = self._active_level_count_for_min_segment(
            self.ptw_depth, self.min_segment
        )
        self.prediction_level = self.active_level_count - 1
        self.index = 0
        self.log_loss = 0.0
        self.segment_counter = 0
        self.samples_since_close = 0

        self.layers = []
        s = seed
        for li, nn in enumerate(self.layer_sizes):
            ki = input_dim if li == 0 else self.layer_sizes[li - 1]
            layer_pool_capacity = self.layer_pool_capacities[li]
            layer = CudaFmnMixtureLayer(
                nn,
                ki,
                input_dim,
                num_halfspaces,
                lr,
                layer_pool_capacity,
                device,
                s,
                posterior_temp=self.posterior_temp,
                pool_update_policy=self.pool_update_policy,
                pool_alpha=self.pool_alpha,
                pool_beta=self.pool_beta,
                pool_reservoir_size=self.pool_reservoir_size,
                pool_evict_policy=self.pool_evict_policy,
                pool_oldest_floor=self.pool_oldest_floor,
                active_state_mode=self.active_state_mode,
                active_level_count=self.active_level_count,
                prediction_level=self.prediction_level,
                prediction_mode=self.prediction_mode,
            )
            self.layers.append(layer)
            s += nn * 0x9E3779B9

    @staticmethod
    def _active_level_count_for_min_segment(depth: int, min_segment: int) -> int:
        """Number of per-level FMN active-state slots to allocate.

        Algorithm 1 of the FMN paper maintains PTW state ``w_j, b_j, r_j`` and
        per-level segment evidence for every level ``0 <= j <= d``.  The
        ``2**c`` (``min_segment``) skip described in §3.3 applies only to
        ``UPDATEMODELPOOL`` calls; the PTW recursion still closes and resets
        every level on its natural MSCB schedule.  We therefore always
        allocate ``depth + 1`` per-level slots and reserve ``min_segment`` for
        UPDATEMODELPOOL eligibility via :func:`mscb_update_eligible_levels`.

        ``min_segment`` is accepted for signature compatibility with
        previously serialised state but is intentionally ignored: shrinking
        the active-state set would re-introduce the paper mismatch that
        truncated PTW close/reset events along with UPDATEMODELPOOL skips.
        """
        del min_segment  # PTW state allocation is depth-driven, not skip-driven.
        return max(1, int(depth) + 1)

    def num_nodes(self):
        return sum(self.layer_sizes)

    def total_pool_size(self):
        return sum(l.pool_sizes.sum().item() for l in self.layers)

    @staticmethod
    def _mscb_boundary_plan(
        current_index: int, batch_len: int, depth: int, min_segment: int
    ):
        return mscb_boundary_plan(current_index, batch_len, depth, min_segment)

    @staticmethod
    def _coerce_symbols(symbols: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Place ``symbols`` on ``reference.device`` and cast to int32.

        The CUDA kernel binding dereferences ``symbols.data_ptr<int>()``, so
        the tensor must live on the same device as the GGM weights *and* be
        contiguous int32.  This helper centralises the conversion so that
        callers (notably ``run_split_mnist.py``) can pass int64 CPU labels and
        still drive the GPU pipeline.
        """
        if symbols.dtype != torch.int32 or symbols.device != reference.device:
            symbols = symbols.to(device=reference.device, dtype=torch.int32)
        return symbols.contiguous()

    @torch.no_grad()
    @_profiled("network.train_subchunk")
    def _train_subchunk(
        self,
        z_sub,
        symbols_sub,
        close_events: "list[tuple[int, list[int]]] | None" = None,
    ):
        # Match the input z device once up front so that the CUDA kernels
        # receive a device-resident contiguous int32 symbol buffer.
        symbols_sub = self._coerce_symbols(symbols_sub, z_sub)
        p_prev = torch.sigmoid(z_sub)
        final_psym = None
        for layer in self.layers:
            if hasattr(layer, "observe_segment_batch"):
                layer.observe_segment_batch(z_sub, p_prev, symbols_sub)
            # forward_update_batch returns P(observed symbol) for loss.  The
            # existing GPU NCTL implementation learns by then feeding the next
            # layer from the updated lower-layer P(symbol=1) predictions.
            # Phase 5F D-step 4: close_events are forwarded to the layer
            # so per-level FMN active state and the PTW DP state both
            # reset at the right offsets inside the sub-chunk, without
            # us having to cut the sub-chunk on every close event.
            # Only pass close_events when non-None so older/mock layers
            # with the pre-5F signature still work.
            if close_events is None:
                final_psym = layer.forward_update_batch(z_sub, p_prev, symbols_sub)
            else:
                final_psym = layer.forward_update_batch(
                    z_sub, p_prev, symbols_sub, close_events=close_events,
                )
            p_prev = layer.forward_only_batch(z_sub, p_prev)
        assert final_psym is not None
        return -torch.log(final_psym[:, 0].clamp(min=1e-30)).sum().item()

    @torch.no_grad()
    def close_open_segment(
        self,
        task_id: int | None = None,
        global_index: int | None = None,
        level: int | None = None,
        levels: list[int] | None = None,
        update_levels: list[int] | None = None,
    ) -> bool:
        """Commit the currently open flat FMN segment, if it has data.

        Sparse MSCB settings can leave a task-tail segment open at a known
        benchmark task boundary.  Closing it explicitly preserves the final
        task-specialized state before the next task starts adapting the active
        mixture.  The method is a no-op for empty segments, avoiding duplicate
        pool entries when a natural MSCB close already happened on the boundary.

        ``levels`` (or ``level``) lists the PTW levels whose per-level active
        state must be reset on this close (paper Algorithm 1 lines 4--6, 8).
        ``update_levels`` is the subset of those that are also eligible to call
        ``UPDATEMODELPOOL`` (paper §3.3 ``2**c`` skip); if ``None`` it defaults
        to ``levels`` (paper-faithful at ``min_segment <= 1``).
        """
        if getattr(self, "samples_since_close", 0) <= 0:
            return False
        close_index = self.index if global_index is None else int(global_index)
        if levels is None:
            close_level = self.ptw_depth if level is None else int(level)
            close_levels = [close_level]
        else:
            close_levels = [int(l) for l in levels]
        if update_levels is None:
            update_levels_arg = list(close_levels)
        else:
            update_levels_arg = [int(l) for l in update_levels]
        for layer in self.layers:
            layer.segment_close(
                levels=close_levels,
                update_levels=update_levels_arg,
                task_id=task_id,
                global_index=close_index,
            )
        self.samples_since_close = 0
        return True

    @torch.no_grad()
    @_profiled("network.train_chunk")
    def train_chunk(self, z_chunk, symbols_chunk, task_id: int | None = None):
        """Process samples and close FMN segments inside large user chunks.

        Phase 5F D-step 4 path (ptw_dp + per_level only): :func:`mscb_update_split_plan`
        carves the chunk into sub-chunks at UPDATEMODELPOOL boundaries only.
        Inside each sub-chunk, pure-close-only events ride along as
        ``close_events`` consumed by the per-layer DP path; the multilevel
        kernel applies per-level active-state resets in-kernel (when the
        Phase 5F .so is built) and the segmented DP combine resets the
        PTW DP state in-line.  This is the change that restores
        B ~ sub_chunk_size kernel-launch batching after Phase 1's
        per-sample close cadence at depth=15.

        Legacy path (selected_level or non-per_level active state): falls
        back to the original per-event sub-chunk split because the legacy
        path does not have an in-kernel reset and the segmented DP combine
        applies only to the DP state (which selected_level mode does not
        maintain).
        """
        if not hasattr(self, "samples_since_close"):
            # Some tests construct lightweight instances with __new__ to
            # exercise boundary planning without paying constructor cost.
            self.samples_since_close = 0
        B = z_chunk.size(0)
        # Lightweight test instances built via __new__ may lack
        # prediction_mode/active_state_mode entirely.  getattr keeps that
        # legacy path working (those tests exercise the legacy schedule).
        prediction_mode = getattr(self, "prediction_mode", "selected_level")
        active_state_mode = getattr(self, "active_state_mode", "flat")
        if prediction_mode == "ptw_dp" and active_state_mode == "per_level":
            # Phase 5F D-step 4 requires the in-kernel reset variant of
            # the multilevel kernel for correct per-level FMN active
            # state across close-only events.  When the .so predates 5F
            # (or isn't built), fall back to the legacy split-on-every-
            # event path so per-level resets fire via segment_close.
            ml = _get_fmn_multilevel_cuda()
            if ml is not None and hasattr(ml, "forward_update_with_resets"):
                return self._train_chunk_d_step4(z_chunk, symbols_chunk, task_id, B)
        return self._train_chunk_legacy(z_chunk, symbols_chunk, task_id, B)

    @torch.no_grad()
    @_profiled("network.train_chunk.legacy")
    def _train_chunk_legacy(self, z_chunk, symbols_chunk, task_id, B):
        events = self._mscb_boundary_plan(
            self.index, B, self.ptw_depth, self.min_segment
        )

        chunk_loss = 0.0
        start = 0
        for event in events:
            if len(event) == 3:
                offset, close_levels, update_levels = event
            else:  # legacy 2-tuple from test stubs
                offset, close_levels = event
                update_levels = list(close_levels)
            if offset > start:
                chunk_loss += self._train_subchunk(
                    z_chunk[start:offset], symbols_chunk[start:offset]
                )
                processed = offset - start
                self.index += processed
                self.segment_counter += processed
                self.samples_since_close += processed
            self.close_open_segment(
                task_id=task_id,
                global_index=self.index,
                levels=close_levels,
                update_levels=update_levels,
            )
            start = offset

        if start < B:
            chunk_loss += self._train_subchunk(z_chunk[start:B], symbols_chunk[start:B])
            processed = B - start
            self.index += processed
            self.segment_counter += processed
            self.samples_since_close += processed

        self.log_loss += chunk_loss
        return chunk_loss

    @torch.no_grad()
    @_profiled("network.train_chunk.d_step4")
    def _train_chunk_d_step4(self, z_chunk, symbols_chunk, task_id, B):
        plan = mscb_update_split_plan(
            self.index, B, self.ptw_depth, self.min_segment
        )

        chunk_loss = 0.0
        for (start, end, boundary_close_levels, boundary_update_levels, close_only_events) in plan:
            if end > start:
                # Convert close_only_events from chunk-local offsets to
                # sub-chunk-local offsets for the layer-level API.
                local_events: list[tuple[int, list[int]]] = [
                    (offset - start, list(levels))
                    for offset, levels in close_only_events
                    if start <= offset < end
                ]
                chunk_loss += self._train_subchunk(
                    z_chunk[start:end],
                    symbols_chunk[start:end],
                    close_events=local_events if local_events else None,
                )
                processed = end - start
                self.index += processed
                self.segment_counter += processed
                # samples_since_close diagnostic semantics from the
                # legacy path: every close event (UPDATE or close-only)
                # resets the counter, then subsequent samples increment
                # it.  We compute the post-sub-chunk value as the number
                # of samples after the LAST close_only event in this
                # sub-chunk; if none, it accumulates from the pre-sub-
                # chunk value.
                if local_events:
                    last_event_offset = max(o for o, _ in local_events)
                    # Samples [last_event_offset, end-start) lie after the
                    # last close-only event.  At last_event_offset itself
                    # the close fires BEFORE the sample, so that sample
                    # contributes to the post-close counter.
                    self.samples_since_close = (end - start) - last_event_offset
                else:
                    self.samples_since_close += processed
            # If this sub-chunk has a trailing UPDATE boundary, fire
            # segment_close for the full close_levels set (resetting PTW
            # state for every closing level) AND invoking UPDATEMODELPOOL
            # on the update-eligible subset.  This is the same call the
            # legacy path makes, just done once per sub-chunk instead of
            # once per close event.
            if boundary_close_levels:
                self.close_open_segment(
                    task_id=task_id,
                    global_index=self.index,
                    levels=boundary_close_levels,
                    update_levels=boundary_update_levels,
                )

        self.log_loss += chunk_loss
        return chunk_loss

    @torch.no_grad()
    def predict_batch(self, z_batch):
        """Forward-only for evaluation. Returns [B] P(symbol=1)."""
        p_prev = torch.sigmoid(z_batch)
        for layer in self.layers:
            p_prev = layer.forward_only_batch(z_batch, p_prev)
        return p_prev[:, 0]

    # ----- Snapshot / restore (Phase 5I-D) -----
    _SNAPSHOT_NETWORK_SCALARS = (
        "index",
        "log_loss",
        "segment_counter",
        "samples_since_close",
    )

    def snapshot_state(self) -> dict[str, object]:
        """Return a CPU-resident snapshot of the whole network.

        Per-layer snapshots dominate; the network-level snapshot adds
        only a handful of integer counters.  Used by evaluate_task to
        run a 50-sample adapt + predict in-place without doubling GPU
        memory via clone().
        """
        return {
            "layers": [layer.snapshot_state() for layer in self.layers],
            "scalars": {name: getattr(self, name) for name in self._SNAPSHOT_NETWORK_SCALARS},
        }

    def restore_state(self, snap: dict[str, object]) -> None:
        """Restore in-place from a snapshot produced by snapshot_state."""
        layer_snaps = snap["layers"]
        if len(layer_snaps) != len(self.layers):
            raise RuntimeError(
                f"restore_state: layer count mismatch "
                f"({len(layer_snaps)} snapshot vs {len(self.layers)} live)"
            )
        for layer, lsnap in zip(self.layers, layer_snaps):
            layer.restore_state(lsnap)
        for name, value in snap["scalars"].items():
            setattr(self, name, value)

    def clone(self):
        c = NctlNetwork.__new__(NctlNetwork)
        c.device = self.device
        c.input_dim = self.input_dim
        c.layer_sizes = list(self.layer_sizes)
        c.num_halfspaces = self.num_halfspaces
        c.lr = self.lr
        c.pool_capacity = self.pool_capacity
        c.output_pool_capacity = self.output_pool_capacity
        c.layer_pool_capacities = list(self.layer_pool_capacities)
        c.min_segment = self.min_segment
        c.ptw_depth = self.ptw_depth
        c.posterior_temp = self.posterior_temp
        c.pool_update_policy = self.pool_update_policy
        c.pool_alpha = self.pool_alpha
        c.pool_beta = self.pool_beta
        c.pool_reservoir_size = self.pool_reservoir_size
        c.pool_evict_policy = self.pool_evict_policy
        c.pool_oldest_floor = self.pool_oldest_floor
        c.active_state_mode = self.active_state_mode
        c.active_level_count = self.active_level_count
        c.prediction_level = self.prediction_level
        c.prediction_mode = self.prediction_mode
        c.index = self.index
        c.log_loss = self.log_loss
        c.segment_counter = self.segment_counter
        c.samples_since_close = self.samples_since_close
        c.layers = [l.clone() for l in self.layers]
        return c

    def provenance_summary(self) -> dict[str, Any]:
        layers = [layer.provenance_summary() for layer in self.layers]
        total_task_hist: dict[str, int] = {}
        total_events: dict[str, int] = {}
        total_evicted: dict[str, int] = {}
        for layer in layers:
            for task, count in layer["pool_task_histogram"].items():
                total_task_hist[task] = total_task_hist.get(task, 0) + int(count)
            for action, count in layer["event_counts"].items():
                total_events[action] = total_events.get(action, 0) + int(count)
            for task, count in layer["evicted_task_counts"].items():
                total_evicted[task] = total_evicted.get(task, 0) + int(count)
        return {
            "task_histogram": total_task_hist,
            "event_counts": total_events,
            "evicted_task_counts": total_evicted,
            "pool_update_policy": self.pool_update_policy,
            "pool_alpha": self.pool_alpha,
            "pool_beta": self.pool_beta,
            "pool_reservoir_size": self.pool_reservoir_size,
            "pool_evict_policy": self.pool_evict_policy,
            "pool_oldest_floor": self.pool_oldest_floor,
            "active_state_mode": self.active_state_mode,
            "active_level_count": self.active_level_count,
            "prediction_level": self.prediction_level,
            "prediction_mode": self.prediction_mode,
            "pool_capacity": self.pool_capacity,
            "output_pool_capacity": self.output_pool_capacity,
            "layer_pool_capacities": list(self.layer_pool_capacities),
            "layers": layers,
        }

    def save_snapshot(self, path: str | Path, metadata: dict[str, Any] | None = None):
        path = Path(path)
        payload = {
            "metadata": metadata or {},
            "network": {
                "input_dim": self.input_dim,
                "layer_sizes": self.layer_sizes,
                "num_halfspaces": self.num_halfspaces,
                "lr": self.lr,
                "pool_capacity": self.pool_capacity,
                "output_pool_capacity": self.output_pool_capacity,
                "layer_pool_capacities": list(self.layer_pool_capacities),
                "min_segment": self.min_segment,
                "ptw_depth": self.ptw_depth,
                "posterior_temp": self.posterior_temp,
                "pool_update_policy": self.pool_update_policy,
                "pool_alpha": self.pool_alpha,
                "pool_beta": self.pool_beta,
                "pool_reservoir_size": self.pool_reservoir_size,
                "pool_evict_policy": self.pool_evict_policy,
                "pool_oldest_floor": self.pool_oldest_floor,
                "active_state_mode": self.active_state_mode,
                "active_level_count": self.active_level_count,
                "prediction_level": self.prediction_level,
                "prediction_mode": self.prediction_mode,
                "index": self.index,
                "log_loss": self.log_loss,
                "segment_counter": self.segment_counter,
                "samples_since_close": self.samples_since_close,
            },
            "layers": [layer.state_dict() for layer in self.layers],
        }
        torch.save(payload, path)

    @classmethod
    def load_snapshot(cls, path: str | Path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        net_state = payload["network"]
        c = cls.__new__(cls)
        c.device = device
        c.input_dim = int(net_state["input_dim"])
        c.layer_sizes = list(net_state["layer_sizes"])
        c.num_halfspaces = int(net_state["num_halfspaces"])
        c.lr = float(net_state["lr"])
        c.pool_capacity = int(net_state["pool_capacity"])
        c.output_pool_capacity = int(
            net_state.get("output_pool_capacity", c.pool_capacity)
        )
        c.layer_pool_capacities = list(
            net_state.get(
                "layer_pool_capacities",
                [c.pool_capacity] * len(c.layer_sizes),
            )
        )
        if c.layer_pool_capacities:
            c.layer_pool_capacities[-1] = c.output_pool_capacity
        c.min_segment = int(net_state["min_segment"])
        c.ptw_depth = int(net_state.get("ptw_depth", 15))
        c.posterior_temp = float(net_state.get("posterior_temp", 1.0))
        c.pool_update_policy = str(net_state.get("pool_update_policy", "fifo"))
        c.pool_alpha = float(net_state.get("pool_alpha", math.inf))
        c.pool_beta = float(net_state.get("pool_beta", math.inf))
        c.pool_reservoir_size = int(net_state.get("pool_reservoir_size", 64))
        c.pool_evict_policy = str(net_state.get("pool_evict_policy", "fifo"))
        c.pool_oldest_floor = int(
            net_state.get("pool_oldest_floor", DEFAULT_POOL_OLDEST_FLOOR)
        )
        c.active_state_mode = str(net_state.get("active_state_mode", "flat"))
        c.active_level_count = int(
            net_state.get(
                "active_level_count",
                cls._active_level_count_for_min_segment(c.ptw_depth, c.min_segment),
            )
        )
        c.prediction_level = int(
            net_state.get("prediction_level", max(0, c.active_level_count - 1))
        )
        c.prediction_mode = str(net_state.get("prediction_mode", "selected_level"))
        if c.prediction_mode not in {"selected_level", "ptw_dp"}:
            raise ValueError(
                f"unknown prediction_mode in snapshot: {c.prediction_mode}"
            )
        if c.prediction_mode == "ptw_dp" and c.active_state_mode != "per_level":
            raise ValueError(
                "snapshot has prediction_mode='ptw_dp' but active_state_mode is not 'per_level'"
            )
        c.index = int(net_state["index"])
        c.log_loss = float(net_state["log_loss"])
        c.segment_counter = int(net_state["segment_counter"])
        c.samples_since_close = int(net_state.get("samples_since_close", 0))
        c.layers = [
            CudaFmnMixtureLayer.from_state_dict(state, device)
            for state in payload["layers"]
        ]
        return c
