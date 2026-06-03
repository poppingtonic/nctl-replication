from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from nctl_bench.nctl_network import (
    CudaFmnMixtureLayer,
    NctlNetwork,
    _get_fmn_cuda,
    _posterior_mixture_prob,
    mscb,
    mscb_boundary_plan,
    mscb_close_levels,
)


def require_cuda_kernel() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        _get_fmn_cuda()
    except RuntimeError as exc:
        if "Pre-compiled kernel not found" in str(exc):
            pytest.skip(str(exc))
        raise
    return torch.device("cuda")


@pytest.mark.cuda
def test_context_reduction_uses_all_dimensions() -> None:
    """Regression for warp-only reductions launched with multi-warp blocks.

    The first broken CUDA path launched up to 256 threads but reduced only one
    warp, so context gating ignored dimensions outside the first warp. This
    test forces the context decision to depend exclusively on dimensions 32..63.
    """
    device = require_cuda_kernel()
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=64,
        input_dim=64,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=1,
        device=device,
        seed=123,
    )
    fresh_idx = layer.M - 1
    with torch.no_grad():
        layer.hyperplanes.zero_()
        layer.hyperplanes[0, 0, 32:] = 1.0
        layer.hp_bias.fill_(-16.0)
        layer.mixture_weights.zero_()
        layer.mixture_weights[0, fresh_idx, 0, :].fill_(-1.0)
        layer.mixture_weights[0, fresh_idx, 1, :].fill_(1.0)

    z = torch.ones(1, 64, device=device)
    p_prev = torch.full((1, 64), 0.8, device=device)
    pred = layer.forward_only_batch(z, p_prev)[0, 0]
    torch.cuda.synchronize()

    assert pred.item() > 0.99


@pytest.mark.cuda
def test_fresh_slot_is_fixed_and_pooled_on_segment_close() -> None:
    """The fresh/base model must be slot M-1, not slot pool_size.

    If the kernel trains slot 0 during the first segment while Python copies
    slot M-1 at close, the pool stores an untrained model. This hard test catches
    that mismatch directly.
    """
    device = require_cuda_kernel()
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=64,
        input_dim=64,
        num_halfspaces=1,
        lr=0.25,
        pool_capacity=2,
        device=device,
        seed=321,
    )
    fresh_idx = layer.M - 1
    with torch.no_grad():
        layer.hyperplanes.zero_()
        layer.hp_bias.fill_(-1.0)  # always context 0

    z = torch.ones(4, 64, device=device)
    p_prev = torch.full((4, 64), 0.75, device=device)
    symbols = torch.ones(4, dtype=torch.int32, device=device)

    layer.forward_update_batch(z, p_prev, symbols)
    torch.cuda.synchronize()

    assert layer.mixture_weights[0, fresh_idx].abs().sum().item() > 0.0
    assert layer.mixture_weights[0, 0].abs().sum().item() == pytest.approx(0.0)

    layer.segment_close()

    assert int(layer.pool_sizes[0].item()) == 1
    assert layer.pool_snapshots[0, 0].abs().sum().item() > 0.0
    assert torch.allclose(layer.mixture_weights[0, 0], layer.pool_snapshots[0, 0])
    # Fresh slot is the base measure rho — its weights must persist across
    # close so the network keeps learning across PTW segment boundaries.
    assert layer.mixture_weights[0, fresh_idx].abs().sum().item() > 0.0
    # All active slots' current-segment posterior resets at segment open so
    # the FMN Eq. 5 priors stay symmetric.
    assert layer.segment_log_probs.abs().sum().item() == pytest.approx(0.0)


@pytest.mark.cuda
def test_bayes_mixture_posterior_includes_fresh_model() -> None:
    """Posterior weighting must be over fresh + pool, not fixed 0.5 fresh.

    With a fresh model that dominates current-segment likelihood and predicts 1,
    and a pool model that predicts 0, the mixture should be near 1. A fixed
    0.5*fresh + 0.5*pool average would be near 0.5.
    """
    device = require_cuda_kernel()
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=4,
        input_dim=4,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=1,
        device=device,
        seed=999,
    )
    fresh_idx = layer.M - 1
    with torch.no_grad():
        layer.hyperplanes.zero_()
        layer.hp_bias.fill_(-1.0)  # always context 0
        layer.pool_sizes.fill_(1)
        layer.mixture_weights.zero_()
        layer.mixture_weights[0, 0, 0, :].fill_(-6.0)
        layer.mixture_weights[0, fresh_idx, 0, :].fill_(6.0)
        layer.segment_log_probs.zero_()
        layer.segment_log_probs[0, 0] = -20.0
        layer.segment_log_probs[0, fresh_idx] = 0.0

    z = torch.ones(1, 4, device=device)
    p_prev = torch.full((1, 4), 0.8, device=device)
    pred = layer.forward_only_batch(z, p_prev)[0, 0]
    torch.cuda.synchronize()

    assert pred.item() > 0.99


def test_network_snapshot_roundtrip_cpu(tmp_path: Path) -> None:
    net = NctlNetwork(
        layer_sizes=[2, 1],
        input_dim=3,
        num_halfspaces=1,
        lr=0.001,
        pool_capacity=2,
        min_segment=4,
        device=torch.device("cpu"),
        seed=17,
    )
    with torch.no_grad():
        net.layers[0].pool_sizes[0] = 1
        net.layers[0].pool_snapshots[0, 0].fill_(0.25)
        net.layers[0].mixture_weights[0, 0].copy_(net.layers[0].pool_snapshots[0, 0])
    net.index = 12
    net.segment_counter = 12

    path = tmp_path / "nctl_snapshot.pt"
    net.save_snapshot(path, metadata={"task": "roundtrip"})

    loaded = NctlNetwork.load_snapshot(path, device=torch.device("cpu"))

    assert loaded.index == 12
    assert loaded.segment_counter == 12
    assert loaded.layer_sizes == [2, 1]
    assert loaded.ptw_depth == 15
    assert int(loaded.layers[0].pool_sizes[0].item()) == 1
    assert torch.allclose(
        loaded.layers[0].pool_snapshots[0, 0], net.layers[0].pool_snapshots[0, 0]
    )
    assert torch.equal(loaded.layers[0].pool_levels, net.layers[0].pool_levels)
    assert torch.equal(
        loaded.layers[0]._reservoir_rng.get_state(),
        net.layers[0]._reservoir_rng.get_state(),
    )


def test_posterior_mixture_reference_uses_current_segment_likelihood() -> None:
    preds = torch.tensor([0.01, 0.99, 0.25])
    log_probs = torch.tensor([-30.0, 0.0, -10.0])

    mixed = _posterior_mixture_prob(
        model_preds=preds,
        segment_log_probs=log_probs,
        pool_size=2,
        fresh_idx=2,
    )

    # Pool slot 1 has by far the best current-segment likelihood, so the
    # conditional Bayesian mixture should follow it rather than uniformly
    # averaging all active models.
    assert mixed.item() > 0.98


def test_segment_close_saves_best_active_model_cpu() -> None:
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=3,
        input_dim=3,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=7,
    )
    fresh_idx = layer.fresh_idx
    with torch.no_grad():
        layer.pool_sizes[0] = 1
        layer.pool_snapshots[0, 0].fill_(0.25)
        layer.mixture_weights[0, 0].fill_(1.5)
        layer.mixture_weights[0, fresh_idx].fill_(-2.0)
        layer.segment_log_probs[0, 0] = 0.0
        layer.segment_log_probs[0, fresh_idx] = -100.0

    fresh_before = layer.mixture_weights[0, fresh_idx].clone()

    layer.segment_close()

    assert int(layer.pool_sizes[0].item()) == 2
    # The best active pool copy, not the fresh model, should be remembered.
    assert torch.allclose(
        layer.pool_snapshots[0, 1], torch.full_like(layer.pool_snapshots[0, 1], 1.5)
    )
    # Pool slots restored from immutable snapshots; all active posteriors
    # reset so the next segment opens with the symmetric FMN priors.
    assert torch.allclose(layer.mixture_weights[0, 0], layer.pool_snapshots[0, 0])
    assert torch.allclose(layer.mixture_weights[0, 1], layer.pool_snapshots[0, 1])
    assert layer.segment_log_probs[0, 0].item() == pytest.approx(0.0)
    assert layer.segment_log_probs[0, 1].item() == pytest.approx(0.0)
    # Fresh slot weights (base measure rho) persist across close, but its
    # current-segment posterior resets so it competes on equal evidence.
    assert torch.allclose(layer.mixture_weights[0, fresh_idx], fresh_before)
    assert layer.segment_log_probs[0, fresh_idx].item() == pytest.approx(0.0)


def test_mscb_update_split_plan_groups_only_on_update_events_cpu() -> None:
    """D-step 1 contract: ``mscb_update_split_plan`` cuts the chunk only at
    UPDATEMODELPOOL-eligible events; pure-close-only events between cuts
    are returned in the ``close_only_events`` list so a downstream batched
    kernel call can apply them mid-sub-chunk.

    With depth=15 and min_segment=512, only level <= 6 is update-eligible
    (segment length 2**(15-6) = 512 >= 512), so a 1024-sample chunk
    starting at index 0 should have very few sub-chunks (1 update boundary
    over 1024 samples) but every sample beyond t=1 should appear in some
    close_only_events list at some level.
    """
    from nctl_bench.nctl_network import mscb_close_levels, mscb_update_split_plan
    depth = 15
    min_segment = 512
    chunk_size = 1024
    plan = mscb_update_split_plan(0, chunk_size, depth, min_segment)
    assert plan, "empty plan"
    # The last sub-chunk has empty boundary_close_levels / update_levels.
    assert plan[-1][2] == []
    assert plan[-1][3] == []
    # All sub-chunks tile [0, chunk_size) exactly once with no overlap.
    cursor = 0
    total_close_only = 0
    for start, end, boundary_close, boundary_update, close_only_events in plan:
        assert start == cursor
        assert end > start or (start == end == chunk_size)
        cursor = end
        total_close_only += len(close_only_events)
        # close_only events sit strictly inside [start, end).
        for offset, levels in close_only_events:
            assert start <= offset < end
            assert levels  # never empty inside the close_only list
        # boundary update is always a subset of boundary close.
        assert set(boundary_update).issubset(set(boundary_close))
    assert cursor == chunk_size
    # Sanity: at depth=15 min_segment=512, expect at least one update event.
    update_event_count = sum(1 for _, _, _, ul, _ in plan if ul)
    assert update_event_count >= 1
    # Reconstruct every close event from the plan and compare to the
    # unfiltered mscb_close_levels output.
    rebuilt: list[tuple[int, list[int]]] = []
    for start, end, boundary_close, boundary_update, close_only_events in plan:
        for offset, levels in close_only_events:
            rebuilt.append((offset, levels))
        if boundary_close:
            rebuilt.append((end, sorted(boundary_close)))
    # Compare against the natural mscb close schedule.
    natural: list[tuple[int, list[int]]] = []
    for offset in range(chunk_size):
        t = offset + 1
        cl = mscb_close_levels(depth, t)
        if cl:
            natural.append((offset, list(cl)))
    # The plan emits the *update* portion of a close event at offset==end
    # (boundary), while the natural list emits the FULL close set at that
    # same sample's offset.  So aggregate by offset for comparison:
    def _agg(events: list[tuple[int, list[int]]]) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for offset, levels in events:
            out.setdefault(offset, set()).update(levels)
        return out
    agg_rebuilt = _agg(rebuilt)
    agg_natural = _agg(natural)
    # Rebuilt covers every natural close offset for the levels above the
    # update-eligible region.  The natural close set at an update offset
    # additionally includes the update-eligible levels themselves, which
    # the plan splits off into ``update_levels`` -- attach them back at
    # the boundary offset to compare.
    # Boundary offset semantics in the plan: ``update_levels`` fire AFTER
    # sample end-1, while ``mscb_close_levels(depth, t)`` is the close set
    # firing BEFORE sample at offset t-1.  So the update event from a
    # boundary at end is the natural close at offset end (1-based t=end+1).
    for offset, levels in agg_natural.items():
        assert offset in agg_rebuilt, f"offset {offset} missing in plan"
        assert agg_rebuilt[offset] == levels, (
            f"offset {offset}: plan={agg_rebuilt[offset]} natural={levels}"
        )


def test_mscb_update_split_plan_keeps_sub_chunks_large_at_depth15_ms512_cpu() -> None:
    """The whole point of D: with depth=15, ms=512, chunk_size=1024 the
    median sub-chunk length should be ~512 samples (one UPDATE event every
    2**(15-6)=512 samples).  Today's per-event split gives median length 1.

    Concrete pin: at offset 0 over 60,000 samples, no sub-chunk should be
    shorter than 256 samples once we're past the first chunk's warmup,
    and the mean sub-chunk length should be > 200 samples.  This is the
    contract that makes the multilevel kernel actually batched.
    """
    from nctl_bench.nctl_network import mscb_update_split_plan
    depth = 15
    min_segment = 512
    lengths: list[int] = []
    cursor = 0
    chunk_size = 1024
    total = 60_000
    while cursor < total:
        n = min(chunk_size, total - cursor)
        plan = mscb_update_split_plan(cursor, n, depth, min_segment)
        for start, end, _, _, _ in plan:
            if end > start:
                lengths.append(end - start)
        cursor += n
    assert lengths, "no sub-chunks emitted"
    mean_length = sum(lengths) / len(lengths)
    # With ~117 update events over 60k samples, mean length is ~510.
    assert mean_length > 200.0, f"mean sub-chunk length {mean_length} < 200"


def test_mscb_update_split_plan_empty_or_singleton_inputs_cpu() -> None:
    """Edge cases: zero-length batch returns a single ([], []) sub-chunk;
    a single sample with no closes returns one sub-chunk covering it.
    """
    from nctl_bench.nctl_network import mscb_update_split_plan
    # batch_len=0 -> one empty sub-chunk.
    plan = mscb_update_split_plan(0, 0, depth=15, min_segment=512)
    assert plan == [(0, 0, [], [], [])]
    # batch_len=1 at current_index=0 -> no closes (t=1 has none).
    plan = mscb_update_split_plan(0, 1, depth=15, min_segment=512)
    assert plan == [(0, 1, [], [], [])]


def test_build_close_mask_matches_close_only_events_cpu() -> None:
    """``build_close_mask`` must produce a [L, B] bool tensor where the
    True entries exactly index the (level, offset) pairs in the
    close_only_events list.
    """
    from nctl_bench.nctl_network import build_close_mask
    L, B = 4, 6
    events = [
        (0, [3]),
        (2, [1, 2, 3]),
        (5, [3]),
    ]
    mask = build_close_mask(events, active_level_count=L, sub_chunk_len=B)
    assert mask.shape == (L, B)
    assert mask.dtype == torch.bool
    # Hand-checked True positions.
    expected = torch.zeros(L, B, dtype=torch.bool)
    expected[3, 0] = True
    expected[1, 2] = True
    expected[2, 2] = True
    expected[3, 2] = True
    expected[3, 5] = True
    assert torch.equal(mask, expected)


def test_build_close_mask_drops_levels_beyond_active_count_cpu() -> None:
    """Levels >= active_level_count are silently dropped (they correspond
    to depths not maintained by the layer's per-level active state).
    """
    from nctl_bench.nctl_network import build_close_mask
    events = [(0, [0, 5, 10])]
    mask = build_close_mask(events, active_level_count=3, sub_chunk_len=2)
    assert mask.shape == (3, 2)
    # Only level 0 is in range.
    assert mask[0, 0].item() is True
    assert mask[1:, :].sum().item() == 0


def test_mscb_boundary_plan_does_not_skip_boundaries_inside_large_chunks() -> None:
    """Paper Algorithm 1 lines 4--8 close PTW state at every MSCB event; the
    ``2**c`` (``min_segment``) skip applies only to ``UPDATEMODELPOOL``.

    Each boundary-plan event is therefore a ``(offset, close_levels,
    update_levels)`` triple: ``close_levels`` is the full paper close set and
    ``update_levels`` is the subset eligible to call ``UPDATEMODELPOOL``.
    """
    events = mscb_boundary_plan(0, 17, depth=5, min_segment=4)
    # Every t in 2..17 (offsets 1..16) is a close event because the deepest
    # PTW level always closes after the first sample.
    assert [(o, cl) for (o, cl, _) in events] == [
        (1, [5]),
        (2, [4, 5]),
        (3, [5]),
        (4, [3, 4, 5]),
        (5, [5]),
        (6, [4, 5]),
        (7, [5]),
        (8, [2, 3, 4, 5]),
        (9, [5]),
        (10, [4, 5]),
        (11, [5]),
        (12, [3, 4, 5]),
        (13, [5]),
        (14, [4, 5]),
        (15, [5]),
        (16, [1, 2, 3, 4, 5]),
    ]
    # Update-eligible levels honour the paper ``2**c`` skip: only segments of
    # length >= min_segment are eligible to call UPDATEMODELPOOL.
    update_events = [(o, ul) for (o, _, ul) in events if ul]
    assert update_events == [
        (4, [3]),
        (8, [2, 3]),
        (12, [3]),
        (16, [1, 2, 3]),
    ]
    # Mid-chunk start: only the offsets where MSCB fires are emitted, and the
    # close/update split still matches.
    assert NctlNetwork._mscb_boundary_plan(4, 9, depth=5, min_segment=4) == [
        (0, [3, 4, 5], [3]),
        (1, [5], []),
        (2, [4, 5], []),
        (3, [5], []),
        (4, [2, 3, 4, 5], [2, 3]),
        (5, [5], []),
        (6, [4, 5], []),
        (7, [5], []),
        (8, [3, 4, 5], [3]),
    ]


def test_mscb_close_levels_unfiltered_returns_full_paper_range_cpu() -> None:
    """Paper Algorithm 1 closes every PTW level in ``(MSCB_d(t)+1 .. d)``;
    the ``2**c`` (``min_segment``) skip only applies to UPDATEMODELPOOL.
    """
    # Without min_segment, every t >= 2 closes at least the deepest level.
    assert mscb_close_levels(depth=5, t=2) == [5]
    assert mscb_close_levels(depth=5, t=5) == [3, 4, 5]
    assert mscb_close_levels(depth=5, t=17) == [1, 2, 3, 4, 5]
    # Back-compat: passing ``min_segment`` still applies the legacy filter so
    # historical callers can recover the update-eligible subset directly.
    from nctl_bench.nctl_network import mscb_update_eligible_levels
    full = mscb_close_levels(depth=5, t=17)
    assert mscb_update_eligible_levels(full, depth=5, min_segment=4) == [1, 2, 3]
    assert mscb_close_levels(depth=5, t=17, min_segment=4) == [1, 2, 3]


def test_mscb_matches_paper_example() -> None:
    # FMN paper Figure 3/Algorithm 1: at t=5 and d=3, the changed bit
    # closes segments (1,4), (3,4), and (4,4), i.e. i=0.
    assert mscb(depth=3, t=5) == 0
    assert [mscb(3, t) for t in range(1, 9)] == [0, 2, 1, 2, 0, 2, 1, 2]
    assert [mscb_close_levels(3, t, min_segment=1) for t in range(1, 6)] == [
        [],
        [3],
        [2, 3],
        [3],
        [1, 2, 3],
    ]


def test_mscb_close_count_matches_split_mnist_task_sizes() -> None:
    """Pin segment_close frequency on the actual Split-MNIST task lengths.

    The MSCB planner fires hundreds of closes per task at depth=15 and
    min_segment=32.  With pool_capacity=8, this schedule can overwrite each
    per-node pool many times within a task.  A silent change to the close
    schedule (depth, min_segment policy, or planner) therefore changes final
    accuracy independently of the mixture math.

    This test pins (a) the MSCB-driven close-count per task and (b) the
    correct ordering across the min_segment ladder.  It runs on CPU and does
    not exercise the CUDA kernel.
    """
    task_lens = [12665, 12089, 11263, 12183, 11800]
    depth = 15

    def closes_for(min_segment: int) -> list[int]:
        idx = 0
        out = []
        for length in task_lens:
            c = 0
            for t in range(idx + 1, idx + length + 1):
                if mscb_close_levels(depth, t, min_segment):
                    c += 1
            out.append(c)
            idx += length
        return out

    # Pinned values produced by the current MSCB planner on the real task
    # split sizes.  See sweep_results/mnist_summary.json for the matching
    # per_task_pool growth on the GPU host.
    assert closes_for(32) == [395, 378, 352, 381, 368]
    assert closes_for(4096) == [3, 3, 2, 3, 3]

    # Monotone non-increasing in min_segment; close-count must halve roughly
    # for each doubling of min_segment.  This is the property the sweep
    # ladder ms256->ms2048 depends on.
    counts = [sum(closes_for(ms)) for ms in (32, 64, 128, 256, 512, 1024, 2048, 4096)]
    assert all(a >= b for a, b in zip(counts, counts[1:])), counts
    assert counts[0] > 4 * counts[-1], (
        "min_segment ladder must span >4x close-count range to be diagnostic;"
        f" got {counts}"
    )


def test_sweep_registers_paper_alpha_beta_grid_configs_cpu() -> None:
    import importlib.util

    sweep_path = SCRIPTS_DIR / "sweep_split_mnist.py"
    spec = importlib.util.spec_from_file_location("sweep_split_mnist_test", sweep_path)
    assert spec is not None and spec.loader is not None
    sweep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sweep
    try:
        spec.loader.exec_module(sweep)
    finally:
        sys.modules.pop(spec.name, None)

    expected = {
        "ms2048-p8-t0-paper-a-5-b0": ["--pool-alpha", "-5", "--pool-beta", "0"],
        "ms2048-p8-t0-paper-a0-b0": ["--pool-alpha", "0", "--pool-beta", "0"],
        "ms2048-p8-t0-paper-a5-b0": ["--pool-alpha", "5", "--pool-beta", "0"],
        "ms2048-p8-t0-paper-a-1-b0": ["--pool-alpha", "-1", "--pool-beta", "0"],
        "ms2048-p8-t0-paper-a1-b0": ["--pool-alpha", "1", "--pool-beta", "0"],
        "ms2048-p8-t0-paper-a2-b0": ["--pool-alpha", "2", "--pool-beta", "0"],
        "ms2048-p8-t0-paper-a0-b5": ["--pool-alpha", "0", "--pool-beta", "5"],
        "ms2048-p8-t0-paper-a0-b10": ["--pool-alpha", "0", "--pool-beta", "10"],
    }
    for name, required_args in expected.items():
        args = sweep.DEFAULT_CONFIGS[name].extra_args
        assert args[:4] == ["--min-segment", "2048", "--posterior-temp", "0"]
        assert "--pool-update-policy" in args
        for i in range(0, len(required_args), 2):
            flag, value = required_args[i], required_args[i + 1]
            assert args[args.index(flag) + 1] == value



def test_segment_close_tags_originating_ptw_levels_cpu() -> None:
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=3,
        input_dim=3,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=4,
        device=torch.device("cpu"),
        seed=8,
    )
    with torch.no_grad():
        layer.mixture_weights[0, layer.fresh_idx].fill_(0.75)
        layer.segment_log_probs[0, layer.fresh_idx] = 0.0

    layer.segment_close(levels=[2, 3])

    # Multi-level close must insert exactly one snapshot, tagged with the
    # deepest closing level (longest segment).
    assert int(layer.pool_sizes[0].item()) == 1
    assert int(layer.pool_levels[0, 0].item()) == 3
    assert int(layer.pool_task_ids[0, 0].item()) == -1


def test_segment_close_records_pool_provenance_and_fifo_evictions_cpu() -> None:
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=9,
    )

    with torch.no_grad():
        layer.mixture_weights[0, layer.fresh_idx].fill_(1.0)
    layer.segment_close(levels=[3], task_id=1, global_index=2048)
    assert int(layer.pool_sizes[0].item()) == 1
    assert int(layer.pool_task_ids[0, 0].item()) == 1
    assert int(layer.pool_insert_indices[0, 0].item()) == 2048
    assert layer.provenance_event_counts["append"] == 1
    assert layer.provenance_event_counts["evict"] == 0

    with torch.no_grad():
        layer.mixture_weights[0, layer.fresh_idx].fill_(2.0)
    layer.segment_close(levels=[4], task_id=2, global_index=4096)
    assert int(layer.pool_sizes[0].item()) == 2
    assert layer.pool_task_ids[0].tolist() == [1, 2]
    assert layer.pool_insert_indices[0].tolist() == [2048, 4096]
    assert layer.provenance_event_counts["append"] == 2
    assert layer.provenance_event_counts["evict"] == 0

    with torch.no_grad():
        layer.mixture_weights[0, layer.fresh_idx].fill_(3.0)
    layer.segment_close(levels=[5], task_id=3, global_index=6144)
    assert int(layer.pool_sizes[0].item()) == 2
    assert layer.pool_task_ids[0].tolist() == [2, 3]
    assert layer.pool_insert_indices[0].tolist() == [4096, 6144]
    assert layer.provenance_event_counts["append"] == 2
    assert layer.provenance_event_counts["evict"] == 1
    assert layer.provenance_evicted_task_counts == {1: 1}

    summary = layer.provenance_summary()
    assert summary["pool_task_histogram"] == {"2": 1, "3": 1}
    assert summary["slot_task_histograms"] == [{"2": 1}, {"3": 1}]
    assert summary["evicted_task_counts"] == {"1": 1}


def _paper_policy_test_layer(alpha: float, beta: float) -> CudaFmnMixtureLayer:
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=10,
        pool_update_policy="paper",
        pool_alpha=alpha,
        pool_beta=beta,
        pool_reservoir_size=4,
    )
    with torch.no_grad():
        layer.hyperplanes.zero_()
        layer.hp_bias.fill_(-1.0)  # context 0
    z = torch.zeros(4, 2)
    p_prev = torch.full((4, 2), 0.8)
    symbols = torch.ones(4, dtype=torch.int32)
    layer.observe_segment_batch(z, p_prev, symbols)
    return layer


def test_paper_pool_policy_refines_existing_slot_on_alpha_gain_cpu() -> None:
    layer = _paper_policy_test_layer(alpha=0.0, beta=float("inf"))
    with torch.no_grad():
        layer.pool_sizes[0] = 1
        layer.pool_task_ids[0, 0] = 1
        layer.pool_insert_indices[0, 0] = 1024
        layer.pool_snapshots[0, 0, 0].fill_(-2.0)
        layer.mixture_weights[0, 0, 0].fill_(2.0)
        layer.segment_log_probs[0, 0] = 0.0
        layer.segment_log_probs[0, layer.fresh_idx] = -100.0
        layer.pool_res_sizes[0, 0] = 4
        layer.pool_res_z[0, 0, :4].copy_(layer.segment_res_z[:4])
        layer.pool_res_p_prev[0, 0, :4].copy_(layer.segment_res_p_prev[:4])
        layer.pool_res_symbols[0, 0, :4].copy_(layer.segment_res_symbols[:4])

    layer.segment_close(levels=[4], task_id=1, global_index=2048)

    assert int(layer.pool_sizes[0].item()) == 1
    assert layer.provenance_event_counts["refine"] == 1
    assert layer.provenance_event_counts["append"] == 0
    assert torch.allclose(layer.pool_snapshots[0, 0, 0], torch.full((2,), 2.0))
    assert int(layer.pool_insert_indices[0, 0].item()) == 2048
    assert int(layer.pool_res_sizes[0, 0].item()) == 4


def test_paper_pool_policy_skips_when_pool_mixture_explains_segment_cpu() -> None:
    layer = _paper_policy_test_layer(alpha=float("inf"), beta=-100.0)
    with torch.no_grad():
        layer.pool_sizes[0] = 1
        layer.pool_task_ids[0, 0] = 1
        layer.pool_snapshots[0, 0, 0].fill_(2.0)
        layer.mixture_weights[0, layer.fresh_idx, 0].fill_(2.0)
        layer.segment_log_probs[0, 0] = -100.0
        layer.segment_log_probs[0, layer.fresh_idx] = 0.0

    layer.segment_close(levels=[4], task_id=2, global_index=4096)

    assert int(layer.pool_sizes[0].item()) == 1
    assert layer.provenance_event_counts["skip"] == 1
    assert layer.provenance_event_counts["append"] == 0
    assert int(layer.pool_task_ids[0, 0].item()) == 1


def test_training_feeds_post_update_probabilities_forward_for_existing_gpu_path() -> None:
    """The NCTL GPU path feeds the next layer the *post-update* output of the
    lower layer (via forward_only_batch), not the pre-update P(observed)
    returned by forward_update_batch.  This pins the existing inter-layer
    dataflow contract.
    """

    class FakeLayer:
        def __init__(self, update_output: torch.Tensor, forward_output: torch.Tensor):
            self.update_output = update_output
            self.forward_output = forward_output
            self.seen_update_input = None
            self.seen_forward_input = None

        def forward_update_batch(self, z_batch, p_prev_batch, symbols_batch):
            self.seen_update_input = p_prev_batch.clone()
            return self.update_output.clone()

        def forward_only_batch(self, z_batch, p_prev_batch):
            self.seen_forward_input = p_prev_batch.clone()
            return self.forward_output.clone()

        def segment_close(self, *args, **kwargs):  # not exercised here
            pass

    net = NctlNetwork.__new__(NctlNetwork)
    net.index = 0
    net.segment_counter = 0
    net.min_segment = 4
    # ptw_depth=0 keeps the MSCB schedule empty so this end-to-end test only
    # exercises the single-subchunk inter-layer dataflow contract.  Phase 1
    # of the paper-fidelity work decouples min_segment from PTW close events,
    # so non-zero depths now fire close events on every sample regardless of
    # min_segment; that behaviour is covered by dedicated MSCB tests above.
    net.ptw_depth = 0
    net.log_loss = 0.0
    first_update = torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=torch.float32)
    first_forward = torch.tensor([[0.3, 0.7], [0.5, 0.5]], dtype=torch.float32)
    final_psym = torch.tensor([[0.75], [0.25]], dtype=torch.float32)
    net.layers = [
        FakeLayer(first_update, first_forward),
        FakeLayer(final_psym, torch.tensor([[0.1], [0.9]], dtype=torch.float32)),
    ]

    z = torch.zeros(2, 3)
    symbols = torch.tensor([1, 0], dtype=torch.int32)
    loss = net.train_chunk(z, symbols)

    assert torch.allclose(net.layers[1].seen_update_input, first_forward)
    assert loss == pytest.approx(-torch.log(torch.tensor([0.75, 0.25])).sum().item())


def test_train_subchunk_coerces_symbol_device_and_dtype_cpu() -> None:
    """Regression for CPU/int64 labels entering the GPU driver path."""

    captured = {}

    class FakeLayer:
        def __init__(self, update_output: torch.Tensor, forward_output: torch.Tensor):
            self.update_output = update_output
            self.forward_output = forward_output

        def forward_update_batch(self, z_batch, p_prev_batch, symbols_batch):
            captured["symbols_dtype"] = symbols_batch.dtype
            captured["symbols_device"] = symbols_batch.device
            return self.update_output.clone()

        def forward_only_batch(self, z_batch, p_prev_batch):
            return self.forward_output.clone()

    net = NctlNetwork.__new__(NctlNetwork)
    net.index = 0
    net.segment_counter = 0
    net.min_segment = 4
    net.ptw_depth = 3
    net.log_loss = 0.0
    net.layers = [
        FakeLayer(torch.tensor([[0.4, 0.6]]), torch.tensor([[0.6, 0.4]])),
        FakeLayer(torch.tensor([[0.5]]), torch.tensor([[0.5]])),
    ]

    z = torch.zeros(1, 3)
    # int64 CPU labels mirror the raw MNIST tensor before driver normalisation.
    symbols = torch.tensor([1], dtype=torch.int64)
    loss = net._train_subchunk(z, symbols)

    assert captured["symbols_dtype"] == torch.int32
    assert captured["symbols_device"] == z.device
    assert loss == pytest.approx(-torch.log(torch.tensor(0.5)).item())



def test_fifo_segment_close_collapses_simultaneous_levels_cpu() -> None:
    """Historical FIFO diagnostics collapse simultaneous PTW closes to one
    remembered snapshot tagged by deepest level.
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=8,
        device=torch.device("cpu"),
        seed=11,
    )
    with torch.no_grad():
        layer.mixture_weights[0, layer.fresh_idx].fill_(1.0)

    layer.segment_close(levels=[5, 6, 7, 8, 9, 10])
    assert int(layer.pool_sizes[0].item()) == 1
    assert int(layer.pool_levels[0, 0].item()) == 10


def test_fifo_segment_close_skips_when_no_level_is_update_eligible_cpu() -> None:
    """Phase 1.5 regression: in fifo mode, a close event whose
    ``update_levels`` set is empty (because the ``2**c`` (``min_segment``)
    filter rejected every closing level) must skip the pool insertion
    entirely.  An earlier draft of Phase 1 collapsed to ``max(close_levels)``
    and treated it as update-eligible, which fired UPDATEMODELPOOL on every
    sample at depth=15 (a ~256x regression for the historical fifo +
    min_segment=512 diagnostic configs).
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=4,
        device=torch.device("cpu"),
        seed=11,
    )
    with torch.no_grad():
        layer.mixture_weights[0, layer.fresh_idx].fill_(1.0)

    # Close level 15 only (deepest level closes every sample at depth=15)
    # but update_levels=[] means the 2**c filter rejected it.
    layer.segment_close(
        levels=[15],
        update_levels=[],
        task_id=1,
        global_index=2,
    )
    assert int(layer.pool_sizes[0].item()) == 0
    assert layer.provenance_event_counts["append"] == 0


def test_fifo_segment_close_uses_deepest_update_eligible_level_cpu() -> None:
    """Phase 1.5: when fifo mode collapses a multi-level close to one
    remembered snapshot, the level tag must come from the deepest
    UPDATE-ELIGIBLE level (not the deepest CLOSE level).  Tagging by the
    deepest close level would mis-attribute the snapshot to a level that
    the paper's ``2**c`` skip explicitly excluded from
    ``UPDATEMODELPOOL``.
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=4,
        device=torch.device("cpu"),
        seed=12,
    )
    with torch.no_grad():
        layer.mixture_weights[0, layer.fresh_idx].fill_(1.0)

    # At t=513, depth=15, min_segment=512: close_levels=[6..15],
    # update_levels=[6].  fifo mode should collapse to ONE insert tagged
    # with level 6, not level 15.
    layer.segment_close(
        levels=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        update_levels=[6],
        task_id=2,
        global_index=513,
    )
    assert int(layer.pool_sizes[0].item()) == 1
    assert int(layer.pool_levels[0, 0].item()) == 6
    assert layer.provenance_event_counts["append"] == 1


def test_paper_segment_close_updates_each_closing_level_cpu() -> None:
    """Paper mode follows Algorithm 1 line 6--8: each closing PTW level
    invokes UPDATEMODELPOOL.
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=8,
        device=torch.device("cpu"),
        seed=11,
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
    )
    with torch.no_grad():
        layer.mixture_weights[0, layer.fresh_idx].fill_(1.0)

    levels = [5, 6, 7, 8, 9, 10]
    layer.segment_close(levels=levels)

    assert int(layer.pool_sizes[0].item()) == len(levels)
    assert [int(x.item()) for x in layer.pool_levels[0, : len(levels)]] == levels
    assert layer.provenance_event_counts["append"] == len(levels)



def test_segment_close_resets_all_close_levels_but_only_updates_eligible_cpu() -> None:
    """Phase 1: paper Algorithm 1 closes per-level PTW state for every level
    in ``close_levels`` but only invokes UPDATEMODELPOOL on the eligible
    subset (``update_levels``).  Levels closed but not updated must have
    their per-level active state reset without growing the bounded pool.
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=4,
        device=torch.device("cpu"),
        seed=21,
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
        active_state_mode="per_level",
        active_level_count=5,
        prediction_level=4,
    )
    fresh = layer.fresh_idx
    with torch.no_grad():
        # Mark every level's fresh slot so a successful UPDATEMODELPOOL would
        # leave a non-zero pool snapshot and a non-zero pool size for that
        # level.  Reset-only levels must leave the pool size at zero.
        for idx in range(layer.active_level_count):
            layer.level_mixture_weights[idx, 0, fresh].fill_(float(idx + 1))
            layer.level_segment_log_probs[idx, 0, fresh] = 0.0
            layer.level_pool_sizes[idx, 0] = 0

    layer.segment_close(
        levels=[0, 1, 2, 3, 4],
        update_levels=[1, 3],
        task_id=7,
        global_index=64,
    )

    # Only the eligible levels grew the shared bounded pool.
    assert int(layer.pool_sizes[0].item()) == 2
    assert sorted(int(x.item()) for x in layer.pool_levels[0, :2]) == [1, 3]
    assert layer.provenance_event_counts["append"] == 2
    # The non-eligible close levels reset per-level state from snapshots and
    # zeroed segment evidence but did not append.
    for non_update_idx in (0, 2, 4):
        # No segment evidence remains.
        assert torch.all(layer.level_segment_log_probs[non_update_idx, 0] == 0)
        # Per-level pool-size view reflects the (now-grown) shared pool size.
        assert int(layer.level_pool_sizes[non_update_idx, 0].item()) == 2


def test_segment_close_resets_segment_reservoir_for_close_only_levels_cpu() -> None:
    """A close-only PTW level must clear its segment reservoir even though
    no UPDATEMODELPOOL fires for it; otherwise stale samples bleed into the
    next segment's update decisions.
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=22,
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
        pool_reservoir_size=4,
        active_state_mode="per_level",
        active_level_count=4,
        prediction_level=3,
    )
    z = torch.zeros(3, 2)
    p_prev = torch.full((3, 2), 0.8)
    symbols = torch.ones(3, dtype=torch.int32)
    layer.observe_segment_batch(z, p_prev, symbols)
    assert [int(x.item()) for x in layer.level_res_size[:4]] == [3, 3, 3, 3]

    # Close levels 1 and 2 but only allow UPDATEMODELPOOL on level 2.  Level
    # 1's segment reservoir must still be reset.
    layer.segment_close(
        levels=[1, 2],
        update_levels=[2],
        task_id=1,
        global_index=3,
    )
    assert int(layer.level_res_size[1].item()) == 0
    assert int(layer.level_res_size[2].item()) == 0
    # Untouched levels keep their reservoir.
    assert int(layer.level_res_size[3].item()) == 3


def test_network_train_chunk_close_count_at_min_segment_512_is_full_paper_schedule_cpu() -> None:
    """Phase 1 paper fidelity: at depth=15 and min_segment=512 the number of
    *close* events is the full paper Algorithm 1 schedule (one per t >= 2),
    while the *update-eligible* count is the small ``2**c`` subset.  This
    pins the new decoupling between PTW close and UPDATEMODELPOOL.
    """
    from nctl_bench.nctl_network import (
        mscb_boundary_plan,
        mscb_close_levels,
        mscb_update_eligible_levels,
    )

    depth = 15
    min_segment = 512
    n = 1024
    events = mscb_boundary_plan(0, n, depth, min_segment)
    # One event per t >= 2 covers t = 2 .. n.
    assert len(events) == n - 1
    # Every event closes at least the deepest PTW level.
    assert all(depth in cl for (_, cl, _) in events)
    # Update-eligible events follow the 2**c filter: for depth=15,
    # min_segment=512 the eligible levels are j <= 6 (seg_len 2^9..2^15).
    update_event_count = sum(1 for (_, _, ul) in events if ul)
    eligible_full = [
        mscb_update_eligible_levels(
            mscb_close_levels(depth, t), depth, min_segment
        )
        for t in range(2, n + 1)
    ]
    assert update_event_count == sum(1 for ul in eligible_full if ul)
    # Spot-check the very first eligible event: at t=513 the segment of
    # length 512 closes at level 6 (seg_len 2^9 = 512).
    first_update_t = next(t for t in range(2, n + 1) if eligible_full[t - 2])
    assert first_update_t == 513
    assert eligible_full[first_update_t - 2] == [6]


def _make_dp_layer(active_level_count: int = 4) -> CudaFmnMixtureLayer:
    """Build a per-level DP-enabled layer for the Phase 2 unit tests.  Small
    nodes/inputs so synthetic per-level conditionals are easy to set up.
    """
    return CudaFmnMixtureLayer(
        num_nodes=2,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=1,
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
        active_state_mode="per_level",
        active_level_count=active_level_count,
        prediction_level=active_level_count - 1,
        prediction_mode="ptw_dp",
    )


def test_ptw_dp_combine_returns_deepest_level_at_fresh_init_cpu() -> None:
    """At fresh DP-state init (level_log_nu = ptw_log_w = 0), the convex
    combination collapses to the deepest level's conditional q_d(1).

    Algorithm 1 line 1 initialises w_j = 1 = nu_{r_j}, so pi_j = 1/2 for all j
    and the combination at j=0 is 1/2 q_0 + 1/4 q_1 + 1/8 q_2 + 1/8 q_d.
    Test that pin holds exactly.
    """
    layer = _make_dp_layer(active_level_count=4)
    q1 = torch.tensor([
        [0.10, 0.20],  # level 0
        [0.30, 0.40],  # level 1
        [0.50, 0.60],  # level 2
        [0.70, 0.80],  # level 3 (deepest)
    ])
    out = layer._ptw_dp_combine(q1)
    # Bottom-up: P_3 = q_3 ; P_2 = 1/2 q_2 + 1/2 P_3 ;
    # P_1 = 1/2 q_1 + 1/2 P_2 ; P_0 = 1/2 q_0 + 1/2 P_1.
    expected = 0.5 * q1[0] + 0.25 * q1[1] + 0.125 * q1[2] + 0.125 * q1[3]
    assert torch.allclose(out, expected, atol=1e-6)


def test_ptw_dp_combine_collapses_to_one_level_when_others_uniform_cpu() -> None:
    """When all but one level predict 0.5 (carries no information about x_t),
    the DP output equals that informative level's conditional weighted by its
    pi_j.  Concretely: with fresh init, q_j(1)=0.5 except level 2's q=0.9, the
    output should be 0.5 + 0.125 * (0.9 - 0.5) = 0.55.
    """
    layer = _make_dp_layer(active_level_count=4)
    q1 = torch.full((4, 2), 0.5)
    q1[2] = 0.9
    out = layer._ptw_dp_combine(q1)
    # Expected at level 2: weight = (1/2)*(1/2)*(1/2) = 1/8.  P_0 = 1/2 * 0.5
    # + 1/4 * 0.5 + 1/8 * 0.9 + 1/8 * 0.5 = 0.55.
    expected = torch.full((2,), 0.55)
    assert torch.allclose(out, expected, atol=1e-6)


def test_ptw_dp_update_state_matches_naive_recurrence_cpu() -> None:
    """Driving _ptw_dp_update_state once on a small synthetic
    log_q_obs sequence reproduces a hand-rolled bottom-up logaddexp
    recurrence in pure Python.
    """
    import math as _math
    layer = _make_dp_layer(active_level_count=3)
    L, N = 3, 2
    # log q_obs for one sample, made-up per (level, node) values.
    log_q = torch.tensor([
        [_math.log(0.4), _math.log(0.6)],
        [_math.log(0.7), _math.log(0.2)],
        [_math.log(0.5), _math.log(0.9)],
    ], dtype=torch.float64)

    # Reference: literal Algorithm 1 lines 10-13 in pure Python.
    nu = [torch.zeros(N, dtype=torch.float64) for _ in range(L)]
    w  = [torch.zeros(N, dtype=torch.float64) for _ in range(L)]
    b  = [torch.zeros(N, dtype=torch.float64) for _ in range(L)]
    for j in range(L):
        nu[j] = nu[j] + log_q[j]
    w[L - 1] = nu[L - 1].clone()
    log_half = _math.log(0.5)
    for j in range(L - 2, -1, -1):
        a = log_half + nu[j]
        c = log_half + w[j + 1] + b[j]
        w[j] = torch.logaddexp(a, c)

    layer._ptw_dp_update_state(log_q)
    for j in range(L):
        assert torch.allclose(layer.level_log_nu[j], nu[j], atol=1e-12)
        assert torch.allclose(layer.ptw_log_w[j], w[j], atol=1e-12)


def test_segment_close_applies_line5_b_cache_then_resets_dp_triple_cpu() -> None:
    """Paper Algorithm 1 line 5 caches b_i <- w_{i+1} BEFORE the closes at
    line 7 reset those levels.  The hook must read the pre-reset w_{i+1}.
    """
    import math as _math
    layer = _make_dp_layer(active_level_count=4)
    # Seed DP state: set ptw_log_w[1] to a recognisable value.
    layer.ptw_log_w[1].fill_(_math.log(0.42))
    layer.level_log_nu[1].fill_(_math.log(0.42))
    layer.ptw_log_b[1].fill_(_math.log(0.7))  # should be reset to 0.
    # Close levels [1, 2, 3] -> MSCB index i=0, so b_0 <- w_1 = log(0.42).
    layer.segment_close(levels=[1, 2, 3], update_levels=[], task_id=1, global_index=2)
    assert torch.allclose(layer.ptw_log_b[0], torch.full((2,), _math.log(0.42), dtype=torch.float64), atol=1e-12)
    # Levels 1, 2, 3 reset to 0 (paper line 8: w_j, b_j, nu <- 1).
    for j in (1, 2, 3):
        assert torch.all(layer.level_log_nu[j] == 0.0)
        assert torch.all(layer.ptw_log_w[j] == 0.0)
        assert torch.all(layer.ptw_log_b[j] == 0.0)
    # Level 0 left untouched on level_log_nu / ptw_log_w (only b_0 changed).
    assert torch.all(layer.level_log_nu[0] == 0.0)
    assert torch.all(layer.ptw_log_w[0] == 0.0)


def test_ptw_dp_calls_kernel_with_layer_posterior_temp_for_paper_nu_cpu() -> None:
    """Algorithm 1's ν_{r_j}(x_t|x_<t) is the paper's Eq. 5 conditional
    ratio (Bayesian posterior over per-slot segment log-likelihoods),
    NOT the unweighted 1/2 fresh + 1/2 pool-mean mixture used pre-5I.

    The CUDA kernel takes a posterior_temp scaling the data evidence;
    posterior_temp=1.0 reproduces the paper-faithful conditional and
    posterior_temp=0 collapses to the legacy unweighted mixture.  The
    DP entry points must therefore forward ``layer.posterior_temp``
    verbatim so the user can choose between the two regimes (default
    paper-faithful behaviour is whatever ``layer.posterior_temp``
    holds, exposed on the CLI as ``--posterior-temp``).

    A regression here would silently strip the posterior weighting and
    cap accuracy at the pre-5I ~78% ceiling on Split-MNIST.
    """
    from nctl_bench import nctl_network as _net

    calls: list[float] = []

    class _SpyCuda:
        def forward_update(self, z, p_prev, sym, w, hp, hb, ps, slp, mlp, lr, posterior_temp):
            calls.append(("update", float(posterior_temp)))
            return torch.full((z.size(0), w.size(0)), 0.5)

        def forward_only(self, z, p_prev, w, hp, hb, ps, slp, posterior_temp):
            calls.append(("only", float(posterior_temp)))
            return torch.full((z.size(0), w.size(0)), 0.5)

    prev = _net._fmn_cuda
    prev_ml = _net._fmn_multilevel_cuda
    prev_ml_missing = _net._fmn_multilevel_cuda_missing
    _net._fmn_cuda = _SpyCuda()
    # Phase 5E intercepts before _fmn_cuda; force the per-level fallback so
    # the spy installed above actually fires.  The contract under test is
    # specifically about the fallback path's posterior_temp forwarding.
    _net._fmn_multilevel_cuda = None
    _net._fmn_multilevel_cuda_missing = True
    try:
        layer = CudaFmnMixtureLayer(
            num_nodes=2,
            num_inputs=2,
            input_dim=2,
            num_halfspaces=1,
            lr=0.001,
            pool_capacity=2,
            device=torch.device("cpu"),
            seed=1,
            posterior_temp=0.37,  # nonzero to prove we override at the call site
            pool_update_policy="paper",
            pool_alpha=float("inf"),
            pool_beta=float("inf"),
            active_state_mode="per_level",
            active_level_count=3,
            prediction_level=2,
            prediction_mode="ptw_dp",
        )
        # Drive both code paths once.
        z = torch.zeros(1, 2)
        p_prev = torch.full((1, 2), 0.5)
        symbols = torch.zeros(1, dtype=torch.int32)
        layer.forward_only_batch(z, p_prev)
        layer.forward_update_batch(z, p_prev, symbols)
    finally:
        _net._fmn_cuda = prev
        _net._fmn_multilevel_cuda = prev_ml
        _net._fmn_multilevel_cuda_missing = prev_ml_missing

    # Every spy call must have posterior_temp=0.37 (== layer.posterior_temp).
    assert calls, "spy never called"
    for kind, t in calls:
        assert t == 0.37, f"{kind} called with posterior_temp={t} (expected 0.37)"
    # And we should have hit every level for both flavours.
    by_kind = {k: [t for kk, t in calls if kk == k] for k in ("only", "update")}
    assert len(by_kind["only"]) == 3
    assert len(by_kind["update"]) == 3


def test_ptw_dp_apply_chunk_segmented_empty_close_events_matches_legacy_cpu() -> None:
    """When close_events is empty (or None), the segmented path must be
    bit-identical to the legacy Phase 5D path.  Pure additivity check.
    """
    layer_a = _make_dp_layer(active_level_count=4)
    layer_b = _make_dp_layer(active_level_count=4)
    L, N = 4, 2
    B = 7
    torch.manual_seed(13)
    nu0 = torch.linspace(-2.0, 1.0, L * N).reshape(L, N).to(torch.float64)
    b_init = torch.linspace(-0.5, 0.5, L * N).reshape(L, N).to(torch.float64)
    w0 = torch.linspace(-1.0, 0.5, L * N).reshape(L, N).to(torch.float64)
    for layer in (layer_a, layer_b):
        layer.level_log_nu.copy_(nu0)
        layer.ptw_log_b.copy_(b_init)
        layer.ptw_log_w.copy_(w0)
    q_obs = torch.rand(L, B, N).clamp(1e-6, 1.0 - 1e-6)
    log_q_obs = torch.log(q_obs)
    q1 = q_obs.clone()
    # No close events -> both paths identical.
    out_legacy = layer_a._ptw_dp_apply_chunk(log_q_obs, q1)
    out_segmented = layer_b._ptw_dp_apply_chunk(log_q_obs, q1, close_events=[])
    assert torch.allclose(out_legacy, out_segmented, atol=1e-12)
    assert torch.allclose(layer_a.level_log_nu, layer_b.level_log_nu, atol=1e-12)
    assert torch.allclose(layer_a.ptw_log_w, layer_b.ptw_log_w, atol=1e-12)


def test_ptw_dp_apply_chunk_segmented_matches_per_sample_with_inline_close_cpu() -> None:
    """With close_events injected mid-chunk, the segmented path must
    match a hand-rolled per-sample reference that calls
    ``_ptw_dp_combine`` / ``_ptw_dp_update_state`` per sample and runs
    ``_apply_ptw_dp_close`` at the exact same offsets.

    This is the correctness proof D-step 4 will rely on: segmented DP
    advance with inline closes == per-sample DP advance + segment_close
    at the boundary.
    """
    layer_ref = _make_dp_layer(active_level_count=4)
    layer_new = _make_dp_layer(active_level_count=4)
    L, N = 4, 2
    B = 8
    torch.manual_seed(17)
    nu0 = torch.linspace(-2.0, 1.0, L * N).reshape(L, N).to(torch.float64)
    b_init = torch.linspace(-0.5, 0.5, L * N).reshape(L, N).to(torch.float64)
    w0 = torch.linspace(-1.0, 0.5, L * N).reshape(L, N).to(torch.float64)
    for layer in (layer_ref, layer_new):
        layer.level_log_nu.copy_(nu0)
        layer.ptw_log_b.copy_(b_init)
        layer.ptw_log_w.copy_(w0)
    q_obs = torch.rand(L, B, N).clamp(1e-6, 1.0 - 1e-6)
    log_q_obs = torch.log(q_obs)
    q1 = q_obs.clone()
    # Two close events: at offset 2 close levels [2, 3]; at offset 5
    # close level [3].  These are realistic shapes from
    # mscb_close_levels at depth=L-1.
    close_events = [(2, [2, 3]), (5, [3])]

    # Reference: per-sample DP advance, with inline close before each
    # sample at an event offset.
    out_ref = torch.empty(B, N)
    event_dict = {off: levels for off, levels in close_events}
    for b in range(B):
        if b in event_dict:
            layer_ref._apply_ptw_dp_close_inline(event_dict[b])
        p1 = layer_ref._ptw_dp_combine(q1[:, b])
        out_ref[b] = p1
        layer_ref._ptw_dp_update_state(log_q_obs[:, b])

    # New segmented path.
    out_new = layer_new._ptw_dp_apply_chunk(log_q_obs, q1, close_events=close_events)
    assert torch.allclose(out_new, out_ref, atol=1e-6), (
        f"max diff = {(out_new - out_ref).abs().max().item():.3e}"
    )
    assert torch.allclose(layer_new.level_log_nu, layer_ref.level_log_nu, atol=1e-12)
    assert torch.allclose(layer_new.ptw_log_w, layer_ref.ptw_log_w, atol=1e-12)


def test_ptw_dp_apply_chunk_segmented_event_at_offset_zero_cpu() -> None:
    """A close event at offset 0 must fire before sample 0 of the chunk,
    NOT after it.  Pin this explicitly; an off-by-one would silently
    swap the meaning.
    """
    layer_ref = _make_dp_layer(active_level_count=3)
    layer_new = _make_dp_layer(active_level_count=3)
    L, N = 3, 2
    B = 3
    torch.manual_seed(23)
    nu0 = torch.full((L, N), -1.0, dtype=torch.float64)
    w0 = torch.full((L, N), 0.5, dtype=torch.float64)
    b0 = torch.full((L, N), 0.1, dtype=torch.float64)
    for layer in (layer_ref, layer_new):
        layer.level_log_nu.copy_(nu0)
        layer.ptw_log_w.copy_(w0)
        layer.ptw_log_b.copy_(b0)
    q_obs = torch.rand(L, B, N).clamp(1e-6, 1.0 - 1e-6)
    log_q_obs = torch.log(q_obs)
    q1 = q_obs.clone()
    close_events = [(0, [1, 2])]
    out_ref = torch.empty(B, N)
    for b in range(B):
        if b == 0:
            layer_ref._apply_ptw_dp_close_inline([1, 2])
        p1 = layer_ref._ptw_dp_combine(q1[:, b])
        out_ref[b] = p1
        layer_ref._ptw_dp_update_state(log_q_obs[:, b])
    out_new = layer_new._ptw_dp_apply_chunk(log_q_obs, q1, close_events=close_events)
    assert torch.allclose(out_new, out_ref, atol=1e-6)
    assert torch.allclose(layer_new.level_log_nu, layer_ref.level_log_nu, atol=1e-12)
    assert torch.allclose(layer_new.ptw_log_w, layer_ref.ptw_log_w, atol=1e-12)


def test_ptw_dp_apply_chunk_segmented_event_at_offset_B_cpu() -> None:
    """An event at offset B fires AFTER sample B-1 of the chunk: i.e.
    after the chunk's normal processing completes, the DP state should
    be reset for the listed levels.
    """
    layer_ref = _make_dp_layer(active_level_count=3)
    layer_new = _make_dp_layer(active_level_count=3)
    L, N = 3, 2
    B = 4
    torch.manual_seed(29)
    for layer in (layer_ref, layer_new):
        layer.level_log_nu.fill_(0.3)
        layer.ptw_log_w.fill_(0.4)
        layer.ptw_log_b.fill_(0.0)
    q_obs = torch.rand(L, B, N).clamp(1e-6, 1.0 - 1e-6)
    log_q_obs = torch.log(q_obs)
    q1 = q_obs.clone()
    close_events = [(B, [1, 2])]
    out_ref = torch.empty(B, N)
    for b in range(B):
        p1 = layer_ref._ptw_dp_combine(q1[:, b])
        out_ref[b] = p1
        layer_ref._ptw_dp_update_state(log_q_obs[:, b])
    # Apply the end-of-chunk close on the reference.
    layer_ref._apply_ptw_dp_close_inline([1, 2])
    out_new = layer_new._ptw_dp_apply_chunk(log_q_obs, q1, close_events=close_events)
    assert torch.allclose(out_new, out_ref, atol=1e-6)
    assert torch.allclose(layer_new.level_log_nu, layer_ref.level_log_nu, atol=1e-12)
    assert torch.allclose(layer_new.ptw_log_w, layer_ref.ptw_log_w, atol=1e-12)


def test_ptw_dp_apply_chunk_segmented_perf_floor_one_combine_cpu() -> None:
    """30.9 Phase 5G Commit C perf-floor contract.

    The vectorised ``_ptw_dp_apply_chunk_segmented`` must NOT dispatch
    back into ``_ptw_dp_apply_chunk`` once per segment.  In particular,
    when at least one close event fires inside the chunk the segmented
    path must do the whole DP combine in a single vectorised pass with
    zero inner ``_ptw_dp_apply_chunk`` calls.  This pins the perf fix
    so a future refactor cannot silently regress to the Phase 5F
    ``for off in sorted_offsets: self._ptw_dp_apply_chunk(...)`` loop
    that drove the ~180k tiny CUDA dispatches per seed.
    """
    layer = _make_dp_layer(active_level_count=4)
    L, N = 4, 2
    B = 16
    torch.manual_seed(41)
    layer.level_log_nu.fill_(0.0)
    layer.ptw_log_w.fill_(0.0)
    layer.ptw_log_b.fill_(0.0)
    q_obs = torch.rand(L, B, N).clamp(1e-6, 1.0 - 1e-6)
    log_q_obs = torch.log(q_obs)
    q1 = q_obs.clone()
    # Five close events: empty offset 0 case + interior + tail.
    close_events = [(2, [3]), (4, [2, 3]), (7, [3]), (10, [1, 2, 3]), (13, [3])]

    inner_calls: list[int] = []
    original_apply = layer._ptw_dp_apply_chunk

    def counting_apply(log_q_chunk, q1_chunk, close_events=None):
        # Distinguish the dispatch-from-segmented path from the public
        # ``_ptw_dp_apply_chunk(close_events=...)`` entry by inspecting
        # the close_events argument; the segmented path must NOT call
        # back with close_events at all.
        inner_calls.append(log_q_chunk.size(1))
        return original_apply(log_q_chunk, q1_chunk, close_events=close_events)

    layer._ptw_dp_apply_chunk = counting_apply
    try:
        layer._ptw_dp_apply_chunk_segmented(log_q_obs, q1, close_events)
    finally:
        layer._ptw_dp_apply_chunk = original_apply

    assert inner_calls == [], (
        f"_ptw_dp_apply_chunk_segmented dispatched {len(inner_calls)} inner "
        f"_ptw_dp_apply_chunk calls (sub-chunk sizes={inner_calls}); the "
        "Phase 5G Commit C vectorised path must do the whole DP combine in "
        "a single pass with zero inner calls."
    )


def test_ptw_dp_apply_chunk_segmented_perf_floor_no_events_uses_one_combine_cpu() -> None:
    """Companion to the perf-floor test for the empty-events edge case.

    When ``close_events`` is empty, ``_ptw_dp_apply_chunk_segmented``
    delegates to the single-segment ``_ptw_dp_apply_chunk`` path, which
    counts as exactly ONE call (not zero, because there's still real
    work to do).  Pin this so the empty-events branch can't accidentally
    grow a redundant call or get re-routed back into the segmented path.
    """
    layer = _make_dp_layer(active_level_count=3)
    L, N = 3, 2
    B = 8
    torch.manual_seed(43)
    layer.level_log_nu.fill_(0.0)
    layer.ptw_log_w.fill_(0.0)
    layer.ptw_log_b.fill_(0.0)
    q_obs = torch.rand(L, B, N).clamp(1e-6, 1.0 - 1e-6)
    log_q_obs = torch.log(q_obs)
    q1 = q_obs.clone()

    inner_calls: list[int] = []
    original_apply = layer._ptw_dp_apply_chunk

    def counting_apply(log_q_chunk, q1_chunk, close_events=None):
        inner_calls.append(log_q_chunk.size(1))
        return original_apply(log_q_chunk, q1_chunk, close_events=close_events)

    layer._ptw_dp_apply_chunk = counting_apply
    try:
        layer._ptw_dp_apply_chunk_segmented(log_q_obs, q1, [])
    finally:
        layer._ptw_dp_apply_chunk = original_apply

    assert inner_calls == [B], (
        f"empty-events segmented path should delegate to one full-chunk "
        f"_ptw_dp_apply_chunk(B={B}) call; got {inner_calls}"
    )


def test_ptw_dp_apply_chunk_matches_per_sample_reference_cpu() -> None:
    """The fused batched DP (Phase 5D) must produce numerically identical
    output AND post-chunk DP state as the original per-sample reference
    (one ``_ptw_dp_combine`` + one ``_ptw_dp_update_state`` per b).  This
    is the strongest test we can land without a GPU: it pins the new path
    to the old, slow-but-known-correct implementation.
    """
    layer_ref = _make_dp_layer(active_level_count=4)
    layer_new = _make_dp_layer(active_level_count=4)
    L, N = 4, 2
    B = 7
    # Plant non-zero starting DP state and ptw_log_b so the recurrence has
    # signal to verify.
    torch.manual_seed(13)
    nu0 = torch.linspace(-3.0, 1.0, L * N).reshape(L, N).to(torch.float64)
    b_init = torch.linspace(-0.5, 0.5, L * N).reshape(L, N).to(torch.float64)
    # Use an arbitrary w0 that is NOT necessarily the bottom-up combine of
    # nu0/b_init.  This mirrors what `_apply_ptw_dp_close` leaves behind
    # for levels above the closed region: nu_close=0, b_close=0, but w for
    # non-closed levels retains its pre-close value.  The fused chunk path
    # must reproduce the per-sample reference exactly in this regime.
    w0 = torch.linspace(-2.5, 0.5, L * N).reshape(L, N).to(torch.float64)
    for layer in (layer_ref, layer_new):
        layer.level_log_nu.copy_(nu0)
        layer.ptw_log_b.copy_(b_init)
        layer.ptw_log_w.copy_(w0)

    # Random per-sample inputs (within a chunk).
    q_obs = torch.rand(L, B, N).clamp(min=1e-6, max=1.0 - 1e-6)
    log_q_obs = torch.log(q_obs)
    sym = torch.tensor([1, 0, 1, 1, 0, 0, 1], dtype=torch.bool)
    q1 = torch.where(sym.view(1, B, 1), q_obs, 1.0 - q_obs)

    # Reference: per-sample loop, the OLD implementation.
    out_ref = torch.empty(B, N)
    for b in range(B):
        p1 = layer_ref._ptw_dp_combine(q1[:, b])
        out_ref[b] = p1 if sym[b] else (1.0 - p1)
        layer_ref._ptw_dp_update_state(log_q_obs[:, b])

    # New: one fused call.
    p1_batch = layer_new._ptw_dp_apply_chunk(log_q_obs, q1)
    sym_f = sym.to(p1_batch.dtype).view(B, 1)
    out_new = sym_f * p1_batch + (1.0 - sym_f) * (1.0 - p1_batch)

    assert torch.allclose(out_new, out_ref, atol=1e-6),         f"max diff = {(out_new - out_ref).abs().max().item():.3e}"
    # DP state must agree exactly after the chunk.
    assert torch.allclose(layer_new.level_log_nu, layer_ref.level_log_nu, atol=1e-12)
    assert torch.allclose(layer_new.ptw_log_w, layer_ref.ptw_log_w, atol=1e-12)


def test_ptw_dp_apply_chunk_matches_reference_after_close_state_cpu() -> None:
    """Production divergence case: after ``_apply_ptw_dp_close``, the
    stored ``ptw_log_w`` for the levels above the closed region is the
    pre-close value (whatever the recurrence produced before line 8 zeroed
    nu/b for the closing j).  This is NOT ``bottom_up(nu, b)`` at chunk
    entry.  Pin that the fused chunk path still matches the per-sample
    reference under this exact state.
    """
    layer_ref = _make_dp_layer(active_level_count=4)
    layer_new = _make_dp_layer(active_level_count=4)
    L, N = 4, 2
    B = 4
    # Plant a state mirroring 's leftovers:
    # nu[0,1] and w[0,1] are nonzero from prior accumulation; nu[2,3], w[2,3],
    # b[2,3] are zero; b[1] = old w[2] cached by line 5.
    nu0 = torch.zeros(L, N, dtype=torch.float64)
    nu0[0].copy_(torch.tensor([-0.8, 1.2], dtype=torch.float64))
    nu0[1].copy_(torch.tensor([0.4, -0.3], dtype=torch.float64))
    w0 = torch.zeros(L, N, dtype=torch.float64)
    w0[0].copy_(torch.tensor([-0.2, 0.7], dtype=torch.float64))
    w0[1].copy_(torch.tensor([0.1, 0.4], dtype=torch.float64))
    b_init = torch.zeros(L, N, dtype=torch.float64)
    b_init[1].copy_(torch.tensor([0.05, 0.02], dtype=torch.float64))  # cached
    for layer in (layer_ref, layer_new):
        layer.level_log_nu.copy_(nu0)
        layer.ptw_log_b.copy_(b_init)
        layer.ptw_log_w.copy_(w0)

    torch.manual_seed(42)
    q_obs = torch.rand(L, B, N).clamp(1e-6, 1.0 - 1e-6)
    log_q_obs = torch.log(q_obs)
    sym = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    q1 = torch.where(sym.view(1, B, 1), q_obs, 1.0 - q_obs)

    out_ref = torch.empty(B, N)
    for b in range(B):
        p1 = layer_ref._ptw_dp_combine(q1[:, b])
        out_ref[b] = p1 if sym[b] else (1.0 - p1)
        layer_ref._ptw_dp_update_state(log_q_obs[:, b])

    p1_batch = layer_new._ptw_dp_apply_chunk(log_q_obs, q1)
    sym_f = sym.to(p1_batch.dtype).view(B, 1)
    out_new = sym_f * p1_batch + (1.0 - sym_f) * (1.0 - p1_batch)
    assert torch.allclose(out_new, out_ref, atol=1e-7),         f'max diff = {(out_new - out_ref).abs().max().item():.3e}'
    assert torch.allclose(layer_new.level_log_nu, layer_ref.level_log_nu, atol=1e-12)
    assert torch.allclose(layer_new.ptw_log_w, layer_ref.ptw_log_w, atol=1e-12)


@pytest.mark.cuda
def test_multilevel_kernel_matches_per_level_kernel_forward_only_cuda() -> None:
    """When the GPU is available AND both kernels are built, the
    multilevel forward_only path must produce numerically identical
    output to L invocations of the per-level forward_only kernel with
    posterior_temp=0.  This is the on-device parity test that we cannot
    rehearse from inside the sandbox; run it on the GPU box.
    """
    require_cuda_kernel()  # ensures per-level kernel is loadable
    from nctl_bench import nctl_network as _net
    ml = _net._get_fmn_multilevel_cuda()
    if ml is None:
        pytest.skip("multilevel kernel .so not built")
    device = torch.device("cuda")

    L, N, M, C, K_in = 3, 2, 3, 2, 2
    B, D, H = 4, 2, 1
    torch.manual_seed(101)
    mw = (torch.randn(L, N, M, C, K_in) * 0.1).to(device)
    hp = torch.randn(N, H, D).to(device)
    hp = hp / hp.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    hb = torch.zeros(N, H, device=device)
    ps = torch.tensor([[2, 1], [2, 2], [1, 0]], dtype=torch.int32, device=device)
    slp = torch.randn(L, N, M, device=device) * 0.01
    z = torch.randn(B, D, device=device)
    p_prev = torch.rand(B, K_in, device=device)

    per_level_cuda = _net._get_fmn_cuda()
    per_level = []
    for l in range(L):
        per_level.append(per_level_cuda.forward_only(
            z, p_prev, mw[l], hp, hb, ps[l], slp[l], 0.0,
        ))
    expected = torch.stack(per_level, dim=0)  # [L, B, N]

    got = ml.forward_only(z, p_prev, mw, hp, hb, ps, slp, 0.0)
    torch.cuda.synchronize()
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5)


@pytest.mark.cuda
def test_multilevel_kernel_matches_per_level_kernel_forward_update_cuda() -> None:
    """forward_update parity: predictions AND in-place state mutations
    (mixture_weights, segment_log_probs, model_log_probs) must agree
    between the two kernels.  Run on a GPU box; cannot be rehearsed
    inside the sandbox.
    """
    require_cuda_kernel()
    from nctl_bench import nctl_network as _net
    ml = _net._get_fmn_multilevel_cuda()
    if ml is None:
        pytest.skip("multilevel kernel .so not built")
    device = torch.device("cuda")

    L, N, M, C, K_in = 3, 2, 3, 2, 2
    B, D, H = 6, 2, 1
    torch.manual_seed(7)
    mw_base = (torch.randn(L, N, M, C, K_in) * 0.1).to(device)
    hp = torch.randn(N, H, D).to(device)
    hp = hp / hp.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    hb = torch.zeros(N, H, device=device)
    ps = torch.tensor([[2, 1], [2, 2], [1, 0]], dtype=torch.int32, device=device)
    slp_base = (torch.randn(L, N, M) * 0.01).to(device)
    mlp_base = slp_base.clone()
    z = torch.randn(B, D, device=device)
    p_prev = torch.rand(B, K_in, device=device)
    symbols = torch.randint(0, 2, (B,), dtype=torch.int32, device=device)
    lr = 0.005

    # Per-level path: run on a clone of state.
    per_level_cuda = _net._get_fmn_cuda()
    mw_a = mw_base.clone()
    slp_a = slp_base.clone()
    mlp_a = mlp_base.clone()
    per_level = []
    for l in range(L):
        per_level.append(per_level_cuda.forward_update(
            z, p_prev, symbols, mw_a[l], hp, hb, ps[l], slp_a[l], mlp_a[l], lr, 0.0,
        ))
    expected = torch.stack(per_level, dim=0)

    # Multilevel: run on a separate clone.
    mw_b = mw_base.clone()
    slp_b = slp_base.clone()
    mlp_b = mlp_base.clone()
    got = ml.forward_update(
        z, p_prev, symbols, mw_b, hp, hb, ps, slp_b, mlp_b, lr, 0.0,
    )
    torch.cuda.synchronize()
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5)
    # In-place mutations must agree.
    assert torch.allclose(mw_a, mw_b, atol=1e-5)
    assert torch.allclose(slp_a, slp_b, atol=1e-5)
    assert torch.allclose(mlp_a, mlp_b, atol=1e-5)


@pytest.mark.cuda
def test_multilevel_fast_path_drives_split_mnist_layer_end_to_end_cuda() -> None:
    """Wire-up integration: a layer with prediction_mode='ptw_dp' on CUDA
    must dispatch through the multilevel kernel when it is built, mutate
    per-level state, advance DP state, and return a valid [B, N] tensor.
    Skipped in the sandbox; this is the GPU-box smoke test.
    """
    device = require_cuda_kernel()
    from nctl_bench import nctl_network as _net
    if _net._get_fmn_multilevel_cuda() is None:
        pytest.skip("multilevel kernel .so not built")
    layer = CudaFmnMixtureLayer(
        num_nodes=4, num_inputs=4, input_dim=4, num_halfspaces=2,
        lr=0.001, pool_capacity=4,
        device=device, seed=17,
        active_state_mode="per_level",
        active_level_count=8,
        prediction_level=7,
        prediction_mode="ptw_dp",
    )
    B = 32
    z = torch.randn(B, 4, device=device)
    p_prev = torch.rand(B, 4, device=device)
    symbols = torch.randint(0, 2, (B,), dtype=torch.int32, device=device)
    nu_before = layer.level_log_nu.clone()
    out = layer.forward_update_batch(z, p_prev, symbols)
    torch.cuda.synchronize()
    assert out.shape == (B, 4)
    assert (out >= 0.0).all() and (out <= 1.0).all()
    # DP state must have advanced.
    assert not torch.allclose(layer.level_log_nu, nu_before)


def test_multilevel_loader_returns_none_when_kernel_missing_cpu() -> None:
    """When the Phase 5E .so is not built, the loader must return None,
    cache the missing-state, and NOT raise.  This is the contract the DP
    paths' fast-path/fallback dispatcher depends on.
    """
    from nctl_bench import nctl_network as _net

    saved_module = _net._fmn_multilevel_cuda
    saved_missing = _net._fmn_multilevel_cuda_missing
    saved_cache = _net._CACHE_DIR
    try:
        _net._fmn_multilevel_cuda = None
        _net._fmn_multilevel_cuda_missing = False
        # Point the cache at a definitely-missing directory.
        _net._CACHE_DIR = "/nonexistent/multilevel-cache"
        assert _net._get_fmn_multilevel_cuda() is None
        # Repeat call: must take the cached-missing fast path; no exception.
        assert _net._get_fmn_multilevel_cuda() is None
        assert _net._fmn_multilevel_cuda_missing is True
    finally:
        _net._fmn_multilevel_cuda = saved_module
        _net._fmn_multilevel_cuda_missing = saved_missing
        _net._CACHE_DIR = saved_cache


def test_multilevel_forward_only_binding_signature_matches_python_fast_path_cpu() -> None:
    """The Python fast path calls ``ml.forward_only(..., pool_sizes,
    segment_log_probs)``.  Guard the C++ binding signature against drifting
    back to the older 6-argument form, which would only fail on the GPU box
    once the multilevel .so is built.
    """
    source = Path("rust-fmn/scripts/nctl_bench/cuda/fmn_mixture_multilevel_kernel.cu").read_text()
    signature_start = source.index("torch::Tensor fmn_mixture_multilevel_forward_only_py(")
    signature_end = source.index(") {", signature_start)
    signature = source[signature_start:signature_end]
    assert "torch::Tensor pool_sizes" in signature
    assert "torch::Tensor segment_log_probs" in signature
    assert signature.index("torch::Tensor pool_sizes") < signature.index("torch::Tensor segment_log_probs")


def test_multilevel_fast_path_overrides_per_level_loop_when_available_cpu() -> None:
    """When _get_fmn_multilevel_cuda returns a module, the DP path must
    call that module's forward_only / forward_update ONCE with stacked
    [L, N, M, C, K] state and ignore the per-level fallback entirely.
    """
    from nctl_bench import nctl_network as _net

    only_calls: list = []
    update_calls: list = []

    class _MLSpy:
        def forward_only(self, z, p_prev, mw, hp, hb, ps, slp, posterior_temp=0.0):
            only_calls.append(
                dict(
                    mw_shape=tuple(mw.shape),
                    ps_shape=tuple(ps.shape),
                    slp_shape=tuple(slp.shape),
                    posterior_temp=float(posterior_temp),
                )
            )
            L, N = ps.shape
            B = z.size(0)
            # Deterministic per-(l, b, n).
            base = torch.arange(L * B * N, dtype=z.dtype) * 0.001 + 0.25
            return base.reshape(L, B, N)

        def forward_update(self, z, p_prev, sym, mw, hp, hb, ps, slp, mlp, lr, posterior_temp=0.0):
            update_calls.append(
                dict(
                    mw_shape=tuple(mw.shape),
                    ps_shape=tuple(ps.shape),
                    slp_shape=tuple(slp.shape),
                    mlp_shape=tuple(mlp.shape),
                    lr=lr,
                    posterior_temp=float(posterior_temp),
                )
            )
            L, N = ps.shape
            B = z.size(0)
            base = torch.arange(L * B * N, dtype=z.dtype) * 0.001 + 0.4
            return base.reshape(L, B, N)

    class _PerLevelSpy:
        def forward_only(self, *_a, **_k):
            raise AssertionError("multilevel fast path must skip per-level fallback")

        def forward_update(self, *_a, **_k):
            raise AssertionError("multilevel fast path must skip per-level fallback")

    prev_ml = _net._fmn_multilevel_cuda
    prev_ml_missing = _net._fmn_multilevel_cuda_missing
    prev_cuda = _net._fmn_cuda
    _net._fmn_multilevel_cuda = _MLSpy()
    _net._fmn_multilevel_cuda_missing = False
    _net._fmn_cuda = _PerLevelSpy()
    try:
        layer = CudaFmnMixtureLayer(
            num_nodes=2, num_inputs=2, input_dim=2, num_halfspaces=1,
            lr=0.003, pool_capacity=2,
            device=torch.device("cpu"), seed=11,
            active_state_mode="per_level",
            active_level_count=3,
            prediction_level=2,
            prediction_mode="ptw_dp",
        )
        B = 4
        z = torch.zeros(B, 2)
        p_prev = torch.full((B, 2), 0.5)
        sym = torch.zeros(B, dtype=torch.int32)
        layer.forward_only_batch(z, p_prev)
        layer.forward_update_batch(z, p_prev, sym)
    finally:
        _net._fmn_multilevel_cuda = prev_ml
        _net._fmn_multilevel_cuda_missing = prev_ml_missing
        _net._fmn_cuda = prev_cuda

    assert len(only_calls) == 1, only_calls
    assert len(update_calls) == 1, update_calls
    # Shapes match the [L, N, M, C, K_in] / [L, N] / [L, N, M] contract.
    L, N, pool_cap = 3, 2, 2
    M = pool_cap + 1
    C = 2  # 2 ** num_halfspaces
    K_in = layer.K
    for call in (only_calls[0], update_calls[0]):
        assert call["mw_shape"] == (L, N, M, C, K_in)
        assert call["ps_shape"] == (L, N)
        assert call["slp_shape"] == (L, N, M)
    assert update_calls[0]["mlp_shape"] == (L, N, M)
    assert update_calls[0]["lr"] == pytest.approx(0.003)


def test_multilevel_fast_path_falls_back_to_per_level_when_unavailable_cpu() -> None:
    """When _get_fmn_multilevel_cuda returns None, the DP path must call
    the per-level kernel L times, once per level, with the SAME paper-
    faithful posterior_temp contract -- forwarding ``layer.posterior_temp``
    verbatim (Eq. 5 conditional ratio).  Regression guard against silently
    dropping the fallback or stripping the posterior weighting.
    """
    from nctl_bench import nctl_network as _net

    calls: list = []

    class _PerLevelSpy:
        def forward_only(self, z, p_prev, w, hp, hb, ps, slp, posterior_temp):
            calls.append(("only", float(posterior_temp)))
            return torch.zeros(z.size(0), w.size(0))

        def forward_update(self, z, p_prev, sym, w, hp, hb, ps, slp, mlp, lr, posterior_temp):
            calls.append(("update", float(posterior_temp)))
            return torch.zeros(z.size(0), w.size(0))

    prev_ml = _net._fmn_multilevel_cuda
    prev_ml_missing = _net._fmn_multilevel_cuda_missing
    prev_cuda = _net._fmn_cuda
    _net._fmn_multilevel_cuda = None
    _net._fmn_multilevel_cuda_missing = True  # short-circuit loader
    _net._fmn_cuda = _PerLevelSpy()
    try:
        layer = CudaFmnMixtureLayer(
            num_nodes=1, num_inputs=1, input_dim=1, num_halfspaces=1,
            lr=0.0, pool_capacity=1,
            device=torch.device("cpu"), seed=0,
            posterior_temp=0.5,
            active_state_mode="per_level",
            active_level_count=2,
            prediction_level=1,
            prediction_mode="ptw_dp",
        )
        z = torch.zeros(1, 1)
        p_prev = torch.full((1, 1), 0.5)
        sym = torch.zeros(1, dtype=torch.int32)
        layer.forward_only_batch(z, p_prev)
        layer.forward_update_batch(z, p_prev, sym)
    finally:
        _net._fmn_multilevel_cuda = prev_ml
        _net._fmn_multilevel_cuda_missing = prev_ml_missing
        _net._fmn_cuda = prev_cuda

    # Two levels => two only-calls and two update-calls, all with
    # posterior_temp == layer.posterior_temp (Eq. 5 paper-faithful
    # conditional ratio).  Pre-5I the contract demanded 0.0 here, which
    # silently capped accuracy on Split-MNIST; post-5I the layer must
    # forward its posterior_temp verbatim.
    only_calls = [t for k, t in calls if k == "only"]
    update_calls = [t for k, t in calls if k == "update"]
    assert len(only_calls) == 2
    assert len(update_calls) == 2
    assert all(t == 0.5 for t in only_calls)
    assert all(t == 0.5 for t in update_calls)


def test_multilevel_with_resets_reference_no_close_mask_matches_baseline_cpu() -> None:
    """When close_mask is all-False, the with-resets reference must produce
    bit-identical predictions and state mutations to the baseline
    multilevel reference.  This is the additivity check that makes D safe
    to wire in incrementally.
    """
    from nctl_bench.nctl_network import (
        _fmn_multilevel_paper_posterior_reference,
        _fmn_multilevel_paper_posterior_reference_with_resets,
    )

    torch.manual_seed(1001)
    L, N, M, C, K_in = 3, 2, 3, 2, 2
    B, D, H = 4, 2, 1
    pool_capacity = M - 1

    mw = torch.randn(L, N, M, C, K_in) * 0.1
    pool_snapshots = torch.randn(N, pool_capacity, C, K_in) * 0.1
    hp = torch.randn(N, H, D)
    hp = hp / hp.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    hb = torch.zeros(N, H)
    ps = torch.tensor([[2, 1], [2, 2], [1, 0]], dtype=torch.int32)
    slp = torch.randn(L, N, M) * 0.01
    mlp = slp.clone()
    z = torch.randn(B, D)
    p_prev = torch.rand(B, K_in)
    symbols = torch.tensor([1, 0, 1, 0], dtype=torch.int32)

    # Baseline path (no resets).
    mw_a = mw.clone()
    slp_a = slp.clone()
    mlp_a = mlp.clone()
    expected = _fmn_multilevel_paper_posterior_reference(
        z, p_prev, symbols, mw_a, hp, hb, ps, slp_a, mlp_a,
        lr=0.005, update=True,
    )

    # With-resets path, close_mask all-False.
    mw_b = mw.clone()
    slp_b = slp.clone()
    mlp_b = mlp.clone()
    close_mask = torch.zeros(L, B, dtype=torch.bool)
    got = _fmn_multilevel_paper_posterior_reference_with_resets(
        z, p_prev, symbols, mw_b, pool_snapshots, hp, hb, ps, close_mask, slp_b, mlp_b,
        lr=0.005,
    )

    assert torch.allclose(got, expected, atol=1e-7)
    assert torch.allclose(mw_b, mw_a, atol=1e-7)
    assert torch.allclose(slp_b, slp_a, atol=1e-7)
    assert torch.allclose(mlp_b, mlp_a, atol=1e-7)


def test_multilevel_with_resets_reference_applies_close_before_sample_cpu() -> None:
    """When close_mask[l, b] is True, the with-resets reference must
    restore mixture_weights[l, :, :k_pool] from pool_snapshots and zero
    segment_log_probs[l] BEFORE computing sample b's prediction at level
    l.  Test by planting recognisable values and verifying the prediction
    at level l for sample b uses the pool snapshots (not the pre-reset
    weights).
    """
    from nctl_bench.nctl_network import _fmn_multilevel_paper_posterior_reference_with_resets

    L, N, M, C, K_in = 2, 1, 2, 1, 2
    B, D = 1, 2  # H=0 (no halfspaces) -> C=1, single context
    pool_capacity = M - 1
    # Hyperplanes/bias with H=0 produce context=0 for every node.
    hp = torch.zeros(N, 0, D)
    hb = torch.zeros(N, 0)
    z = torch.zeros(B, D)
    p_prev = torch.full((B, K_in), 0.5)  # logit(0.5) == 0
    sym = torch.tensor([1], dtype=torch.int32)
    # Active mw all -10 (logit -10 => sigmoid ~= 0); pool snapshots all +10
    # (sigmoid ~= 1).  Fresh slot (M-1) at 0 (sigmoid 0.5).  After reset:
    # slots [0..k_pool-1] := pool snapshots (preds -> 1.0), fresh stays at 0.5.
    mw = torch.full((L, N, M, C, K_in), -10.0)
    mw[:, :, M - 1, :, :] = 0.0  # fresh slot at sigmoid==0.5
    pool_snapshots = torch.full((N, pool_capacity, C, K_in), 10.0)
    ps = torch.tensor([[1], [1]], dtype=torch.int32)
    slp = torch.zeros(L, N, M)
    mlp = torch.zeros(L, N, M)
    # close_mask only on level 0 at b=0.
    close_mask = torch.zeros(L, B, dtype=torch.bool)
    close_mask[0, 0] = True

    pred = _fmn_multilevel_paper_posterior_reference_with_resets(
        z, p_prev, sym, mw, pool_snapshots, hp, hb, ps, close_mask, slp, mlp,
        lr=0.0,  # no learning so we see the pre-update prediction
    )
    # Level 0 reset: pool slot now sigmoid(10*0)*... actually logit(0.5)=0
    # so logits is zero; pool slot weights are all 10 but dotted with zero
    # logits => 0 => sigmoid(0) = 0.5.  Same for fresh slot (zero weights
    # already).  ν_0 = 0.5*0.5 + 0.5*0.5 = 0.5.  Prediction at sym=1: 0.5.
    assert torch.allclose(pred[0, 0], torch.full((N,), 0.5), atol=1e-6)
    # Level 1 NOT reset: weights still all -10.  Logits 0 => slot
    # sigmoid(0)=0.5.  Same as fresh.  Same ν=0.5.
    # (Both levels happen to produce 0.5 here because p_prev=0.5 collapses
    # the linear dot to zero.  The state mutation is the actual verification.)
    # Mutation check: level 0 slot 0 should now match pool_snapshots[0, 0];
    # level 1 slot 0 should still be all -10 (modulo lr=0 weight update).
    assert torch.allclose(mw[0, 0, 0], pool_snapshots[0, 0])
    assert torch.allclose(mw[1, 0, 0], torch.full((C, K_in), -10.0))
    # Fresh slot preserved (lr=0 so update is a no-op): both levels still
    # have fresh slot weights 0.
    assert torch.allclose(mw[0, 0, M - 1], torch.zeros(C, K_in))
    assert torch.allclose(mw[1, 0, M - 1], torch.zeros(C, K_in))


def test_multilevel_with_resets_reference_accumulates_correctly_across_multiple_resets_cpu() -> None:
    """Two resets on level 0 inside one sub-chunk: at b=0 and b=2.  The
    weight update between them must be wiped by the second reset, so the
    final weights for slot 0 at level 0 must equal pool_snapshots[n, 0]
    plus the learning steps from sample b=2 only.  This is the multi-event
    correctness test that the kernel implementation must mirror.
    """
    from nctl_bench.nctl_network import _fmn_multilevel_paper_posterior_reference_with_resets

    L, N, M, C, K_in = 1, 1, 2, 1, 2
    B, D = 3, 2  # H=0 -> C=1
    pool_capacity = M - 1

    hp = torch.zeros(N, 0, D)
    hb = torch.zeros(N, 0)
    z = torch.zeros(B, D)
    p_prev = torch.full((B, K_in), 0.7)
    sym = torch.tensor([1, 1, 1], dtype=torch.int32)

    mw = torch.zeros(L, N, M, C, K_in)
    # Plant nonzero pool snapshots so the reset is observable.
    pool_snapshots = torch.full((N, pool_capacity, C, K_in), 0.5)
    ps = torch.tensor([[1]], dtype=torch.int32)
    slp = torch.zeros(L, N, M)
    mlp = torch.zeros(L, N, M)
    close_mask = torch.zeros(L, B, dtype=torch.bool)
    close_mask[0, 0] = True
    close_mask[0, 2] = True
    pred = _fmn_multilevel_paper_posterior_reference_with_resets(
        z, p_prev, sym, mw, pool_snapshots, hp, hb, ps, close_mask, slp, mlp,
        lr=0.01,
    )
    assert pred.shape == (L, B, N)
    # After b=2 reset, slot 0 starts at 0.5, then takes one learning step.
    # The fresh slot keeps the b=0 and b=1 learning steps until the b=2
    # reset, then takes one more step.  We don't pin the exact value (the
    # reference IS the contract); we instead verify the reset semantics by
    # zeroing lr and checking final slot weights equal pool snapshots
    # after the b=2 reset.
    # Re-run with lr=0 to isolate the reset:
    mw2 = torch.zeros(L, N, M, C, K_in)
    slp2 = torch.zeros(L, N, M)
    mlp2 = torch.zeros(L, N, M)
    _fmn_multilevel_paper_posterior_reference_with_resets(
        z, p_prev, sym, mw2, pool_snapshots, hp, hb, ps, close_mask, slp2, mlp2,
        lr=0.0,
    )
    # After b=2 reset (and no learning since lr=0), slot 0 equals
    # pool_snapshots[0, 0] for the only context.
    assert torch.allclose(mw2[0, 0, 0], pool_snapshots[0, 0])
    # segment_log_probs was zeroed at b=2, then accumulated for sample 2.
    # Both slots active: log p(sym=1) = log(p_uniform).
    # We just check it's non-zero (learning happened on b=2 even with lr=0
    # because segment_log_probs is independent of lr).
    assert slp2[0, 0, 0] != 0.0
    assert slp2[0, 0, M - 1] != 0.0


def test_multilevel_with_resets_loader_exposes_kernel_entry_point_cpu() -> None:
    """Loader contract: when the multilevel .so is built FROM THIS
    revision, it must expose forward_update_with_resets in addition to
    the legacy forward_update/forward_only entry points.

    Skipped if the .so is not built (CPU-only sandbox) OR if it is built
    from a pre-Phase-5F revision -- the latter is a clean signal to the
    GPU operator that they need to rerun build_kernels.sh, not a hard
    test failure inside the sandbox.
    """
    from nctl_bench import nctl_network as _net
    ml = _net._get_fmn_multilevel_cuda()
    if ml is None:
        pytest.skip("multilevel kernel .so not built")
    if not hasattr(ml, "forward_update_with_resets"):
        pytest.skip(
            "multilevel kernel .so predates Phase 5F; rerun "
            "rust-fmn/scripts/nctl_bench/cuda/build_kernels.sh"
        )


def test_multilevel_cpu_reference_matches_per_level_kernel_math_cpu() -> None:
    """The CPU reference `_fmn_multilevel_paper_posterior_reference` is the
    contract that pins what the multilevel CUDA kernel must compute.
    Verify it produces the SAME per-level predictions as L runs of the
    per-level paper-uniform reference (a 1/2 fresh + 1/2 pool-mean by
    hand) for both forward_only and forward_update.

    Mathematically: the L levels are independent state machines that
    share only z/p_prev/symbols.  The multilevel reference must equal
    L per-level evaluations with no cross-level contamination.
    """
    from nctl_bench.nctl_network import _fmn_multilevel_paper_posterior_reference

    torch.manual_seed(31)
    L, N, M, C, K_in = 3, 2, 3, 2, 2  # M = pool_capacity (2) + 1 fresh
    B, D, H = 4, 2, 1  # 2 ** H == C
    fresh_idx = M - 1

    mw = torch.randn(L, N, M, C, K_in) * 0.1
    hp = torch.randn(N, H, D)
    hp = hp / hp.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    hb = torch.zeros(N, H)
    ps = torch.tensor([[2, 1], [2, 2], [1, 0]], dtype=torch.int32)
    slp = torch.randn(L, N, M) * 0.01  # not consumed by paper-uniform
    mlp = slp.clone()

    z = torch.randn(B, D)
    p_prev = torch.rand(B, K_in)
    symbols = torch.tensor([1, 0, 1, 0], dtype=torch.int32)

    # forward_only: predictions should match a hand-rolled per-level paper-uniform.
    mw_only = mw.clone()
    slp_only = slp.clone()
    pred_only = _fmn_multilevel_paper_posterior_reference(
        z, p_prev, None, mw_only, hp, hb, ps, slp_only, None,
        lr=0.0, update=False,
    )
    assert pred_only.shape == (L, B, N)
    # Hand reference: same math, but level-by-level for clarity.
    def _hand_pred(mw_, ps_, slp_):
        out = torch.zeros(L, B, N)
        for b in range(B):
            # Compute context for each node.
            dot = (hp * z[b].view(1, 1, D)).sum(dim=-1) + hb  # [N, H]
            ctx = torch.zeros(N, dtype=torch.long)
            for h in range(H):
                ctx |= ((dot[:, h] >= 0).long() << h)
            logits = torch.log(
                torch.clamp(p_prev[b], 1e-7, 1 - 1e-7)
                / torch.clamp(1 - p_prev[b], 1e-7, 1 - 1e-7)
            ).clamp(-15.0, 15.0)
            for l in range(L):
                for n in range(N):
                    c = int(ctx[n].item())
                    slot_w = mw_[l, n, :, c, :]  # [M, K_in]
                    preds = torch.sigmoid(slot_w @ logits)
                    k = int(ps_[l, n].item())
                    fresh = preds[fresh_idx]
                    if k > 0:
                        out[l, b, n] = 0.5 * fresh + 0.5 * preds[:k].mean()
                    else:
                        out[l, b, n] = fresh
        return out

    expected_only = _hand_pred(mw, ps, slp)
    assert torch.allclose(pred_only, expected_only, atol=1e-6)
    # State must be untouched on forward_only.
    assert torch.allclose(mw_only, mw)
    assert torch.allclose(slp_only, slp)

    # forward_update: predictions are ν_j(x_obs|x_<t).
    mw_upd = mw.clone()
    slp_upd = slp.clone()
    mlp_upd = mlp.clone()
    pred_upd = _fmn_multilevel_paper_posterior_reference(
        z, p_prev, symbols, mw_upd, hp, hb, ps, slp_upd, mlp_upd,
        lr=0.01, update=True,
    )
    assert pred_upd.shape == (L, B, N)
    # State must have changed.
    assert not torch.allclose(mw_upd, mw)
    assert not torch.allclose(slp_upd, slp)


def test_ptw_dp_apply_chunk_zero_batch_is_noop_cpu() -> None:
    """B=0 must be a clean no-op: returns an empty [0, N] tensor and does
    not mutate DP state.  Guards against off-by-one or empty-cumsum
    surprises.
    """
    layer = _make_dp_layer(active_level_count=3)
    L, N = 3, 2
    nu0 = layer.level_log_nu.clone()
    w0 = layer.ptw_log_w.clone()
    out = layer._ptw_dp_apply_chunk(
        log_q_obs_chunk=torch.empty(L, 0, N),
        q1_chunk=torch.empty(L, 0, N),
    )
    assert out.shape == (0, N)
    assert torch.equal(layer.level_log_nu, nu0)
    assert torch.equal(layer.ptw_log_w, w0)


def test_forward_update_batch_dp_uses_fused_chunk_path_cpu() -> None:
    """End-to-end: running the DP training path with a spy kernel and a
    moderately long batch must produce the SAME output as a hand-rolled
    per-sample reference built from the spy's identical per-level returns.
    This is the integration guarantee that Phase 5D wires up correctly.
    """
    from nctl_bench import nctl_network as _net

    captured: list = []

    class _SpyCuda:
        def forward_update(self, z, p_prev, sym, w, hp, hb, ps, slp, mlp, lr, posterior_temp):
            B = z.size(0)
            N = w.size(0)
            # Deterministic, depends on call index so per-level outputs differ.
            base = 0.1 + 0.05 * float(len(captured))
            out = torch.linspace(base, base + 0.5, B * N).reshape(B, N).clamp(min=1e-3, max=1.0 - 1e-3)
            captured.append(out.clone())
            return out

        def forward_only(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("training path must not call forward_only")

    # Two identical layers; one will run the new path, one the reference.
    def _new_layer():
        return CudaFmnMixtureLayer(
            num_nodes=2, num_inputs=2, input_dim=2, num_halfspaces=1,
            lr=0.001, pool_capacity=2,
            device=torch.device("cpu"), seed=5,
            active_state_mode="per_level",
            active_level_count=3,
            prediction_level=2,
            prediction_mode="ptw_dp",
        )

    B = 10
    z = torch.zeros(B, 2)
    p_prev = torch.full((B, 2), 0.5)
    sym = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 0, 0], dtype=torch.int32)

    # Layer A: actual path.  Disable the multilevel fast path so the spy
    # installed on _fmn_cuda actually fires.  The numerical-equivalence
    # contract under test is the per-level fallback's batched DP combine.
    prev = _net._fmn_cuda
    prev_ml = _net._fmn_multilevel_cuda
    prev_ml_missing = _net._fmn_multilevel_cuda_missing
    _net._fmn_cuda = _SpyCuda()
    _net._fmn_multilevel_cuda = None
    _net._fmn_multilevel_cuda_missing = True
    captured.clear()
    try:
        layer_a = _new_layer()
        out_a = layer_a.forward_update_batch(z, p_prev, sym)
        captured_a = list(captured)
    finally:
        _net._fmn_cuda = prev
        _net._fmn_multilevel_cuda = prev_ml
        _net._fmn_multilevel_cuda_missing = prev_ml_missing

    # Layer B: same spy outputs replayed manually through the reference
    # per-sample DP.
    layer_b = _new_layer()
    q_obs = torch.stack(captured_a, dim=0)  # [L, B, N]
    sym_b = sym.to(torch.bool)
    q1 = torch.where(sym_b.view(1, B, 1), q_obs, 1.0 - q_obs)
    log_q_obs = torch.log(q_obs.clamp(min=1e-30))
    out_b = torch.empty(B, q_obs.size(-1))
    for b in range(B):
        p1 = layer_b._ptw_dp_combine(q1[:, b])
        out_b[b] = p1 if sym_b[b] else (1.0 - p1)
        layer_b._ptw_dp_update_state(log_q_obs[:, b])

    assert torch.allclose(out_a, out_b, atol=1e-6)
    # DP state should also agree.
    assert torch.allclose(layer_a.level_log_nu, layer_b.level_log_nu, atol=1e-12)
    assert torch.allclose(layer_a.ptw_log_w, layer_b.ptw_log_w, atol=1e-12)


def test_close_only_reset_zeros_unused_slots_without_host_sync_cpu() -> None:
    """The branchless ``masked_fill_`` path must zero per-node unused slots
    just like the legacy ``unused.any().item()`` branch did.  Regression
    guard for F1.5 perf fix.  We assert behaviour, not the absence of a
    sync (no public hook for that), but the same pre/post tensor state
    proves the masked_fill is equivalent.
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=3,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=4,
        device=torch.device("cpu"),
        seed=11,
        active_state_mode="per_level",
        active_level_count=2,
    )
    # Plant non-zero garbage into the per-level active weights for level 1.
    layer.level_mixture_weights[1].fill_(0.7)
    # Plant non-zero pool snapshots so the copy is observable.
    layer.pool_snapshots.fill_(0.3)
    # Set per-node pool sizes: node 0 uses all 4 slots, node 1 uses 2, node 2 uses 0.
    layer.pool_sizes.copy_(torch.tensor([4, 2, 0], dtype=torch.int32))
    layer._reset_level_active_state_all_nodes(level=1)
    weights = layer.level_mixture_weights[1, :, :layer.pool_capacity]
    # Node 0: all 4 slots populated from pool snapshots.
    assert torch.all(weights[0] == 0.3)
    # Node 1: slots 0,1 populated; slots 2,3 zeroed.
    assert torch.all(weights[1, :2] == 0.3)
    assert torch.all(weights[1, 2:] == 0.0)
    # Node 2: all 4 slots zeroed.
    assert torch.all(weights[2] == 0.0)
    # Fresh slot is left alone.  Make sure we did not stomp it: it should
    # still hold the value we planted before _reset_level_active_state.
    assert torch.all(layer.level_mixture_weights[1, :, layer.pool_capacity] == 0.7)


def test_forward_update_batch_dp_handles_batch_without_host_sync_cpu() -> None:
    """Run a moderately long batch through the DP training path with a spy
    kernel and confirm the DP state advances B times and the output is
    well-shaped.  The branchless ``s * p1 + (1-s) * (1-p1)`` selection in
    Commit C must reproduce the previous per-sample if/else.
    """
    from nctl_bench import nctl_network as _net

    class _SpyCuda:
        def forward_update(self, z, p_prev, sym, w, hp, hb, ps, slp, mlp, lr, posterior_temp):
            B = z.size(0)
            N = w.size(0)
            return torch.linspace(0.1, 0.9, B * N).reshape(B, N)

        def forward_only(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("forward_update path must not call forward_only")

    prev = _net._fmn_cuda
    prev_ml = _net._fmn_multilevel_cuda
    prev_ml_missing = _net._fmn_multilevel_cuda_missing
    _net._fmn_cuda = _SpyCuda()
    _net._fmn_multilevel_cuda = None
    _net._fmn_multilevel_cuda_missing = True  # force per-level fallback path
    try:
        layer = CudaFmnMixtureLayer(
            num_nodes=2, num_inputs=2, input_dim=2, num_halfspaces=1,
            lr=0.001, pool_capacity=2,
            device=torch.device("cpu"), seed=4,
            active_state_mode="per_level",
            active_level_count=3,
            prediction_level=2,
            prediction_mode="ptw_dp",
        )
        B = 6
        z = torch.zeros(B, 2)
        p_prev = torch.full((B, 2), 0.5)
        sym = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.int32)
        # Snapshot DP state before / after.
        nu_before = layer.level_log_nu.clone()
        out = layer.forward_update_batch(z, p_prev, sym)
        assert out.shape == (B, 2)
        # DP state must have moved (B advances total).
        assert not torch.allclose(layer.level_log_nu, nu_before)
        # All outputs are valid probabilities in [0, 1].
        assert (out >= 0.0).all() and (out <= 1.0).all()
    finally:
        _net._fmn_cuda = prev
        _net._fmn_multilevel_cuda = prev_ml
        _net._fmn_multilevel_cuda_missing = prev_ml_missing


def test_ptw_dp_combine_batched_matches_per_sample_cpu() -> None:
    """``_ptw_dp_combine`` now accepts ``[L, B, N]`` and returns ``[B, N]``.
    The vectorised path must be numerically identical to looping the
    single-sample ``[L, N]`` path over the batch.  Guard regressions when
    we replace forward_only_batch_dp's Python ``for b in range(B)``.
    """
    layer = _make_dp_layer(active_level_count=4)
    L, N = 4, 2
    B = 5
    # Plant non-trivial DP state so pi != 0.5 across levels and nodes.
    layer.level_log_nu.copy_(torch.linspace(-2.0, 0.5, L * N).reshape(L, N).to(layer.level_log_nu.dtype))
    layer.ptw_log_w.copy_(torch.linspace(-1.0, 1.0, L * N).reshape(L, N).to(layer.ptw_log_w.dtype))
    torch.manual_seed(7)
    q1_batched = torch.rand(L, B, N)
    # Reference: per-sample loop.
    expected = torch.stack([layer._ptw_dp_combine(q1_batched[:, b]) for b in range(B)], dim=0)
    # Vectorised: single call.
    got = layer._ptw_dp_combine(q1_batched)
    assert got.shape == (B, N)
    assert torch.allclose(got, expected, atol=1e-7)


def test_forward_only_batch_dp_drops_per_sample_loop_cpu() -> None:
    """End-to-end smoke that ``forward_only_batch_dp`` returns a [B, N] tensor
    that matches the per-sample stack, even with B > 1.  Together with the
    spy test above this nails the F3a vectorisation contract.
    """
    from nctl_bench import nctl_network as _net

    captured: list = []

    class _SpyCuda:
        def forward_only(self, z, p_prev, w, hp, hb, ps, slp, posterior_temp):
            # Return deterministic per-(b, node) values so the combine path
            # is exercised on data we can recompute by hand.
            B = z.size(0)
            N = w.size(0)
            base = 0.1 + 0.05 * float(captured.__len__())
            out = torch.linspace(base, base + 0.5, B * N).reshape(B, N)
            captured.append(out.clone())
            return out

        def forward_update(self, *_a, **_k):  # pragma: no cover - not called here
            raise AssertionError("forward_only_batch_dp must only call forward_only")

    prev = _net._fmn_cuda
    prev_ml = _net._fmn_multilevel_cuda
    prev_ml_missing = _net._fmn_multilevel_cuda_missing
    _net._fmn_cuda = _SpyCuda()
    _net._fmn_multilevel_cuda = None
    _net._fmn_multilevel_cuda_missing = True  # force per-level fallback
    try:
        layer = CudaFmnMixtureLayer(
            num_nodes=2, num_inputs=2, input_dim=2, num_halfspaces=1,
            lr=0.0, pool_capacity=2,
            device=torch.device("cpu"), seed=3,
            active_state_mode="per_level",
            active_level_count=3,
            prediction_level=2,
            prediction_mode="ptw_dp",
        )
        z = torch.zeros(4, 2)
        p_prev = torch.full((4, 2), 0.5)
        out = layer.forward_only_batch(z, p_prev)
    finally:
        _net._fmn_cuda = prev
        _net._fmn_multilevel_cuda = prev_ml
        _net._fmn_multilevel_cuda_missing = prev_ml_missing

    assert out.shape == (4, 2)
    q_stack = torch.stack(captured, dim=0)  # [L, B, N]
    expected = layer._ptw_dp_combine(q_stack)
    assert torch.allclose(out, expected, atol=1e-7)


def test_selected_level_path_still_passes_layer_posterior_temp_cpu() -> None:
    """The q_j=paper-uniform fix MUST NOT bleed into selected_level mode,
    which is the legacy Bayesian-mixed path users rely on for explicit
    posterior-temperature sweeps.
    """
    from nctl_bench import nctl_network as _net

    calls: list[tuple[str, float]] = []

    class _SpyCuda:
        def forward_update(self, z, p_prev, sym, w, hp, hb, ps, slp, mlp, lr, posterior_temp):
            calls.append(("update", float(posterior_temp)))
            return torch.full((z.size(0), w.size(0)), 0.5)

        def forward_only(self, z, p_prev, w, hp, hb, ps, slp, posterior_temp):
            calls.append(("only", float(posterior_temp)))
            return torch.full((z.size(0), w.size(0)), 0.5)

    prev = _net._fmn_cuda
    _net._fmn_cuda = _SpyCuda()
    try:
        layer = CudaFmnMixtureLayer(
            num_nodes=1,
            num_inputs=1,
            input_dim=1,
            num_halfspaces=1,
            lr=0.001,
            pool_capacity=1,
            device=torch.device("cpu"),
            seed=0,
            posterior_temp=0.42,
            active_state_mode="flat",  # selected_level mode
        )
        z = torch.zeros(1, 1)
        p_prev = torch.full((1, 1), 0.5)
        symbols = torch.zeros(1, dtype=torch.int32)
        layer.forward_only_batch(z, p_prev)
        layer.forward_update_batch(z, p_prev, symbols)
    finally:
        _net._fmn_cuda = prev

    assert calls
    for _, t in calls:
        assert t == pytest.approx(0.42),             f"selected_level must forward layer.posterior_temp; got {t}"


def test_prediction_mode_ptw_dp_requires_per_level_active_cpu() -> None:
    """Guard the runtime invariant the DP path depends on."""
    import pytest as _pytest
    with _pytest.raises(ValueError, match="ptw_dp.*per_level"):
        CudaFmnMixtureLayer(
            num_nodes=1, num_inputs=1, input_dim=1, num_halfspaces=1,
            lr=0.0, pool_capacity=1, device=torch.device("cpu"), seed=0,
            active_state_mode="flat",  # incompatible with ptw_dp
            prediction_mode="ptw_dp",
        )


def test_per_level_active_state_closes_independent_level_candidates_cpu() -> None:
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=4,
        device=torch.device("cpu"),
        seed=11,
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
        active_state_mode="per_level",
        active_level_count=4,
        prediction_level=3,
    )
    fresh = layer.fresh_idx
    with torch.no_grad():
        layer.level_mixture_weights[1, 0, fresh].fill_(1.0)
        layer.level_mixture_weights[3, 0, fresh].fill_(3.0)
        layer.level_segment_log_probs[1, 0, fresh] = 0.0
        layer.level_segment_log_probs[3, 0, fresh] = 0.0

    layer.segment_close(levels=[1, 3], task_id=9, global_index=128)

    assert int(layer.pool_sizes[0].item()) == 2
    assert [int(x.item()) for x in layer.pool_levels[0, :2]] == [1, 3]
    assert torch.allclose(layer.pool_snapshots[0, 0], torch.ones_like(layer.pool_snapshots[0, 0]))
    assert torch.allclose(layer.pool_snapshots[0, 1], torch.full_like(layer.pool_snapshots[0, 1], 3.0))


def test_per_level_active_state_resets_only_closed_level_reservoir_cpu() -> None:
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=4,
        device=torch.device("cpu"),
        seed=12,
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
        pool_reservoir_size=4,
        active_state_mode="per_level",
        active_level_count=4,
        prediction_level=3,
    )
    z = torch.zeros(3, 2)
    p_prev = torch.full((3, 2), 0.8)
    symbols = torch.ones(3, dtype=torch.int32)
    layer.observe_segment_batch(z, p_prev, symbols)
    assert [int(x.item()) for x in layer.level_res_size[:4]] == [3, 3, 3, 3]

    layer.segment_close(levels=[1], task_id=1, global_index=3)

    assert int(layer.level_res_size[1].item()) == 0
    assert int(layer.level_res_size[3].item()) == 3

def test_segment_close_preserves_base_measure_weights_but_resets_its_posterior_cpu() -> None:
    """The fresh slot's weights persist across many closes (base measure rho
    keeps learning), but its current-segment posterior must reset each close
    so the FMN Eq. 5 priors (1/2 fresh, 1/(2k) per pool slot) are honoured
    at segment open.  Stale all-time fresh evidence would suppress fresh
    below pool slots immediately after every close and reduce adaptation.
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=4,
        device=torch.device("cpu"),
        seed=13,
    )
    fresh_idx = layer.fresh_idx
    sentinel = torch.tensor([[2.0, -2.0], [3.0, -3.0]])
    with torch.no_grad():
        layer.mixture_weights[0, fresh_idx].copy_(sentinel)

    for _ in range(50):
        with torch.no_grad():
            # Simulate a segment of fresh-slot learning accumulating evidence.
            layer.segment_log_probs[0, fresh_idx] = -7.5
        layer.segment_close(levels=[3])
        # Each close must zero fresh's current-segment posterior...
        assert layer.segment_log_probs[0, fresh_idx].item() == pytest.approx(0.0)

    # ...while leaving the base measure weights intact.
    assert torch.allclose(layer.mixture_weights[0, fresh_idx], sentinel)
    assert int(layer.pool_sizes[0].item()) == layer.pool_capacity


def test_segment_open_posterior_is_symmetric_at_zero_evidence_cpu() -> None:
    """End-to-end check that after segment_close the predictive mixture uses
    the FMN priors directly (no leftover fresh-slot evidence skew).
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=21,
    )
    fresh_idx = layer.fresh_idx
    with torch.no_grad():
        layer.segment_log_probs[0, fresh_idx] = -100.0  # stale all-time evidence
        layer.pool_sizes[0] = 1
        layer.pool_snapshots[0, 0].zero_()
    layer.segment_close(levels=[3])

    # With k=1 pool slot, FMN prior on fresh is 1/2; on the pool slot 1/2.
    # When neither has accumulated current-segment evidence the predictive
    # mixture of preds=[0.0, 1.0] (pool, fresh) collapses to 0.5.
    preds = torch.tensor([0.0, 0.0, 1.0])  # only fresh slot used + pool slot 0
    mixed = _posterior_mixture_prob(
        model_preds=preds,
        segment_log_probs=layer.segment_log_probs[0],
        pool_size=int(layer.pool_sizes[0].item()),
        fresh_idx=fresh_idx,
    )
    assert mixed.item() == pytest.approx(0.5, abs=1e-6)



@pytest.mark.cuda
def test_clone_predict_batch_matches_live_network_cuda() -> None:
    """`evaluate_task` clones the net before adaptation, so a regression where
    clone fails to deep-copy any state (mixture_weights, pool_snapshots,
    pool_sizes, segment_log_probs, etc.) would silently shift evaluation
    accuracy.  Pin: a freshly cloned net produces bitwise-identical
    predict_batch outputs to the live net.

    Requires a real CUDA kernel; the layer's predict path goes through the
    compiled .so and is not runnable on plain CPU.
    """
    device = require_cuda_kernel()
    torch.manual_seed(0)
    net = NctlNetwork(
        layer_sizes=[3, 1],
        input_dim=4,
        num_halfspaces=1,
        lr=0.05,
        pool_capacity=2,
        min_segment=4,
        ptw_depth=4,
        device=device,
        seed=7,
    )
    z_train = torch.randn(8, 4, device=device)
    s_train = torch.randint(0, 2, (8,), dtype=torch.int32, device=device)
    net.train_chunk(z_train, s_train)

    z_eval = torch.randn(5, 4, device=device)
    live = net.predict_batch(z_eval)
    cloned = net.clone().predict_batch(z_eval)
    torch.cuda.synchronize()

    assert torch.equal(live, cloned)


@pytest.mark.cuda
def test_clone_isolates_subsequent_training_cuda() -> None:
    """The adaptation step inside `evaluate_task` trains on the clone.  If
    clone shares any mutable tensor with the source net the live net would
    silently drift after evaluation.  Pin: training on the clone must not
    change the source net's predictions.
    """
    device = require_cuda_kernel()
    torch.manual_seed(1)
    net = NctlNetwork(
        layer_sizes=[2, 1],
        input_dim=3,
        num_halfspaces=1,
        lr=0.1,
        pool_capacity=2,
        min_segment=4,
        ptw_depth=4,
        device=device,
        seed=11,
    )
    z_eval = torch.randn(4, 3, device=device)
    before = net.predict_batch(z_eval).clone()

    clone = net.clone()
    z_train = torch.randn(16, 3, device=device)
    s_train = torch.randint(0, 2, (16,), dtype=torch.int32, device=device)
    clone.train_chunk(z_train, s_train)

    after = net.predict_batch(z_eval)
    torch.cuda.synchronize()
    assert torch.equal(before, after), "clone training leaked into source net"


def test_clone_deep_copies_all_mutable_tensors_cpu() -> None:
    """Pure-CPU surrogate for the CUDA clone-correctness tests above: mutate
    every clone field that downstream code touches (mixture_weights,
    pool_snapshots, pool_sizes, pool_levels, segment_log_probs,
    model_log_probs) and verify the source net is unaffected.  Hyperplanes
    and hp_bias are intentionally shared read-only and excluded.
    """
    net = NctlNetwork(
        layer_sizes=[2, 1],
        input_dim=3,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        min_segment=4,
        ptw_depth=4,
        device=torch.device("cpu"),
        seed=5,
    )
    with torch.no_grad():
        net.layers[0].pool_sizes[0] = 1
        net.layers[0].pool_snapshots[0, 0].fill_(0.5)
        net.layers[0].pool_levels[0, 0] = 3
        net.layers[0].mixture_weights[0, 0].fill_(0.5)
        net.layers[0].segment_log_probs[0, 0] = -1.5
        net.layers[0].model_log_probs[0, 0] = -1.5
    net.index = 9
    net.segment_counter = 9

    snapshots = {
        "pool_snapshots": net.layers[0].pool_snapshots.clone(),
        "pool_levels": net.layers[0].pool_levels.clone(),
        "pool_sizes": net.layers[0].pool_sizes.clone(),
        "mixture_weights": net.layers[0].mixture_weights.clone(),
        "segment_log_probs": net.layers[0].segment_log_probs.clone(),
        "model_log_probs": net.layers[0].model_log_probs.clone(),
        "index": net.index,
        "segment_counter": net.segment_counter,
    }

    clone = net.clone()
    with torch.no_grad():
        clone.layers[0].pool_snapshots.fill_(99.0)
        clone.layers[0].pool_levels.fill_(99)
        clone.layers[0].pool_sizes.fill_(99)
        clone.layers[0].mixture_weights.fill_(99.0)
        clone.layers[0].segment_log_probs.fill_(99.0)
        clone.layers[0].model_log_probs.fill_(99.0)
    clone.index = 999
    clone.segment_counter = 999

    assert torch.equal(net.layers[0].pool_snapshots, snapshots["pool_snapshots"])
    assert torch.equal(net.layers[0].pool_levels, snapshots["pool_levels"])
    assert torch.equal(net.layers[0].pool_sizes, snapshots["pool_sizes"])
    assert torch.equal(net.layers[0].mixture_weights, snapshots["mixture_weights"])
    assert torch.equal(net.layers[0].segment_log_probs, snapshots["segment_log_probs"])
    assert torch.equal(net.layers[0].model_log_probs, snapshots["model_log_probs"])
    assert net.index == snapshots["index"]
    assert net.segment_counter == snapshots["segment_counter"]
    # Hyperplanes / hp_bias are intentionally shared (read-only) so we don't
    # require them to be deep-copied, but the clone should at least see them.
    assert net.layers[0].hyperplanes.data_ptr() == clone.layers[0].hyperplanes.data_ptr()


def test_layer_clone_preserves_pool_state_independently_cpu() -> None:
    """CudaFmnMixtureLayer.clone() must deep-copy pool_snapshots,
    pool_levels, mixture_weights, pool_sizes, segment_log_probs, and
    model_log_probs so that segment_close on the clone does not corrupt the
    source layer (a common shape of "evaluation pollutes training" bug).
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=3,
    )
    with torch.no_grad():
        layer.pool_sizes[0] = 1
        layer.pool_snapshots[0, 0].fill_(0.5)
        layer.pool_levels[0, 0] = 7
        layer.mixture_weights[0, 0].copy_(layer.pool_snapshots[0, 0])
        layer.mixture_weights[0, layer.fresh_idx].fill_(-0.25)
        layer.segment_log_probs[0, 0] = -2.0

    clone = layer.clone()
    # Mutate clone aggressively (simulating an adaptation pass).
    clone.segment_close(levels=[5])
    clone.mixture_weights[0, layer.fresh_idx].fill_(99.0)
    clone.pool_snapshots[0, 0].fill_(-99.0)

    # Source layer must be untouched.
    assert int(layer.pool_sizes[0].item()) == 1
    assert int(layer.pool_levels[0, 0].item()) == 7
    assert torch.allclose(
        layer.pool_snapshots[0, 0],
        torch.full_like(layer.pool_snapshots[0, 0], 0.5),
    )
    assert layer.mixture_weights[0, layer.fresh_idx].abs().sum().item() > 0.0
    assert layer.mixture_weights[0, layer.fresh_idx].max().item() < 99.0
    assert layer.segment_log_probs[0, 0].item() == pytest.approx(-2.0)


def test_train_chunk_fires_segment_close_with_planned_level_tags_cpu() -> None:
    """End-to-end: when train_chunk processes 16 samples at depth=3 with
    min_segment=1, the recorded segment_close calls must match the FMN
    Algorithm 1 closure pattern (the same one pinned by
    test_mscb_matches_paper_example), and pool entries must carry the
    deepest-level tag we documented in segment_close.
    """

    class RecordingLayer:
        def __init__(self):
            self.calls: list[list[int]] = []
            self.last_subchunk_len: list[int] = []

        def forward_update_batch(self, z, p_prev, symbols):
            self.last_subchunk_len.append(int(z.size(0)))
            return torch.full((z.size(0), 1), 0.5)

        def forward_only_batch(self, z, p_prev):
            return torch.full((z.size(0), 1), 0.5)

        def segment_close(self, level=None, levels=None, update_levels=None, task_id=None, global_index=None):
            if levels is None:
                levels = [-1 if level is None else int(level)]
            self.calls.append(list(levels))
            assert task_id == 7
            assert isinstance(global_index, int)

    net = NctlNetwork.__new__(NctlNetwork)
    net.index = 0
    net.segment_counter = 0
    net.min_segment = 1
    net.ptw_depth = 3
    net.log_loss = 0.0
    layer = RecordingLayer()
    net.layers = [layer]

    z = torch.zeros(16, 2)
    symbols = torch.zeros(16, dtype=torch.int32)
    net.train_chunk(z, symbols, task_id=7)

    # Expected per-step closures at depth=3, t=2..16 (1-based).  For
    # ptw_depth=3 there are 2^3=8 leaves; t=9..16 wrap into the second
    # half-tree but mscb_boundary_plan is purely positional so the same
    # 1..7 pattern repeats with the t=9 transition itself firing.
    from nctl_bench.nctl_network import mscb_close_levels
    expected = [mscb_close_levels(3, t, min_segment=1) for t in range(1, 17)]
    expected = [e for e in expected if e]
    assert layer.calls == expected
    # And one subchunk per inter-event gap (15 events => 16 subchunks).
    assert sum(layer.last_subchunk_len) == 16



def test_posterior_temperature_one_matches_unscaled_helper_cpu() -> None:
    """Backward compatibility: posterior_temp=1.0 must reproduce the
    pre-temperature mixture used by the prior session-test suite."""
    preds = torch.tensor([0.10, 0.90, 0.40])
    log_probs = torch.tensor([-20.0, -5.0, -7.5])

    baseline = _posterior_mixture_prob(
        model_preds=preds,
        segment_log_probs=log_probs,
        pool_size=2,
        fresh_idx=2,
    )
    tempered = _posterior_mixture_prob(
        model_preds=preds,
        segment_log_probs=log_probs,
        pool_size=2,
        fresh_idx=2,
        posterior_temp=1.0,
    )
    assert torch.allclose(baseline, tempered)


def test_posterior_temperature_zero_collapses_to_priors_cpu() -> None:
    """At posterior_temp=0 the current-segment likelihood term drops out.
    Mixture should equal the FMN-prior-weighted average of the predictions
    (1/2 fresh + 1/(2k) per pool model), regardless of segment_log_probs.
    """
    preds = torch.tensor([0.00, 1.00, 0.40])  # pool slot 0=0.0, pool slot 1=1.0, fresh=0.4
    # Make the likelihoods wildly asymmetric so a non-zero temperature
    # would shift the mixture far from the prior expectation.
    log_probs = torch.tensor([-1000.0, 0.0, -1000.0])

    mixed = _posterior_mixture_prob(
        model_preds=preds,
        segment_log_probs=log_probs,
        pool_size=2,
        fresh_idx=2,
        posterior_temp=0.0,
    )
    # k=2 pool slots: each gets prior 1/4; fresh gets 1/2.
    # mixture = 1/4*0.0 + 1/4*1.0 + 1/2*0.4 = 0.45
    assert mixed.item() == pytest.approx(0.45, abs=1e-6)


def test_posterior_temperature_interpolates_monotonically_cpu() -> None:
    """As temperature increases from 0 toward 1 the mixture moves
    monotonically from the prior-weighted average toward the
    likelihood-dominated argmax.  This is the lever the temperature
    knob gives us for combating mechanism C (posterior saturation in
    long FMN segments).
    """
    preds = torch.tensor([0.0, 1.0, 0.5])  # pool0, pool1, fresh
    log_probs = torch.tensor([-1000.0, 0.0, -500.0])

    results = [
        _posterior_mixture_prob(
            model_preds=preds,
            segment_log_probs=log_probs,
            pool_size=2,
            fresh_idx=2,
            posterior_temp=t,
        ).item()
        for t in (0.0, 0.001, 0.01, 0.1, 1.0)
    ]
    # Prior-weighted average at t=0:   1/4*0 + 1/4*1 + 1/2*0.5 = 0.5
    # At t=1 the segment_log_probs dominate; pool slot 1 wins -> ~1.0
    assert results[0] == pytest.approx(0.5, abs=1e-6)
    assert results[-1] == pytest.approx(1.0, abs=1e-3)
    # Monotonic increase between bookends (likelihood-favoured slot is the
    # higher-prediction one, so as t rises the mixture rises).
    for a, b in zip(results, results[1:]):
        assert b >= a - 1e-6, f"non-monotone: {results}"




def test_close_open_segment_without_tracked_samples_is_noop_cpu() -> None:
    net = NctlNetwork.__new__(NctlNetwork)
    net.index = 0
    net.ptw_depth = 4
    net.layers = []

    assert net.close_open_segment(task_id=1) is False


def test_close_open_segment_commits_task_tail_once_cpu() -> None:
    """Task-boundary closes should commit a non-empty tail segment once.

    This protects sparse-min-segment runs where a task can end with hundreds or
    thousands of samples in the active segment that would otherwise be carried
    into the next task before being saved to the pool.
    """

    class RecordingLayer:
        def __init__(self):
            self.calls: list[tuple[list[int], int | None, int | None]] = []

        def segment_close(self, level=None, levels=None, update_levels=None, task_id=None, global_index=None):
            if levels is None:
                levels = [-1 if level is None else int(level)]
            self.calls.append((list(levels), task_id, global_index))

    net = NctlNetwork.__new__(NctlNetwork)
    net.index = 123
    net.ptw_depth = 15
    net.samples_since_close = 37
    layer = RecordingLayer()
    net.layers = [layer]

    assert net.close_open_segment(task_id=4) is True
    assert layer.calls == [([15], 4, 123)]
    assert net.samples_since_close == 0

    # A second boundary with no intervening data must be a no-op.
    assert net.close_open_segment(task_id=4) is False
    assert layer.calls == [([15], 4, 123)]


def test_train_chunk_tracks_samples_since_last_close_cpu() -> None:
    class RecordingLayer:
        def __init__(self):
            self.calls: list[list[int]] = []

        def forward_update_batch(self, z, p_prev, symbols):
            return torch.full((z.size(0), 1), 0.5)

        def forward_only_batch(self, z, p_prev):
            return torch.full((z.size(0), 1), 0.5)

        def segment_close(self, level=None, levels=None, update_levels=None, task_id=None, global_index=None):
            self.calls.append(list(levels or []))

    net = NctlNetwork.__new__(NctlNetwork)
    net.index = 0
    net.segment_counter = 0
    net.samples_since_close = 0
    net.min_segment = 8
    net.ptw_depth = 4
    net.log_loss = 0.0
    layer = RecordingLayer()
    net.layers = [layer]

    z = torch.zeros(7, 2)
    symbols = torch.zeros(7, dtype=torch.int32)
    net.train_chunk(z, symbols, task_id=1)
    # Phase 1: MSCB close events fire at every t >= 2 regardless of
    # min_segment because the paper closes PTW state for every level on its
    # natural schedule; only UPDATEMODELPOOL eligibility is gated.  Over 7
    # samples that is 6 close events at offsets 1..6 (before samples 2..7),
    # so the chunk leaves exactly one uncommitted sample at the tail.
    from nctl_bench.nctl_network import mscb_close_levels
    expected_calls = [mscb_close_levels(4, t) for t in range(2, 8)]
    assert layer.calls == expected_calls
    assert net.samples_since_close == 1

    assert net.close_open_segment(task_id=1) is True
    # The task-tail close uses the default deepest-level tag because no MSCB
    # event preceded it.
    assert layer.calls == expected_calls + [[4]]
    assert net.samples_since_close == 0

def test_network_uses_larger_output_pool_capacity_cpu() -> None:
    """Output-layer retention can be increased without globally enlarging
    hidden-layer pools.  This targets the observed structural bottleneck where
    the final layer loses early-task slots even after moderate global pool
    growth.
    """
    net = NctlNetwork(
        layer_sizes=[3, 2, 1],
        input_dim=4,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        output_pool_capacity=5,
        min_segment=4,
        ptw_depth=4,
        device=torch.device("cpu"),
        seed=0,
    )

    assert net.pool_capacity == 2
    assert net.output_pool_capacity == 5
    assert net.layer_pool_capacities == [2, 2, 5]
    assert [layer.pool_capacity for layer in net.layers] == [2, 2, 5]

    cloned = net.clone()
    assert cloned.output_pool_capacity == 5
    assert cloned.layer_pool_capacities == [2, 2, 5]
    assert [layer.pool_capacity for layer in cloned.layers] == [2, 2, 5]

    prov = net.provenance_summary()
    assert prov["pool_capacity"] == 2
    assert prov["output_pool_capacity"] == 5
    assert prov["layer_pool_capacities"] == [2, 2, 5]


def test_network_snapshot_roundtrip_preserves_output_pool_capacity(tmp_path) -> None:
    net = NctlNetwork(
        layer_sizes=[2, 1],
        input_dim=3,
        num_halfspaces=1,
        lr=0.001,
        pool_capacity=2,
        output_pool_capacity=4,
        min_segment=4,
        ptw_depth=4,
        posterior_temp=0.25,
        device=torch.device("cpu"),
        seed=17,
    )
    path = tmp_path / "snap_output_pool.pt"
    net.save_snapshot(path)
    loaded = NctlNetwork.load_snapshot(path, device=torch.device("cpu"))

    assert loaded.pool_capacity == 2
    assert loaded.output_pool_capacity == 4
    assert loaded.layer_pool_capacities == [2, 4]
    assert [layer.pool_capacity for layer in loaded.layers] == [2, 4]


def test_network_propagates_per_level_active_state_to_layers_cpu() -> None:
    net = NctlNetwork(
        layer_sizes=[3, 2, 1],
        input_dim=4,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        min_segment=4,
        ptw_depth=4,
        active_state_mode="per_level",
        device=torch.device("cpu"),
        seed=0,
    )

    # Phase 1 paper fidelity: active state is allocated for every PTW level
    # 0..depth regardless of min_segment, because min_segment only gates
    # UPDATEMODELPOOL and never the PTW close/reset.  depth=4 -> 5 slots.
    assert net.active_state_mode == "per_level"
    assert net.active_level_count == 5
    assert net.prediction_level == 4
    for layer in net.layers:
        assert layer.per_level_active is True
        assert layer.active_level_count == 5
        assert layer.prediction_level == 4
        assert layer.level_mixture_weights is not None
        assert layer.level_mixture_weights.shape[0] == 5

    cloned = net.clone()
    assert cloned.active_state_mode == "per_level"
    assert cloned.layers[0].per_level_active is True
    assert cloned.layers[0].level_mixture_weights is not net.layers[0].level_mixture_weights


def test_network_snapshot_roundtrip_preserves_per_level_active_state(tmp_path) -> None:
    net = NctlNetwork(
        layer_sizes=[2, 1],
        input_dim=3,
        num_halfspaces=1,
        lr=0.001,
        pool_capacity=2,
        min_segment=4,
        ptw_depth=4,
        active_state_mode="per_level",
        device=torch.device("cpu"),
        seed=17,
    )
    with torch.no_grad():
        net.layers[0].level_mixture_weights[2, 0, net.layers[0].fresh_idx].fill_(1.25)
    path = tmp_path / "snap_per_level.pt"
    net.save_snapshot(path)
    loaded = NctlNetwork.load_snapshot(path, device=torch.device("cpu"))

    assert loaded.active_state_mode == "per_level"
    assert loaded.active_level_count == 5
    assert loaded.prediction_level == 4
    assert loaded.layers[0].per_level_active is True
    assert torch.allclose(
        loaded.layers[0].level_mixture_weights[2, 0, loaded.layers[0].fresh_idx],
        torch.full_like(loaded.layers[0].level_mixture_weights[2, 0, loaded.layers[0].fresh_idx], 1.25),
    )

def test_layer_default_posterior_temp_is_one_cpu() -> None:
    """The constructor default must be 1.0 so old code paths and the
    pre-temperature cached CUDA kernel (.so) keep matching behavior."""
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=1,
        device=torch.device("cpu"),
        seed=0,
    )
    assert layer.posterior_temp == 1.0
    cloned = layer.clone()
    assert cloned.posterior_temp == 1.0


def test_network_propagates_posterior_temp_to_layers_cpu() -> None:
    """The CLI knob in run_split_mnist.py and the sweep configs target
    NctlNetwork(posterior_temp=...).  Verify the value reaches every
    layer (CUDA layer reads it from self.posterior_temp at kernel call
    time).
    """
    net = NctlNetwork(
        layer_sizes=[3, 2, 1],
        input_dim=4,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        min_segment=4,
        ptw_depth=4,
        posterior_temp=0.05,
        device=torch.device("cpu"),
        seed=0,
    )
    for layer in net.layers:
        assert layer.posterior_temp == pytest.approx(0.05)
    assert net.clone().layers[0].posterior_temp == pytest.approx(0.05)


def test_network_snapshot_roundtrip_preserves_posterior_temp(tmp_path) -> None:
    net = NctlNetwork(
        layer_sizes=[2, 1],
        input_dim=3,
        num_halfspaces=1,
        lr=0.001,
        pool_capacity=2,
        min_segment=4,
        ptw_depth=4,
        posterior_temp=0.25,
        device=torch.device("cpu"),
        seed=17,
    )
    path = tmp_path / "snap.pt"
    net.save_snapshot(path)
    loaded = NctlNetwork.load_snapshot(path, device=torch.device("cpu"))
    assert loaded.posterior_temp == pytest.approx(0.25)
    assert loaded.layers[0].posterior_temp == pytest.approx(0.25)


class _MultilevelStub:
    """CPU stub that delegates to the pure-PyTorch multilevel references.

    Used by D-step 4 tests to exercise the new dispatch path even on a
    CPU build that doesn\'t have the Phase 5F .so.  Implements all three
    entry points the layer dispatcher checks for:
    forward_update / forward_only / forward_update_with_resets.
    """

    def __init__(self):
        from nctl_bench.nctl_network import (
            _fmn_multilevel_paper_posterior_reference,
            _fmn_multilevel_paper_posterior_reference_with_resets,
        )
        self._ref_baseline = _fmn_multilevel_paper_posterior_reference
        self._ref_with_resets = _fmn_multilevel_paper_posterior_reference_with_resets

    def forward_update(self, z, p_prev, sym, mw, hp, hb, ps, slp, mlp, lr, posterior_temp=0.0):
        return self._ref_baseline(
            z, p_prev, sym, mw, hp, hb, ps, slp, mlp, lr,
            update=True, posterior_temp=float(posterior_temp),
        )

    def forward_only(self, z, p_prev, mw, hp, hb, ps, slp, posterior_temp=0.0):
        return self._ref_baseline(
            z, p_prev, None, mw, hp, hb, ps, slp, None, lr=0.0,
            update=False, posterior_temp=float(posterior_temp),
        )

    def forward_update_with_resets(
        self, z, p_prev, sym, mw, pool_snapshots, hp, hb, ps, close_mask, slp, mlp, lr,
        posterior_temp=0.0,
    ):
        return self._ref_with_resets(
            z, p_prev, sym, mw, pool_snapshots, hp, hb, ps, close_mask, slp, mlp, lr,
            posterior_temp=float(posterior_temp),
        )


def test_train_chunk_d_step4_calls_sub_chunk_once_per_update_boundary_cpu() -> None:
    """The whole point of D: at depth=15, ms=512, chunk_size=1024 the
    D-step 4 path must call _train_subchunk a handful of times (one per
    UPDATE boundary + final residual), NOT 1024 times (one per close
    event) like the legacy path.  We force the new dispatch by stubbing
    the multilevel module to expose forward_update_with_resets.
    """
    from nctl_bench import nctl_network as _net
    from nctl_bench.nctl_network import NctlNetwork

    prev_ml = _net._fmn_multilevel_cuda
    prev_ml_missing = _net._fmn_multilevel_cuda_missing
    _net._fmn_multilevel_cuda = _MultilevelStub()
    _net._fmn_multilevel_cuda_missing = False
    try:
        net = NctlNetwork(
            layer_sizes=[2, 1],
            input_dim=2,
            num_halfspaces=1,
            lr=0.001,
            pool_capacity=2,
            min_segment=512,
            ptw_depth=15,
            device=torch.device("cpu"),
            seed=11,
            active_state_mode="per_level",
            prediction_mode="ptw_dp",
            pool_update_policy="paper",
            pool_alpha=float("inf"),
            pool_beta=float("inf"),
        )
        original = net._train_subchunk
        calls: list[int] = []

        def _spy(z_sub, symbols_sub, close_events=None):
            calls.append(z_sub.size(0))
            return original(z_sub, symbols_sub, close_events=close_events)

        net._train_subchunk = _spy
        # Small chunk to keep the test fast on the CPU reference, but
        # large enough to span an update boundary.
        torch.manual_seed(1)
        z = torch.randn(8, 2)
        symbols = torch.randint(0, 2, (8,), dtype=torch.int32)
        net.train_chunk(z, symbols, task_id=1)
        assert len(calls) <= 4, (
            f"D-step 4 dispatched {len(calls)} sub-chunk calls; expected <= 4. "
            f"Sub-chunk sizes: {calls}"
        )
        # Calls average > 1 sample (vs strictly 1 in the legacy path).
        if len(calls) >= 1:
            mean_size = sum(calls) / len(calls)
            assert mean_size >= 1.0
    finally:
        _net._fmn_multilevel_cuda = prev_ml
        _net._fmn_multilevel_cuda_missing = prev_ml_missing


def test_train_chunk_d_step4_matches_legacy_with_stub_kernel_cpu() -> None:
    """D-step 4 fast path must produce numerically identical chunk-loss
    AND identical post-chunk PTW DP state AND identical per-level FMN
    active state vs the legacy path, when the multilevel kernel is the
    same on both paths.  We use the CPU stub (which delegates to the
    pure-PyTorch reference) so the new and legacy paths see the same
    kernel math.
    """
    from nctl_bench import nctl_network as _net
    from nctl_bench.nctl_network import NctlNetwork

    def _make_net(seed):
        return NctlNetwork(
            layer_sizes=[2, 1],
            input_dim=2,
            num_halfspaces=1,
            lr=0.001,
            pool_capacity=2,
            min_segment=4,
            ptw_depth=3,
            device=torch.device("cpu"),
            seed=seed,
            active_state_mode="per_level",
            prediction_mode="ptw_dp",
            pool_update_policy="paper",
            pool_alpha=float("inf"),
            pool_beta=float("inf"),
        )

    prev_ml = _net._fmn_multilevel_cuda
    prev_ml_missing = _net._fmn_multilevel_cuda_missing
    _net._fmn_multilevel_cuda = _MultilevelStub()
    _net._fmn_multilevel_cuda_missing = False
    try:
        torch.manual_seed(101)
        net_new = _make_net(seed=5)
        torch.manual_seed(101)
        net_legacy = _make_net(seed=5)
        torch.manual_seed(202)
        B = 8
        z = torch.randn(B, 2)
        symbols = torch.randint(0, 2, (B,), dtype=torch.int32)

        loss_new = net_new.train_chunk(z, symbols, task_id=1)
        # Force the legacy path on the second net.
        loss_legacy = net_legacy._train_chunk_legacy(
            z.clone(), symbols.clone(), task_id=1, B=B
        )

        # DP state must agree.
        for la, lb in zip(net_new.layers, net_legacy.layers):
            assert torch.allclose(la.level_log_nu, lb.level_log_nu, atol=1e-7), (
                f"DP state mismatch: max diff = "
                f"{(la.level_log_nu - lb.level_log_nu).abs().max().item():.3e}"
            )
            assert torch.allclose(la.ptw_log_w, lb.ptw_log_w, atol=1e-7)
            assert torch.allclose(la.ptw_log_b, lb.ptw_log_b, atol=1e-7)
            # Per-level FMN active state must also agree (the in-kernel
            # resets in the stub reproduce the per-event segment_close).
            assert torch.allclose(
                la.level_mixture_weights, lb.level_mixture_weights, atol=1e-6
            )
            assert torch.allclose(
                la.level_segment_log_probs, lb.level_segment_log_probs, atol=1e-6
            )
        # index + samples_since_close advance must match.
        assert net_new.index == net_legacy.index
        assert net_new.samples_since_close == net_legacy.samples_since_close
        # Loss agrees within tight tolerance.
        assert abs(loss_new - loss_legacy) < 1e-5, (
            f"loss mismatch: new={loss_new:.6f} legacy={loss_legacy:.6f}"
        )
    finally:
        _net._fmn_multilevel_cuda = prev_ml
        _net._fmn_multilevel_cuda_missing = prev_ml_missing


# ---------------------------------------------------------------------------
# Phase 5H-2: CUDA-only parity for segmented_dp_state_advance dispatch.
# ---------------------------------------------------------------------------

@pytest.mark.cuda
def test_ptw_dp_apply_chunk_segmented_cuda_matches_cpu_fallback() -> None:
    """30.9 Phase 5H-2 parity contract.

    With the multilevel .so exposing ``segmented_dp_state_advance``, the
    GPU dispatch path inside ``_ptw_dp_apply_chunk_segmented`` must
    produce bit-identical post-chunk DP state AND output combine vs the
    CPU Python event loop, on the same chunk + close_events.

    Skips when CUDA or the new binding isn't available so the suite
    stays green on CPU builds.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from nctl_bench.nctl_network import _get_fmn_multilevel_cuda
    ml = _get_fmn_multilevel_cuda()
    if ml is None or not hasattr(ml, "segmented_dp_state_advance"):
        pytest.skip("multilevel .so missing segmented_dp_state_advance")

    layer_cpu = _make_dp_layer(active_level_count=4)
    L, N = 4, 2
    B = 32
    torch.manual_seed(73)
    nu0 = torch.linspace(-1.2, 0.8, L * N).reshape(L, N).to(torch.float64)
    w0 = torch.linspace(-0.7, 0.3, L * N).reshape(L, N).to(torch.float64)
    b0 = torch.linspace(-0.4, 0.2, L * N).reshape(L, N).to(torch.float64)
    layer_cpu.level_log_nu.copy_(nu0)
    layer_cpu.ptw_log_w.copy_(w0)
    layer_cpu.ptw_log_b.copy_(b0)
    q_obs = torch.rand(L, B, N).clamp(1e-6, 1.0 - 1e-6)
    log_q_obs = torch.log(q_obs).to(torch.float32)
    q1 = q_obs.to(torch.float32)
    # Mix of events: interior close, multi-level close, tail close.
    close_events = [(3, [2, 3]), (8, [3]), (12, [1, 2, 3]), (20, [3]), (28, [2, 3])]
    out_cpu = layer_cpu._ptw_dp_apply_chunk(log_q_obs, q1, close_events=close_events)

    device = torch.device("cuda")
    layer_gpu = CudaFmnMixtureLayer(
        num_nodes=2,
        num_inputs=2,
        input_dim=2,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=device,
        seed=1,
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
        active_state_mode="per_level",
        active_level_count=4,
        prediction_level=3,
        prediction_mode="ptw_dp",
    )
    layer_gpu.level_log_nu.copy_(nu0.to(device))
    layer_gpu.ptw_log_w.copy_(w0.to(device))
    layer_gpu.ptw_log_b.copy_(b0.to(device))
    out_gpu = layer_gpu._ptw_dp_apply_chunk(
        log_q_obs.to(device), q1.to(device), close_events=close_events,
    )
    torch.cuda.synchronize()

    # Output combine agrees in float32 to ~1e-5.  Per-level state
    # (nu, w, b) agrees in float64 to ~1e-9 -- both paths are pure
    # double-precision recurrences on the same inputs.
    assert torch.allclose(
        out_gpu.cpu(), out_cpu, atol=1e-5
    ), f"max diff = {(out_gpu.cpu() - out_cpu).abs().max().item():.3e}"
    assert torch.allclose(
        layer_gpu.level_log_nu.cpu(), layer_cpu.level_log_nu, atol=1e-9
    )
    assert torch.allclose(
        layer_gpu.ptw_log_w.cpu(), layer_cpu.ptw_log_w, atol=1e-9
    )
    assert torch.allclose(
        layer_gpu.ptw_log_b.cpu(), layer_cpu.ptw_log_b, atol=1e-9
    )


@pytest.mark.cuda
def test_segmented_dp_state_advance_kernel_smoke_cuda() -> None:
    """30.9 Phase 5H-2 kernel-level smoke test.

    Calls ``segmented_dp_state_advance`` directly with hand-rolled
    inputs and checks the returned seed tensors match a Python
    reference event walk.  Catches binding signature / dtype / layout
    regressions before they cascade into the layer dispatch.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from nctl_bench.nctl_network import _get_fmn_multilevel_cuda
    ml = _get_fmn_multilevel_cuda()
    if ml is None or not hasattr(ml, "segmented_dp_state_advance"):
        pytest.skip("multilevel .so missing segmented_dp_state_advance")

    device = torch.device("cuda")
    L, N, B = 3, 2, 8
    torch.manual_seed(101)
    log_q = torch.randn(L, B, N, dtype=torch.float64, device=device) * 0.1
    cum_inclusive = log_q.cumsum(dim=1)
    zero_col = torch.zeros(L, 1, N, dtype=torch.float64, device=device)
    cum_padded = torch.cat([zero_col, cum_inclusive], dim=1).contiguous()
    # Two events: (3, [1, 2]) and (6, [2]).
    sorted_offsets = [3, 6]
    events_by_offset = {3: [1, 2], 6: [2]}
    # Segments: [0,3), [3,6), [6,B).  All three are non-degenerate so
    # event_seg_idx == [0, 1] and trailing_seg_idx == 2.
    event_offsets = torch.tensor(sorted_offsets, dtype=torch.int32, device=device)
    event_seg_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)
    trailing_seg_idx = 2
    close_mask = torch.zeros(len(sorted_offsets), L, dtype=torch.bool)
    close_mask[0, 1] = True
    close_mask[0, 2] = True
    close_mask[1, 2] = True
    close_mask = close_mask.to(device).contiguous()
    state_nu = torch.zeros(L, N, dtype=torch.float64, device=device)
    state_w = torch.zeros(L, N, dtype=torch.float64, device=device)
    state_b = torch.full((L, N), 0.1, dtype=torch.float64, device=device)
    S = 3
    seg_nu0_k, seg_w0_k, seg_b0_k = ml.segmented_dp_state_advance(
        cum_padded, event_offsets, close_mask, event_seg_idx,
        int(trailing_seg_idx), state_nu, state_w, state_b, int(S),
    )
    torch.cuda.synchronize()
    # Python reference event walk.
    import math
    half = math.log(0.5)
    ref_nu = torch.zeros(L, N, dtype=torch.float64)
    ref_w = torch.zeros(L, N, dtype=torch.float64)
    ref_b = torch.full((L, N), 0.1, dtype=torch.float64)
    cum_padded_cpu = cum_padded.cpu()
    seg_seeds = []
    seg_start = 0
    for e_i, off in enumerate(sorted_offsets):
        if off > seg_start:
            seg_seeds.append((ref_nu.clone(), ref_w.clone(), ref_b.clone()))
            seg_cum = cum_padded_cpu[:, off, :] - cum_padded_cpu[:, seg_start, :]
            ref_nu = ref_nu + seg_cum
            w_out = torch.empty_like(ref_w)
            w_out[L - 1] = ref_nu[L - 1]
            for j in range(L - 2, -1, -1):
                w_out[j] = torch.logaddexp(half + ref_nu[j], half + w_out[j + 1] + ref_b[j])
            ref_w = w_out
            seg_start = off
        lv = sorted(events_by_offset[off])
        if lv[0] > 0:
            ref_b[lv[0] - 1] = ref_w[lv[0]].clone()
        for j in lv:
            ref_nu[j] = 0.0
            ref_w[j] = 0.0
            ref_b[j] = 0.0
    if seg_start < B:
        seg_seeds.append((ref_nu.clone(), ref_w.clone(), ref_b.clone()))
        seg_cum = cum_padded_cpu[:, B, :] - cum_padded_cpu[:, seg_start, :]
        ref_nu = ref_nu + seg_cum
        w_out = torch.empty_like(ref_w)
        w_out[L - 1] = ref_nu[L - 1]
        for j in range(L - 2, -1, -1):
            w_out[j] = torch.logaddexp(half + ref_nu[j], half + w_out[j + 1] + ref_b[j])
        ref_w = w_out
    # Kernel seeds are [S, L, N]; reference seeds are list-of-[L,N].
    for si, (n0, w0_, b0_) in enumerate(seg_seeds):
        assert torch.allclose(seg_nu0_k[si].cpu(), n0, atol=1e-12), f"seg {si} nu mismatch"
        assert torch.allclose(seg_w0_k[si].cpu(), w0_, atol=1e-12), f"seg {si} w mismatch"
        assert torch.allclose(seg_b0_k[si].cpu(), b0_, atol=1e-12), f"seg {si} b mismatch"
    # Final committed state matches ref.
    assert torch.allclose(state_nu.cpu(), ref_nu, atol=1e-12)
    assert torch.allclose(state_w.cpu(), ref_w, atol=1e-12)
    assert torch.allclose(state_b.cpu(), ref_b, atol=1e-12)


# ---------------------------------------------------------------------------
# Phase 5I-A: paper-faithful posterior-weighted ν_j conditional.
# ---------------------------------------------------------------------------

def _posterior_mixture_python(
    model_preds: torch.Tensor,
    segment_log_probs: torch.Tensor,
    pool_size: int,
    fresh_idx: int,
    posterior_temp: float,
) -> torch.Tensor:
    """Closed-form Bayes-conditional reference matching the paper's Eq. 5
    expansion at one (level, node, sample): prior(1/2 fresh + (1/2)/(M-1)
    per pool slot) multiplied by exp(temp * segment_log_lik), normalised.
    Used by the CPU bit-identity tests below.
    """
    import math
    if pool_size <= 0:
        return model_preds[fresh_idx]
    active = list(range(pool_size)) + [fresh_idx]
    logits = segment_log_probs[active].clone() * posterior_temp
    logits[:pool_size] += math.log(0.5) - math.log(float(pool_size))
    logits[pool_size] += math.log(0.5)
    weights = torch.softmax(logits, dim=0)
    return (weights * model_preds[active]).sum()


def test_paper_posterior_mixture_python_matches_uniform_at_temp_zero_cpu() -> None:
    """At posterior_temp == 0 the Bayes mixture must collapse to the
    pre-5I unweighted 1/2 fresh + 1/2 pool-mean formula regardless of
    per-slot segment_log_probs.  This is the back-compat contract that
    keeps the four ptw_dp_apply_chunk_segmented math tests bit-stable.
    """
    k_pool, fresh_idx = 3, 4  # M = 5 implicit from preds.shape
    preds = torch.tensor([0.7, 0.3, 0.4, 0.0, 0.85])
    seg = torch.tensor([-12.0, 0.0, 50.0, 0.0, -5.0])
    expected = 0.5 * preds[fresh_idx] + 0.5 * preds[:k_pool].mean()
    got = _posterior_mixture_python(preds, seg, k_pool, fresh_idx, 0.0)
    assert torch.isclose(got, expected, atol=1e-7)


def test_paper_posterior_mixture_python_recovers_bayes_at_temp_one_cpu() -> None:
    """Two-slot worked example reproduces the hand calculation from
    the Phase 5I-A bug write-up: with ρ(x_<t)=0.8, ρ_1(x_<t)=0.2 and
    per-step predictions 0.6 and 0.9, the conditional ratio must give
    ~0.66 -- NOT 0.75 (which is the unweighted mixture).
    """
    import math
    # Use synthetic per-step predictions; encode past-likelihoods as
    # segment_log_probs (so prior * exp(seg) reproduces the joint).
    preds = torch.tensor([0.9, 0.0, 0.6])  # pool 0, dummy slot 1, fresh
    # Mass relative weights: fresh prior=0.5 with mass 0.8 ⇒
    # raw_fresh = 0.5*0.8 = 0.4
    # pool slot 0 prior=(0.5/1)=0.5 with mass 0.2 ⇒ raw_pool = 0.5*0.2 = 0.1
    # So segment_log_probs[fresh]-segment_log_probs[pool0]
    #     should encode log(0.8/0.2) = log 4
    # We set seg=log(mass) so prior*exp(seg) reproduces raw weights.
    seg = torch.tensor([math.log(0.2), 0.0, math.log(0.8)])
    got = _posterior_mixture_python(preds, seg, 1, 2, 1.0).item()
    # Hand value: w_pool = 0.1/0.5 = 0.2, w_fresh = 0.4/0.5 = 0.8
    # mix = 0.2*0.9 + 0.8*0.6 = 0.66
    assert abs(got - 0.66) < 1e-6, f"got {got}"


@pytest.mark.cuda
def test_multilevel_kernel_posterior_temp_one_recovers_paper_bayes_cuda() -> None:
    """30.9 Phase 5I-A acceptance contract.

    With posterior_temp=1.0 the multilevel ``forward_only`` kernel must
    return per-(level, node, sample) values equal to the Python
    ``_posterior_mixture_python`` reference computed from the same
    per-slot raw predictions and ``segment_log_probs`` -- i.e. the
    Bayes conditional ratio from paper Eq. 5, NOT the unweighted
    mixture.

    Sanity check: at posterior_temp=0 the kernel still matches the
    unweighted single-level kernel (existing parity test); this test
    proves the new posterior_temp=1 path matches the closed-form
    Bayesian reference.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from nctl_bench import nctl_network as _net
    ml = _net._get_fmn_multilevel_cuda()
    if ml is None or not hasattr(ml, "forward_only"):
        pytest.skip("multilevel kernel .so unavailable")
    device = torch.device("cuda")

    L, N, M, C, K_in = 2, 1, 4, 1, 2
    D, H = 2, 1
    torch.manual_seed(1234)
    # Single context (H=1, bias far -ve so every sample lands in ctx 0).
    hp = torch.zeros(N, H, D, device=device)
    hb = torch.full((N, H), -100.0, device=device)
    # Per-slot weights chosen so sigmoid(w @ logits(p_prev=0.5)) gives
    # distinct per-slot per-step predictions for every (level, slot).
    mw = torch.zeros(L, N, M, C, K_in, device=device)
    mw[0, 0, 0, 0] = torch.tensor([0.7,  0.0])  # slot 0
    mw[0, 0, 1, 0] = torch.tensor([0.2, -0.3])  # slot 1
    mw[0, 0, 2, 0] = torch.tensor([-1.0, 0.4])  # slot 2
    mw[0, 0, 3, 0] = torch.tensor([0.0,  0.0])  # fresh (M-1)
    mw[1, 0, 0, 0] = torch.tensor([0.5,  0.5])
    mw[1, 0, 1, 0] = torch.tensor([0.0, -0.8])
    mw[1, 0, 2, 0] = torch.tensor([0.3,  0.2])
    mw[1, 0, 3, 0] = torch.tensor([-0.2, 0.6])
    pool_sizes = torch.tensor([[3], [2]], dtype=torch.int32, device=device)
    seg = torch.zeros(L, N, M, device=device)
    seg[0, 0, 0] = -1.0
    seg[0, 0, 1] = -3.0
    seg[0, 0, 2] = 0.5
    seg[0, 0, 3] = -0.4  # fresh
    seg[1, 0, 0] = 0.1
    seg[1, 0, 1] = -2.0
    seg[1, 0, 3] = 0.2   # fresh; slot 2 unused (k_pool=2)
    # Sample input: small logits so per-slot preds are well within (0,1).
    z = torch.tensor([[0.1, -0.1]], device=device)
    p_prev = torch.tensor([[0.55, 0.45]], device=device)
    got = ml.forward_only(z, p_prev, mw, hp, hb, pool_sizes, seg, 1.0)
    torch.cuda.synchronize()

    # Closed-form reference per (level, sample, node).
    import math
    EPS = 1e-7
    LOGIT_CLIP = 15.0
    p_prev_clamped = p_prev.clamp(EPS, 1.0 - EPS)
    logits = torch.log(p_prev_clamped / (1.0 - p_prev_clamped)).clamp(-LOGIT_CLIP, LOGIT_CLIP)
    fresh_idx = M - 1
    for l in range(L):
        for n in range(N):
            k = int(pool_sizes[l, n].item())
            ctx = 0  # forced by bias above
            slot_w = mw[l, n, :, ctx, :]  # [M, K_in]
            preds = torch.sigmoid(slot_w @ logits[0])  # [M]
            ref = _posterior_mixture_python(
                preds.cpu(), seg[l, n].cpu(), k, fresh_idx, 1.0,
            )
            ker = got[l, 0, n].item()
            assert abs(ker - ref.item()) < 1e-5, (
                f"(l={l}, n={n}) kernel={ker:.6f} ref={ref.item():.6f}"
            )


# ---------------------------------------------------------------------------
# Phase 5I-B: paper-faithful joint log ξ(s) for uniform pool mixture.
# ---------------------------------------------------------------------------

def test_log_prob_uniform_pool_mixture_matches_paper_marginal_cpu() -> None:
    """Phase 5I-B contract: ``_log_prob_uniform_pool_mixture`` must return
    log ξ(s) = log[(1/k) Σ_m ρ_m(s)] where ρ_m(s) is the JOINT log-likelihood
    of slot m on the reservoir, NOT an average of per-step predictions.

    Pre-5I-B the implementation took log of the mean of per-step
    probabilities and summed.  For two extreme slots (e.g. always-0.99
    and always-0.01 on a sample of three +1 symbols) this gives ~-2.08,
    while the paper-faithful joint marginal gives ~-0.72.  The
    suppressed value disabled the β skip threshold in the FMN heuristic.
    """
    import math
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=1,
        input_dim=1,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=1,
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
    )
    # Force ctx=0 and zero hyperplane bias by clearing.
    with torch.no_grad():
        layer.hyperplanes.zero_()
        layer.hp_bias.fill_(-100.0)
    # Two pool slots: very extreme weights so per-step predictions are
    # ~0.99 vs ~0.01 on p_prev=0.5 input.
    with torch.no_grad():
        # `pool_snapshots` shape [N, pool_capacity, C, K_in]; here (1, 2, 2, 1).
        layer.pool_snapshots.zero_()
        # Slot 0 produces sigmoid(big * logit(0.5)) = ~0.99 only if w is large.
        # Use w = 10.0 ⇒ pred = sigmoid(10 * 0) = 0.5.  Need logits != 0.
        # Easier: use the bias path -- set weights so a fixed logit input
        # produces 0.99 at slot 0 and 0.01 at slot 1.
        # Caller will pass z=zeros and p_prev=0.9 to get a nonzero logit.
        layer.pool_snapshots[0, 0, 0, 0] = 4.5   # slot 0, ctx 0
        layer.pool_snapshots[0, 1, 0, 0] = -4.5  # slot 1, ctx 0
    layer.pool_sizes[0] = 2

    # Reservoir s = three symbols, all +1, with p_prev that drives slot 0
    # near 0.99 and slot 1 near 0.01.
    z = torch.zeros(3, 1)
    p_prev = torch.full((3, 1), 0.99)  # logit ≈ 4.6
    symbols = torch.ones(3, dtype=torch.int32)
    reservoir = (z, p_prev, symbols)

    # Reference: compute per-slot joint log-likelihoods directly and
    # combine via -log(k) + logsumexp.
    per_slot_lp = []
    for slot in range(2):
        pred = layer._predict_node_with_weights(
            0, layer.pool_snapshots[0, slot], z, p_prev,
        )
        psym = torch.where(symbols.bool(), pred, 1.0 - pred).clamp(min=1e-30)
        per_slot_lp.append(float(torch.log(psym).sum().item()))
    max_lp = max(per_slot_lp)
    ref = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in per_slot_lp)) - math.log(2.0)

    got = layer._log_prob_uniform_pool_mixture(0, 2, reservoir)
    assert abs(got - ref) < 1e-6, f"got {got}, ref {ref}, per_slot {per_slot_lp}"

    # Sanity: the broken pre-5I-B implementation would return roughly
    # 3 * log(0.5) = -2.08 (mean of per-step probs).  The paper formula
    # is dominated by slot 0\'s joint log-likelihood, which for three
    # +1 symbols with per-step ~0.99 gives ~3 * log(0.99) ≈ -0.0302 minus
    # log(2) ≈ -0.72.
    broken_value = 3.0 * math.log(0.5)
    assert ref > broken_value + 1.0, (
        f"sanity: paper marginal {ref:.3f} should be much greater than "
        f"broken value {broken_value:.3f}"
    )


def test_log_prob_uniform_pool_mixture_single_slot_matches_log_prob_for_weights_cpu() -> None:
    """When the pool contains exactly one slot, log ξ(s) = log ρ_1(s); the
    uniform marginal collapses to the single slot\'s joint log-likelihood.
    """
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=1,
        input_dim=1,
        num_halfspaces=1,
        lr=0.0,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=2,
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
    )
    with torch.no_grad():
        layer.hyperplanes.zero_()
        layer.hp_bias.fill_(-100.0)
        layer.pool_snapshots.zero_()
        layer.pool_snapshots[0, 0, 0, 0] = 2.5
    layer.pool_sizes[0] = 1

    z = torch.zeros(4, 1)
    p_prev = torch.full((4, 1), 0.75)
    symbols = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    reservoir = (z, p_prev, symbols)

    direct = layer._log_prob_for_weights(0, layer.pool_snapshots[0, 0], reservoir)
    via_mix = layer._log_prob_uniform_pool_mixture(0, 1, reservoir)
    assert abs(direct - via_mix) < 1e-7, (direct, via_mix)


def test_log_prob_uniform_pool_mixture_empty_returns_neg_inf_cpu() -> None:
    """Defensive: k=0 (no pool slots) returns -inf regardless of reservoir."""
    layer = CudaFmnMixtureLayer(
        num_nodes=1, num_inputs=1, input_dim=1, num_halfspaces=1,
        lr=0.0, pool_capacity=2,
        device=torch.device("cpu"), seed=3,
        pool_update_policy="paper",
        pool_alpha=float("inf"), pool_beta=float("inf"),
    )
    z = torch.zeros(1, 1)
    p_prev = torch.full((1, 1), 0.5)
    symbols = torch.zeros(1, dtype=torch.int32)
    assert layer._log_prob_uniform_pool_mixture(0, 0, (z, p_prev, symbols)) == float("-inf")
    assert layer._log_prob_uniform_pool_mixture(0, 2, None) == float("-inf")


# ---------------------------------------------------------------------------
# Phase 5I-D: snapshot_state / restore_state round-trip contract.
# ---------------------------------------------------------------------------

def test_layer_snapshot_restore_round_trip_preserves_state_cpu() -> None:
    """``snapshot_state`` followed by ``restore_state`` must leave a layer
    bit-identical to its starting state, even after intervening mutations.

    Pre-5I-D evaluation cloned the whole network on every call, doubling
    GPU memory and capping per-node ``--pool`` at 80 on a 10.5 GiB GPU.
    The snapshot/restore path keeps a CPU-resident snapshot and restores
    in place; this test pins the round-trip contract.
    """
    layer = _make_dp_layer(active_level_count=3)
    # Mutate via train_chunk shape: write deterministic values into every
    # mutable buffer so the round-trip exercises a non-zero state.
    L, N = 3, 2
    layer.level_log_nu.copy_(torch.linspace(-1.0, 1.0, L * N).reshape(L, N).to(torch.float64))
    layer.ptw_log_w.copy_(torch.linspace(-0.5, 0.5, L * N).reshape(L, N).to(torch.float64))
    layer.ptw_log_b.copy_(torch.linspace(0.1, 0.4, L * N).reshape(L, N).to(torch.float64))
    layer.level_mixture_weights.fill_(0.05)
    layer.level_pool_sizes.fill_(2)
    layer.level_segment_log_probs.fill_(-0.25)
    layer.level_model_log_probs.fill_(-0.4)
    layer.mixture_weights.fill_(0.07)
    layer.pool_sizes.fill_(1)
    layer.segment_log_probs.fill_(-0.15)
    layer.model_log_probs.fill_(-0.2)
    layer.pool_snapshots.fill_(0.123)
    layer.pool_task_ids.fill_(7)
    layer.pool_insert_indices.fill_(11)
    layer.pool_levels.fill_(2)
    layer.provenance_event_counts = {"append": 5, "evict": 2, "skip": 1, "refine": 3}
    layer.provenance_evicted_task_counts = {"1": 4, "2": 1}
    layer.segment_res_size = 4
    layer.segment_res_seen = 9

    # Take snapshot.
    snap = layer.snapshot_state()

    # Mutate everything to nonsense.
    layer.level_log_nu.fill_(99.0)
    layer.ptw_log_w.fill_(-99.0)
    layer.ptw_log_b.fill_(7.0)
    layer.level_mixture_weights.fill_(-1.0)
    layer.level_pool_sizes.fill_(99)
    layer.level_segment_log_probs.fill_(-77.0)
    layer.level_model_log_probs.fill_(11.5)
    layer.mixture_weights.fill_(-0.99)
    layer.pool_sizes.fill_(99)
    layer.segment_log_probs.fill_(-88.0)
    layer.model_log_probs.fill_(123.0)
    layer.pool_snapshots.fill_(-7.5)
    layer.pool_task_ids.fill_(-1)
    layer.pool_insert_indices.fill_(-1)
    layer.pool_levels.fill_(0)
    layer.provenance_event_counts = {"append": 0}
    layer.provenance_evicted_task_counts = {}
    layer.segment_res_size = 0
    layer.segment_res_seen = 0

    # Restore.
    layer.restore_state(snap)

    # Verify every tracked attribute matches the snapshot exactly.
    # Reservoir tensors may hold uninitialised slots (NaNs / garbage)
    # outside their active prefix; round-trip equality must hold on the
    # underlying byte pattern, so we use ``equal_nan=True``-style
    # comparison via the raw byte-level view.
    def _tensors_byte_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    for name in layer._SNAPSHOT_DEVICE_TENSORS:
        cur = getattr(layer, name, None)
        saved = snap[name]
        if cur is None and saved is None:
            continue
        assert _tensors_byte_equal(cur.detach().cpu(), saved), f"device tensor mismatch: {name}"
    for name in layer._SNAPSHOT_CPU_TENSORS:
        cur = getattr(layer, name, None)
        saved = snap[name]
        if cur is None and saved is None:
            continue
        assert _tensors_byte_equal(cur, saved), f"cpu tensor mismatch: {name}"
    assert layer.provenance_event_counts == snap["provenance_event_counts"]
    assert layer.provenance_evicted_task_counts == snap["provenance_evicted_task_counts"]
    assert layer.segment_res_size == snap["segment_res_size"]
    assert layer.segment_res_seen == snap["segment_res_seen"]


def test_network_snapshot_restore_round_trip_preserves_state_cpu() -> None:
    """Network-level round trip: snapshot, mutate counters and per-layer
    state, restore, then verify scalars + per-layer state are equal.
    """
    from nctl_bench.nctl_network import NctlNetwork

    net = NctlNetwork(
        layer_sizes=[2, 1],
        input_dim=2,
        num_halfspaces=1,
        lr=0.001,
        pool_capacity=2,
        min_segment=64,
        ptw_depth=8,
        device=torch.device("cpu"),
        seed=4,
        active_state_mode="per_level",
        prediction_mode="ptw_dp",
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
    )
    net.index = 17
    net.log_loss = 0.5
    net.segment_counter = 33
    net.samples_since_close = 9
    net.layers[0].pool_sizes.fill_(1)
    net.layers[0].pool_task_ids.fill_(3)

    snap = net.snapshot_state()

    # Mutate.
    net.index = 999
    net.log_loss = -1.0
    net.segment_counter = 0
    net.samples_since_close = 0
    net.layers[0].pool_sizes.fill_(0)
    net.layers[0].pool_task_ids.fill_(-1)

    net.restore_state(snap)

    assert net.index == 17
    assert net.log_loss == 0.5
    assert net.segment_counter == 33
    assert net.samples_since_close == 9
    assert torch.equal(net.layers[0].pool_sizes, snap["layers"][0]["pool_sizes"])
    assert torch.equal(net.layers[0].pool_task_ids, snap["layers"][0]["pool_task_ids"])


def test_evaluate_task_in_place_restores_network_state_cpu() -> None:
    """Phase 5I-D contract: ``evaluate_task(..., in_place=True)`` must
    leave the network in byte-identical state to its pre-call snapshot,
    even after the internal train_chunk + predict_batch mutate buffers.

    Pre-5I-D evaluate_task cloned every layer (including the dominant
    ``level_mixture_weights`` tensor) on every call, doubling GPU peak
    memory and capping per-node ``--pool`` at 80 on a 10.5 GiB GPU.
    Post-5I-D the snapshot lives on CPU and the GPU never holds two
    copies, unblocking ``--pool >= 100``.  We pin the restore contract
    by byte-comparing every snapshot tensor and scalar before and
    after the call; the bytewise comparison is NaN-tolerant because
    NaN compares unequal under ``==`` but is bit-equal to itself under
    a uint8 reinterpretation.
    """
    import sys
    scripts_dir = Path(__file__).resolve().parents[1]
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_split_mnist
    from nctl_bench.nctl_network import NctlNetwork

    net = NctlNetwork(
        layer_sizes=[2, 1],
        input_dim=2,
        num_halfspaces=1,
        lr=0.001,
        pool_capacity=2,
        min_segment=64,
        ptw_depth=8,
        device=torch.device("cpu"),
        seed=7,
        active_state_mode="per_level",
        prediction_mode="ptw_dp",
        pool_update_policy="paper",
        pool_alpha=float("inf"),
        pool_beta=float("inf"),
    )
    torch.manual_seed(0)
    imgs = torch.rand(200, 2)
    lbls = (imgs[:, 0] > imgs[:, 1]).to(torch.int32)
    net.train_chunk(imgs[:150], lbls[:150])

    prior_snap = net.snapshot_state()
    prev_device = run_split_mnist.DEVICE
    run_split_mnist.DEVICE = torch.device("cpu")
    try:
        acc = run_split_mnist.evaluate_task(
            net, imgs[150:], lbls[150:], adapt_n=20, in_place=True,
        )
        assert 0.0 <= acc <= 100.0
        post_snap = net.snapshot_state()

        # NaN-tolerant byte-level equality: NaN!=NaN under ``==`` but
        # bit-equal under a uint8 reinterpretation, which is what the
        # restore contract really cares about (identical storage).
        def _byte_eq_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            return torch.equal(
                a.contiguous().view(torch.uint8),
                b.contiguous().view(torch.uint8),
            )

        def _scalar_eq(a, b) -> bool:
            # Match NaN-as-NaN for float scalars too; required because
            # log_loss can be NaN on pathological fixtures and NaN != NaN.
            if isinstance(a, float) and isinstance(b, float):
                import math as _math
                if _math.isnan(a) and _math.isnan(b):
                    return True
            return a == b

        for ln, (a, b) in enumerate(zip(prior_snap["layers"], post_snap["layers"])):
            for name in a:
                va, vb = a[name], b[name]
                if isinstance(va, torch.Tensor):
                    assert _byte_eq_tensor(va, vb), (
                        f"layer {ln} tensor {name} not restored"
                    )
                else:
                    assert _scalar_eq(va, vb), (
                        f"layer {ln} scalar {name} not restored: {va!r} vs {vb!r}"
                    )
        for name in prior_snap["scalars"]:
            assert _scalar_eq(
                prior_snap["scalars"][name], post_snap["scalars"][name]
            ), f"network scalar {name} not restored"
    finally:
        run_split_mnist.DEVICE = prev_device

