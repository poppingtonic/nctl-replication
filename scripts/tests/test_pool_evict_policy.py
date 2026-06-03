"""CPU unit tests for bd 30.11 pool eviction policy.

These tests intentionally avoid every CUDA-dependent code path: they
construct a ``CudaFmnMixtureLayer`` on the CPU device, populate the
pool by direct tensor assignment (the same primitives
``_insert_pool_snapshot_fifo`` uses internally), and then drive a
single insertion to inspect which slot was overwritten.  This keeps
the tests fast and exercisable in the sandbox without the multilevel
CUDA extension.

The FIFO regression test pins the pre-30.11 behaviour bit-identically
so the refactor cannot drift the 30.9 acceptance recipe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from nctl_bench.nctl_network import CudaFmnMixtureLayer  # noqa: E402


def _make_layer(pool_capacity: int, policy: str) -> CudaFmnMixtureLayer:
    """Build a tiny CPU layer with the given pool capacity and policy."""
    return CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=1,
        num_halfspaces=1,
        lr=0.1,
        pool_capacity=pool_capacity,
        device=torch.device("cpu"),
        seed=0,
        pool_evict_policy=policy,
    )


def _seed_pool(layer: CudaFmnMixtureLayer, task_ids: list[int]) -> None:
    """Populate the layer's pool with the given task ids in slot order."""
    assert len(task_ids) == layer.pool_capacity
    layer.pool_sizes[0] = layer.pool_capacity
    for slot, tid in enumerate(task_ids):
        layer.pool_task_ids[0, slot] = tid
        layer.pool_insert_indices[0, slot] = slot
        layer.pool_levels[0, slot] = 0
        layer.model_log_probs[0, slot] = 0.0
        # Tag snapshots with the task id so we can verify the surviving
        # set after eviction without trusting pool_task_ids alone.
        layer.pool_snapshots[0, slot].fill_(float(tid))


def _set_insert_indices(layer: CudaFmnMixtureLayer, indices: list[int]) -> None:
    """Overwrite pool insertion indices while preserving slot order."""
    assert len(indices) == layer.pool_capacity
    for slot, idx in enumerate(indices):
        layer.pool_insert_indices[0, slot] = idx


def _insert(layer: CudaFmnMixtureLayer, new_task: int) -> None:
    layer._insert_pool_snapshot_fifo(
        node=0,
        weights=torch.full(
            (layer.C, layer.K), float(new_task), device=layer.device
        ),
        score=torch.tensor(0.0, device=layer.device),
        deepest_level=0,
        task_id=new_task,
        global_index=10_000,
        level=None,
    )


# ---------------------------------------------------------------------------
# Default policy (fifo) — regression parity against pre-30.11 behaviour.
# ---------------------------------------------------------------------------

def test_fifo_default_evicts_slot_zero() -> None:
    layer = _make_layer(pool_capacity=4, policy="fifo")
    _seed_pool(layer, [1, 2, 3, 4])
    _insert(layer, 9)
    assert layer.pool_task_ids[0].tolist() == [2, 3, 4, 9]
    # New snapshot must always land in the last slot regardless of policy.
    assert torch.equal(
        layer.pool_snapshots[0, -1],
        torch.full((layer.C, layer.K), 9.0, device=layer.device),
    )


def test_fifo_unchanged_when_all_singletons() -> None:
    layer = _make_layer(pool_capacity=4, policy="fifo")
    _seed_pool(layer, [11, 22, 33, 44])
    _insert(layer, 55)
    # Identical to the previous test's signature: slot 0 always evicted.
    assert layer.pool_task_ids[0].tolist() == [22, 33, 44, 55]


def test_fifo_unchanged_when_duplicates_present() -> None:
    """Pinning the pre-30.11 behaviour: FIFO never inspects duplicates."""
    layer = _make_layer(pool_capacity=4, policy="fifo")
    _seed_pool(layer, [1, 2, 2, 3])  # task 2 has a duplicate, task 1 singleton
    _insert(layer, 4)
    # FIFO still drops the singleton task 1 at slot 0.
    assert layer.pool_task_ids[0].tolist() == [2, 2, 3, 4]


# ---------------------------------------------------------------------------
# task-floor policy (bd 30.11 diagnostic).
# ---------------------------------------------------------------------------

def test_task_floor_protects_singleton_and_evicts_duplicate() -> None:
    layer = _make_layer(pool_capacity=4, policy="task-floor")
    _seed_pool(layer, [1, 2, 2, 3])  # task 1 singleton must survive
    _insert(layer, 4)
    surviving = layer.pool_task_ids[0].tolist()
    # task 1 still present; one of the task-2 duplicates removed.
    assert 1 in surviving
    assert surviving.count(2) == 1
    assert surviving.count(3) == 1
    assert surviving.count(4) == 1
    # New snapshot always lands at the last slot.
    assert surviving[-1] == 4


def test_task_floor_falls_back_to_fifo_when_all_singletons() -> None:
    layer = _make_layer(pool_capacity=4, policy="task-floor")
    _seed_pool(layer, [1, 2, 3, 4])  # no duplicates -> nothing safe to evict
    _insert(layer, 5)
    # Falls back to FIFO: oldest (task 1) evicted.
    assert layer.pool_task_ids[0].tolist() == [2, 3, 4, 5]


def test_task_floor_picks_oldest_duplicate_first() -> None:
    """When several slots are evictable, the oldest one wins."""
    layer = _make_layer(pool_capacity=5, policy="task-floor")
    # task 2 appears at slots 1 and 3 (slot 1 is older).
    _seed_pool(layer, [1, 2, 3, 2, 4])
    _insert(layer, 9)
    surviving = layer.pool_task_ids[0].tolist()
    # task 1, 3, 4 singletons preserved; new task 9 at end; only one task-2.
    assert surviving == [1, 3, 2, 4, 9]


def test_task_floor_increments_evict_event_count() -> None:
    """Provenance counters must fire identically to the FIFO path."""
    layer = _make_layer(pool_capacity=4, policy="task-floor")
    _seed_pool(layer, [1, 2, 2, 3])
    before = dict(layer.provenance_event_counts)
    _insert(layer, 4)
    after = layer.provenance_event_counts
    assert after["evict"] == before["evict"] + 1
    assert after["append"] == before["append"]  # full pool -> evict, not append


def test_task_floor_records_evicted_task_in_provenance() -> None:
    """The evicted task counter must reflect the actual victim, not slot 0."""
    layer = _make_layer(pool_capacity=4, policy="task-floor")
    _seed_pool(layer, [1, 2, 2, 3])  # FIFO would drop task 1; task-floor drops a 2.
    _insert(layer, 4)
    counts = layer.provenance_evicted_task_counts
    assert counts.get(2, 0) == 1
    assert counts.get(1, 0) == 0  # task 1 protected by the floor


# ---------------------------------------------------------------------------
# age-diversity policy (task-free bd 30.11 follow-up).
# ---------------------------------------------------------------------------

def test_age_diversity_evicts_temporally_redundant_slot() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-diversity")
    _seed_pool(layer, [1, 2, 3, 4])
    _set_insert_indices(layer, [100, 10_000, 10_010, 50_000])
    _insert(layer, 9)
    # Slots 1 and 2 are the closest temporal neighbours; the older duplicate
    # wins the tie, so the isolated oldest slot is retained without task ids.
    assert layer.pool_insert_indices[0].tolist() == [100, 10_010, 50_000, 10_000]
    assert layer.pool_task_ids[0].tolist() == [1, 3, 4, 9]


def test_age_diversity_ties_fall_back_to_oldest_slot() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-diversity")
    _seed_pool(layer, [1, 2, 3, 4])
    _set_insert_indices(layer, [0, 10, 20, 30])
    _insert(layer, 9)
    assert layer.pool_task_ids[0].tolist() == [2, 3, 4, 9]


def test_age_diversity_does_not_read_task_ids() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-diversity")
    # Task ids would make slot 0 look like a protected singleton under
    # task-floor.  age-diversity ignores them and evicts by temporal density.
    _seed_pool(layer, [1, 2, 2, 3])
    _set_insert_indices(layer, [100, 110, 10_000, 50_000])
    _insert(layer, 9)
    assert layer.pool_task_ids[0].tolist() == [2, 2, 3, 9]
    assert 100 not in layer.pool_insert_indices[0].tolist()


# ---------------------------------------------------------------------------
# age-bucket-floor policy (task-free PTW-scale retention).
# ---------------------------------------------------------------------------

def test_age_bucket_floor_protects_singleton_age_buckets() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-bucket-floor")
    _seed_pool(layer, [1, 2, 3, 4])
    # With current_index=max+1=1001, slots 1 and 2 both have age bucket 2
    # (ages 6 and 4).  Slot 0 is much older and slot 3 is newest.
    _set_insert_indices(layer, [1, 995, 997, 1000])
    _insert(layer, 9)
    assert layer.pool_insert_indices[0].tolist() == [1, 997, 1000, 10_000]
    assert layer.pool_task_ids[0].tolist() == [1, 3, 4, 9]


def test_age_bucket_floor_falls_back_to_fifo_when_all_buckets_singletons() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-bucket-floor")
    _seed_pool(layer, [1, 2, 3, 4])
    # Ages at current_index=1001 are 1000, 513, 65, 1: distinct log2 buckets.
    _set_insert_indices(layer, [1, 488, 936, 1000])
    _insert(layer, 9)
    assert layer.pool_task_ids[0].tolist() == [2, 3, 4, 9]


def test_age_bucket_floor_does_not_read_task_ids() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-bucket-floor")
    _seed_pool(layer, [1, 2, 2, 3])
    _set_insert_indices(layer, [995, 997, 1, 1000])
    _insert(layer, 9)
    assert layer.pool_task_ids[0].tolist() == [2, 2, 3, 9]
    assert 995 not in layer.pool_insert_indices[0].tolist()


# ---------------------------------------------------------------------------
# age-diversity-oldest-floor policy (task-free, protects earliest snapshots).
# ---------------------------------------------------------------------------

def test_age_diversity_oldest_floor_protects_oldest_slot() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-diversity-oldest-floor")
    layer.pool_oldest_floor = 1
    _seed_pool(layer, [1, 2, 3, 4])
    # Plain age-diversity would drop slot 0 (the {100,110} pair ties and the
    # oldest slot wins).  The oldest floor protects slot 0, so the next most
    # redundant eligible slot (110) is evicted instead.
    _set_insert_indices(layer, [100, 200, 10_000, 50_000])
    _insert(layer, 9)
    # Slot 1 (index 200) is evicted; surviving slots shift down and the new
    # snapshot (global_index 10_000) lands last.
    assert layer.pool_insert_indices[0].tolist() == [100, 10_000, 50_000, 10_000]
    assert layer.pool_task_ids[0].tolist() == [1, 3, 4, 9]


def test_age_diversity_oldest_floor_falls_back_to_fifo_when_floor_covers_pool() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-diversity-oldest-floor")
    layer.pool_oldest_floor = 4  # floor protects every slot -> nothing eligible
    _seed_pool(layer, [1, 2, 3, 4])
    _set_insert_indices(layer, [100, 110, 10_000, 50_000])
    _insert(layer, 9)
    # FIFO fallback: the oldest slot is evicted.
    assert layer.pool_task_ids[0].tolist() == [2, 3, 4, 9]


def test_age_diversity_oldest_floor_ignores_task_ids() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-diversity-oldest-floor")
    layer.pool_oldest_floor = 1
    # task 2 (slot 1) is a singleton that task-floor would protect; the
    # oldest-floor variant decides purely on temporal density and evicts it.
    _seed_pool(layer, [1, 2, 3, 3])
    _set_insert_indices(layer, [0, 1000, 1010, 5000])
    _insert(layer, 9)
    assert layer.pool_task_ids[0].tolist() == [1, 3, 3, 9]
    assert 1000 not in layer.pool_insert_indices[0].tolist()


def test_age_diversity_oldest_floor_roundtrips_via_state_dict() -> None:
    layer = _make_layer(pool_capacity=4, policy="age-diversity-oldest-floor")
    layer.pool_oldest_floor = 3
    restored = CudaFmnMixtureLayer.from_state_dict(
        layer.state_dict(), torch.device("cpu")
    )
    assert restored.pool_evict_policy == "age-diversity-oldest-floor"
    assert restored.pool_oldest_floor == 3


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------

def test_unknown_evict_policy_rejected() -> None:
    with pytest.raises(ValueError, match="unknown pool_evict_policy"):
        _make_layer(pool_capacity=2, policy="random-spaghetti")


def test_unknown_evict_policy_rejected_from_state_dict() -> None:
    layer = _make_layer(pool_capacity=2, policy="fifo")
    state = layer.state_dict()
    state["pool_evict_policy"] = "random-spaghetti"
    with pytest.raises(ValueError, match="unknown pool_evict_policy in state_dict"):
        CudaFmnMixtureLayer.from_state_dict(state, torch.device("cpu"))


def test_default_policy_is_fifo() -> None:
    layer = CudaFmnMixtureLayer(
        num_nodes=1,
        num_inputs=2,
        input_dim=1,
        num_halfspaces=1,
        lr=0.1,
        pool_capacity=2,
        device=torch.device("cpu"),
        seed=0,
    )
    assert layer.pool_evict_policy == "fifo"
