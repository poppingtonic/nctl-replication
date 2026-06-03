"""CPU unit tests for bd 30.12 evaluate_task common-suffix offset.

These tests use a fake network with deterministic prediction to
verify only the slicing contract -- they do not exercise the CUDA
kernel.  The fake records the train_chunk argument shape and the
predict_batch slice it received into ``events`` (which lives OUTSIDE
the snapshot/restore cycle) so the recording survives evaluate_task's
restore call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_split_mnist import evaluate_task  # noqa: E402


class _FakeNet:
    """Minimal stand-in supporting snapshot/restore + train/predict.

    ``events`` is intentionally OUTSIDE the snapshot domain: callers
    inspect it to verify what the evaluation harness asked the network
    to do, even after evaluate_task's restore_state runs.
    """

    def __init__(self, n_inputs: int = 4) -> None:
        self.n_inputs = n_inputs
        self.trained_first_value: float | None = None
        self.events: dict = {
            "predict_calls": 0,
            "last_predict_first_value": None,
            "last_predict_size": None,
            "train_first_value": None,
        }

    def snapshot_state(self) -> dict:
        return {"trained_first_value": self.trained_first_value}

    def restore_state(self, snap: dict) -> None:
        self.trained_first_value = snap["trained_first_value"]

    def train_chunk(self, imgs: torch.Tensor, lbls: torch.Tensor) -> None:
        if len(imgs):
            self.trained_first_value = float(imgs[0, 0].item())
            self.events["train_first_value"] = self.trained_first_value

    def predict_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        self.events["predict_calls"] += 1
        self.events["last_predict_size"] = int(imgs.shape[0])
        self.events["last_predict_first_value"] = (
            float(imgs[0, 0].item()) if imgs.numel() else None
        )
        return torch.zeros(imgs.shape[0], device=imgs.device)


def _make_data(n: int = 1000, n_inputs: int = 4):
    imgs = torch.arange(n * n_inputs, dtype=torch.float32).reshape(n, n_inputs)
    lbls = torch.zeros(n, dtype=torch.int32)
    return imgs, lbls


def test_default_evaluates_on_suffix_after_adapt() -> None:
    """Baseline: eval_offset=None reproduces the legacy slice."""
    imgs, lbls = _make_data(n=100)
    net = _FakeNet()
    evaluate_task(net, imgs, lbls, adapt_n=10, in_place=True)
    assert net.events["last_predict_size"] == 90
    assert net.events["last_predict_first_value"] == 10 * 4  # imgs[10,0]


def test_eval_offset_above_adapt_uses_common_suffix() -> None:
    """eval_offset overrides the slice to test[eval_offset:]."""
    imgs, lbls = _make_data(n=1000)
    net_small = _FakeNet()
    evaluate_task(net_small, imgs, lbls, adapt_n=10, in_place=True, eval_offset=200)
    net_big = _FakeNet()
    evaluate_task(net_big, imgs, lbls, adapt_n=100, in_place=True, eval_offset=200)
    # Both cells must predict on the SAME suffix (test[200:]) so the
    # adapt_n sweep is comparable.
    assert net_small.events["last_predict_size"] == 800
    assert net_big.events["last_predict_size"] == 800
    assert net_small.events["last_predict_first_value"] == 200 * 4
    assert net_big.events["last_predict_first_value"] == 200 * 4
    # But the adapt step differed in size; both adapt from the start.
    assert net_small.events["train_first_value"] == 0.0
    assert net_big.events["train_first_value"] == 0.0


def test_eval_offset_below_adapt_is_raised_to_adapt() -> None:
    """A misconfigured eval_offset < adapt_n must not leak adapt samples."""
    imgs, lbls = _make_data(n=100)
    net = _FakeNet()
    evaluate_task(net, imgs, lbls, adapt_n=20, in_place=True, eval_offset=5)
    assert net.events["last_predict_size"] == 80
    assert net.events["last_predict_first_value"] == 20 * 4


def test_eval_offset_beyond_data_returns_50_pct() -> None:
    """When eval_offset >= n there is no eval set; the baseline 50% sentinel applies."""
    imgs, lbls = _make_data(n=50)
    net = _FakeNet()
    result = evaluate_task(net, imgs, lbls, adapt_n=10, in_place=True, eval_offset=100)
    assert result == 50.0
    assert net.events["predict_calls"] == 0


def test_snapshot_restored_after_evaluate() -> None:
    """The in-place evaluate path must always restore the snapshot."""
    imgs, lbls = _make_data(n=100)
    net = _FakeNet()
    net.trained_first_value = -123.0  # pre-existing state we expect to survive.
    evaluate_task(net, imgs, lbls, adapt_n=10, in_place=True, eval_offset=50)
    assert net.trained_first_value == -123.0
