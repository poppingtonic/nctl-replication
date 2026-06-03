from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "sweep_results"
    / "phase5o_age_diversity_oldest_floor"
    / "summarise.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("phase5o_summarise", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_result(root: Path, policy: str, seed: int, acc: float, task1: float) -> None:
    payload = {
        "seed": seed,
        "avg_accuracy": acc,
        "avg_forgetting": 10.0 - seed,
        "total_time": 1.5,
        "pool_oldest_floor": 2,
        "acc_matrix": [[None], [task1, 90.0]],
        "pool_provenance": {"task_histogram": {"1": seed}},
    }
    (root / f"{policy}_seed{seed}.json").write_text(json.dumps(payload))


def test_phase5o_summary_reports_policy_and_task_deltas(tmp_path: Path) -> None:
    mod = _load_module()
    mod.ROOT = tmp_path
    _write_result(tmp_path, "fifo", 1, 90.0, 80.0)
    _write_result(tmp_path, "fifo", 2, 92.0, 82.0)
    _write_result(tmp_path, "age-diversity-oldest-floor", 1, 95.0, 88.0)
    _write_result(tmp_path, "age-diversity-oldest-floor", 2, 97.0, 90.0)

    assert mod.main() == 0

    summary = json.loads((tmp_path / "summary_ab.json").read_text())
    assert summary["experiment"] == "phase5o_age_diversity_oldest_floor_ab"
    assert summary["delta_mean_acc"] == 5.0
    assert summary["delta_final_task_accuracy_mean"]["1"] == 8.0
    assert summary["fifo"]["n"] == 2
    assert summary["age_diversity_oldest_floor"]["n"] == 2
