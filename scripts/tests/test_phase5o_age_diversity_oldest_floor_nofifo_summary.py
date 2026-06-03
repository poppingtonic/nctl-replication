from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "sweep_results"
    / "phase5o_age_diversity_oldest_floor"
    / "summarise_nofifo.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("phase5o_nofifo_summarise", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(seed: int, acc: float, task1: float, floor: int = 2) -> dict:
    return {
        "file": f"seed{seed}.json",
        "seed": seed,
        "avg_accuracy": acc,
        "avg_forgetting": 10.0 - seed,
        "total_time": 1.5,
        "pool_oldest_floor": floor,
        "final_task_accuracy": {"1": task1, "2": 90.0},
    }


def _write_floor(root: Path, floor: int, seed: int, acc: float, task1: float) -> None:
    payload = {
        "seed": seed,
        "avg_accuracy": acc,
        "avg_forgetting": 8.0 - seed,
        "total_time": 1.5,
        "pool_oldest_floor": floor,
        "acc_matrix": [[None], [task1, 90.0]],
        "pool_provenance": {"task_histogram": {"1": seed}},
    }
    (root / f"age-diversity-oldest-floor_floor{floor}_seed{seed}.json").write_text(
        json.dumps(payload)
    )


def _write_floor4(root: Path, seed: int, acc: float, task1: float) -> None:
    _write_floor(root, 4, seed, acc, task1)


def test_phase5o_nofifo_summary_compares_floor4_to_prior_baselines(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    mod.ROOT = tmp_path
    (tmp_path / "summary_ab.json").write_text(
        json.dumps(
            {
                "fifo": {
                    "avg_accuracy_mean": 91.0,
                    "avg_forgetting_mean": 9.0,
                    "results": [_row(1, 90.0, 80.0), _row(2, 92.0, 82.0)],
                },
                "age_diversity_oldest_floor": {
                    "avg_accuracy_mean": 94.5,
                    "avg_forgetting_mean": 5.0,
                    "results": [
                        _row(1, 94.0, 95.0),
                        _row(2, 95.0, 97.0),
                    ],
                },
            }
        )
    )
    _write_floor4(tmp_path, 1, 95.0, 98.0)
    _write_floor4(tmp_path, 2, 96.0, 100.0)

    assert mod.main() == 0

    summary = json.loads((tmp_path / "summary_nofifo.json").read_text())
    assert summary["experiment"] == "phase5o_age_diversity_oldest_floor_floor4_nofifo"
    assert summary["floor4_candidate"]["avg_accuracy_mean"] == 95.5
    assert summary["floor4_vs_floor2"]["delta_mean_acc"] == 1.0
    assert summary["floor4_vs_floor2"]["delta_final_task_accuracy_mean"]["1"] == 3.0
    assert summary["floor4_vs_fifo"]["delta_mean_acc"] == 4.5
    assert summary["floor4_crosses_paper_target"] is True


def test_phase5o_nofifo_summary_accepts_dashed_prior_key(tmp_path: Path) -> None:
    mod = _load_module()
    mod.ROOT = tmp_path
    (tmp_path / "summary_ab.json").write_text(
        json.dumps(
            {
                "fifo": {"avg_accuracy_mean": 91.0, "results": []},
                "age-diversity-oldest-floor": {
                    "avg_accuracy_mean": 94.5,
                    "results": [],
                },
            }
        )
    )
    _write_floor4(tmp_path, 1, 95.0, 98.0)

    assert mod.main() == 0
    summary = json.loads((tmp_path / "summary_nofifo.json").read_text())
    assert summary["floor2_baseline"]["avg_accuracy_mean"] == 94.5


def test_phase5o_nofifo_summary_can_target_floor6(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POOL_OLDEST_FLOOR", "6")
    mod = _load_module()
    mod.ROOT = tmp_path
    (tmp_path / "summary_ab.json").write_text(
        json.dumps(
            {
                "fifo": {
                    "avg_accuracy_mean": 91.0,
                    "avg_forgetting_mean": 9.0,
                    "results": [_row(1, 90.0, 80.0), _row(2, 92.0, 82.0)],
                },
                "age_diversity_oldest_floor": {
                    "avg_accuracy_mean": 94.5,
                    "avg_forgetting_mean": 5.0,
                    "results": [
                        _row(1, 94.0, 95.0),
                        _row(2, 95.0, 97.0),
                    ],
                },
            }
        )
    )
    _write_floor(tmp_path, 6, 1, 95.2, 99.0)
    _write_floor(tmp_path, 6, 2, 95.4, 98.0)

    assert mod.main() == 0

    summary = json.loads((tmp_path / "summary_floor6_nofifo.json").read_text())
    assert summary["experiment"] == "phase5o_age_diversity_oldest_floor_floor6_nofifo"
    assert summary["floor6_candidate"]["avg_accuracy_mean"] == pytest.approx(95.3)
    assert summary["floor6_vs_floor2"]["delta_mean_acc"] == pytest.approx(0.8)
    assert summary["floor6_crosses_paper_target"] is True


def test_phase5o_nofifo_summary_fails_fast_on_bad_floor4_json(tmp_path: Path) -> None:
    mod = _load_module()
    mod.ROOT = tmp_path
    (tmp_path / "summary_ab.json").write_text(
        json.dumps(
            {
                "fifo": {"avg_accuracy_mean": 91.0, "results": []},
                "age_diversity_oldest_floor": {
                    "avg_accuracy_mean": 94.5,
                    "results": [],
                },
            }
        )
    )
    (tmp_path / "age-diversity-oldest-floor_floor4_seed1.json").write_text("{")

    with pytest.raises(SystemExit, match="could not read floor-4 result"):
        mod.main()
