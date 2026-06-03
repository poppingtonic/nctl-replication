#!/usr/bin/env python3
"""Aggregate the fifo vs task-floor A/B sweep into a side-by-side summary."""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent


def _final_task_accuracies(d: dict) -> dict[str, float]:
    matrix = d.get("acc_matrix") or d.get("accuracy_matrix")
    if not matrix:
        return {}
    final = matrix[-1]
    return {str(i + 1): float(v) for i, v in enumerate(final) if v is not None}


def _aggregate(policy: str) -> dict:
    files = sorted(ROOT.glob(f"{policy}_seed*.json"))
    rows: list[dict] = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"skip {f.name}: {exc}", file=sys.stderr)
            continue
        rows.append({
            "file": f.name,
            "seed": d.get("seed"),
            "avg_accuracy": d.get("avg_accuracy"),
            "avg_forgetting": d.get("avg_forgetting"),
            "total_time": d.get("total_time"),
            "task_histogram": (d.get("pool_provenance") or {}).get("task_histogram"),
            "final_task_accuracy": _final_task_accuracies(d),
        })
    acc = [r["avg_accuracy"] for r in rows if r["avg_accuracy"] is not None]
    fgt = [r["avg_forgetting"] for r in rows if r["avg_forgetting"] is not None]
    return {
        "policy": policy,
        "n": len(rows),
        "results": rows,
        "avg_accuracy_mean": statistics.mean(acc) if acc else None,
        "avg_accuracy_std_pop": statistics.pstdev(acc) if len(acc) > 1 else None,
        "avg_forgetting_mean": statistics.mean(fgt) if fgt else None,
    }


def main() -> int:
    fifo = _aggregate("fifo")
    floor = _aggregate("task-floor")
    summary = {
        "experiment": "phase5k_task_floor_ab",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fifo": fifo,
        "task_floor": floor,
        "delta_mean_acc": (
            (floor["avg_accuracy_mean"] - fifo["avg_accuracy_mean"])
            if fifo["avg_accuracy_mean"] is not None
            and floor["avg_accuracy_mean"] is not None
            else None
        ),
    }
    (ROOT / "summary_ab.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
