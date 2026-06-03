#!/usr/bin/env python3
"""Render the adapt_n x posterior_temp grid as a per-task accuracy table."""
from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
ADAPTS = [10, 50, 200, 1000]
TEMPS = [0.5, 1.0, 2.0]


def _final_task_accuracies(d: dict) -> dict[str, float]:
    matrix = d.get("acc_matrix") or d.get("accuracy_matrix")
    if not matrix:
        return {}
    final = matrix[-1]
    return {str(i + 1): float(v) for i, v in enumerate(final) if v is not None}


def main() -> int:
    seed = sys.argv[1] if len(sys.argv) > 1 else "1"
    cells: dict[tuple[int, float], dict] = {}
    for adapt in ADAPTS:
        for temp in TEMPS:
            tag = f"adapt{adapt}_temp{temp}"
            path = ROOT / f"seed{seed}_{tag}.json"
            if not path.exists():
                cells[(adapt, temp)] = {"status": "missing"}
                continue
            d = json.loads(path.read_text())
            cells[(adapt, temp)] = {
                "status": "ok",
                "avg_accuracy": d.get("avg_accuracy"),
                "avg_forgetting": d.get("avg_forgetting"),
                "final_task_accuracy": _final_task_accuracies(d),
                "total_time": d.get("total_time"),
            }
    summary = {
        "experiment": "phase5l_adapt_temp",
        "seed": seed,
        "eval_suffix_from": 1000,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": {
            f"adapt={a},temp={t}": cells[(a, t)] for a in ADAPTS for t in TEMPS
        },
    }
    (ROOT / f"summary_seed{seed}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
