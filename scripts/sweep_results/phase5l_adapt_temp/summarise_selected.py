#!/usr/bin/env python3
"""Aggregate selected 30.12 confirmation cells across seeds."""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent


def final_task_accuracies(d: dict) -> dict[str, float]:
    matrix = d.get("acc_matrix") or d.get("accuracy_matrix")
    if not matrix:
        return {}
    final = matrix[-1]
    return {str(i + 1): float(v) for i, v in enumerate(final) if v is not None}


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: summarise_selected.py ADAPT TEMP [SEEDS...]", file=sys.stderr)
        return 1
    adapt, temp = sys.argv[1], sys.argv[2]
    seeds = sys.argv[3:] or ["1", "2", "3", "4", "5"]
    rows = []
    for seed in seeds:
        path = ROOT / f"seed{seed}_adapt{adapt}_temp{temp}.json"
        if not path.exists():
            rows.append({"seed": int(seed), "status": "missing", "file": path.name})
            continue
        d = json.loads(path.read_text())
        rows.append({
            "seed": d.get("seed", int(seed)),
            "status": "ok",
            "file": path.name,
            "avg_accuracy": d.get("avg_accuracy"),
            "avg_forgetting": d.get("avg_forgetting"),
            "total_time": d.get("total_time"),
            "final_task_accuracy": final_task_accuracies(d),
            "task_histogram": (d.get("pool_provenance") or {}).get("task_histogram"),
        })
    ok = [r for r in rows if r.get("status") == "ok"]
    acc = [r["avg_accuracy"] for r in ok]
    fgt = [r["avg_forgetting"] for r in ok]
    summary = {
        "experiment": "phase5l_adapt_temp_selected_confirm",
        "adapt_n": int(adapt),
        "posterior_temp": float(temp),
        "eval_suffix_from": 1000,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_ok": len(ok),
        "rows": rows,
        "avg_accuracy_mean": statistics.mean(acc) if acc else None,
        "avg_accuracy_std_pop": statistics.pstdev(acc) if len(acc) > 1 else None,
        "avg_forgetting_mean": statistics.mean(fgt) if fgt else None,
        "avg_forgetting_std_pop": statistics.pstdev(fgt) if len(fgt) > 1 else None,
    }
    out = ROOT / f"summary_confirm_adapt{adapt}_temp{temp}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
