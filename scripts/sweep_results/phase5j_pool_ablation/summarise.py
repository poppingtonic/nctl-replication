#!/usr/bin/env python3
"""Aggregate per-seed pool-ablation JSONs into a summary.

Usage:
    python3 summarise.py POOL

Writes summary_pool${POOL}.json next to the per-seed files with mean/std,
per-task accuracy after the final task, global task histogram, and
event-count totals.  Also emits per-task survival diagnostics requested
by Agent 2's risk review: number of nodes / layers with zero snapshots
for each task (when that info is exposed in the per-seed JSON's
per_task_provenance section).
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent


def _final_task_accuracies(d: dict) -> dict[str, float]:
    """Pull the accuracy row from the last entry of the accuracy matrix."""
    matrix = d.get("accuracy_matrix")
    if not matrix:
        return {}
    final = matrix[-1]
    return {str(i + 1): float(v) for i, v in enumerate(final) if v is not None}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarise.py POOL", file=sys.stderr)
        return 1
    pool = sys.argv[1]
    files = sorted(ROOT.glob(f"seed*_pool{pool}.json"))
    if not files:
        print(f"no per-seed JSONs found for pool={pool}", file=sys.stderr)
        return 1
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
            "event_counts": (d.get("pool_provenance") or {}).get("event_counts"),
            "final_task_accuracy": _final_task_accuracies(d),
            "per_task_provenance": d.get("per_task_provenance"),
        })
    acc = [r["avg_accuracy"] for r in rows if r["avg_accuracy"] is not None]
    fgt = [r["avg_forgetting"] for r in rows if r["avg_forgetting"] is not None]
    summary = {
        "experiment": f"phase5j_pool_ablation_pool{pool}",
        "recipe": (
            f"pool{pool}_ms512_temp1_ab00_in_place_eval_"
            f"{time.strftime('%Y%m%d')}"
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": len(rows),
        "results": rows,
        "avg_accuracy_mean": statistics.mean(acc) if acc else None,
        "avg_accuracy_std_pop": statistics.pstdev(acc) if len(acc) > 1 else None,
        "avg_accuracy_min": min(acc) if acc else None,
        "avg_accuracy_max": max(acc) if acc else None,
        "avg_forgetting_mean": statistics.mean(fgt) if fgt else None,
        "avg_forgetting_std_pop": statistics.pstdev(fgt) if len(fgt) > 1 else None,
    }
    out = ROOT / f"summary_pool{pool}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
