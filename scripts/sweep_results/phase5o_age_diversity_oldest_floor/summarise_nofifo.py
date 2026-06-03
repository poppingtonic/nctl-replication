#!/usr/bin/env python3
"""Summarise a candidate-only oldest-floor follow-up against existing baselines."""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import time

ROOT = pathlib.Path(__file__).resolve().parent
CANDIDATE = "age-diversity-oldest-floor"
FLOOR = int(os.environ.get("POOL_OLDEST_FLOOR", "4"))
SUMMARY_OUT = os.environ.get(
    "SUMMARY_OUT",
    "summary_nofifo.json" if FLOOR == 4 else f"summary_floor{FLOOR}_nofifo.json",
)
PAPER_TARGET = 95.07


def _final_task_accuracies(d: dict) -> dict[str, float]:
    matrix = d.get("acc_matrix") or d.get("accuracy_matrix")
    if not matrix:
        return {}
    final = matrix[-1]
    return {str(i + 1): float(v) for i, v in enumerate(final) if v is not None}


def _load_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read {path}: {exc}") from exc


def _row_from_file(path: pathlib.Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"could not read floor-{FLOOR} result {path.name}: {exc}"
        ) from exc
    return {
        "file": path.name,
        "seed": d.get("seed"),
        "avg_accuracy": d.get("avg_accuracy"),
        "avg_forgetting": d.get("avg_forgetting"),
        "total_time": d.get("total_time"),
        "pool_oldest_floor": d.get("pool_oldest_floor"),
        "task_histogram": (d.get("pool_provenance") or {}).get("task_histogram"),
        "final_task_accuracy": _final_task_accuracies(d),
    }


def _aggregate_rows(label: str, rows: list[dict]) -> dict:
    acc = [r["avg_accuracy"] for r in rows if r["avg_accuracy"] is not None]
    fgt = [r["avg_forgetting"] for r in rows if r["avg_forgetting"] is not None]
    return {
        "label": label,
        "policy": CANDIDATE,
        "pool_oldest_floor": FLOOR,
        "n": len(rows),
        "results": rows,
        "avg_accuracy_mean": statistics.mean(acc) if acc else None,
        "avg_accuracy_std_pop": statistics.pstdev(acc) if len(acc) > 1 else None,
        "avg_forgetting_mean": statistics.mean(fgt) if fgt else None,
        "avg_forgetting_std_pop": statistics.pstdev(fgt) if len(fgt) > 1 else None,
        "final_task_accuracy_mean": _per_task_mean(rows),
    }


def _aggregate_floor() -> dict:
    files = sorted(ROOT.glob(f"{CANDIDATE}_floor{FLOOR}_seed*.json"))
    rows = [_row_from_file(f) for f in files]
    return _aggregate_rows(f"{CANDIDATE}-floor{FLOOR}", rows)


def _per_task_mean(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for task in ("1", "2", "3", "4", "5"):
        vals = [
            r["final_task_accuracy"][task]
            for r in rows
            if task in r.get("final_task_accuracy", {})
        ]
        if vals:
            out[task] = statistics.mean(vals)
    return out


def _mean_delta(candidate: dict, baseline: dict, metric: str) -> float | None:
    cand = candidate.get(metric)
    base = baseline.get(metric)
    if cand is None or base is None:
        return None
    return cand - base


def _delta_by_task(candidate: dict, baseline: dict) -> dict[str, float]:
    cand_mean = candidate.get("final_task_accuracy_mean") or _per_task_mean(
        candidate.get("results", [])
    )
    base_mean = baseline.get("final_task_accuracy_mean") or _per_task_mean(
        baseline.get("results", [])
    )
    return {
        task: cand_mean[task] - base_mean[task]
        for task in sorted(set(cand_mean) & set(base_mean), key=int)
    }


def _comparison(candidate: dict, baseline: dict) -> dict:
    return {
        "delta_mean_acc": _mean_delta(candidate, baseline, "avg_accuracy_mean"),
        "delta_mean_forgetting": _mean_delta(
            candidate, baseline, "avg_forgetting_mean"
        ),
        "delta_final_task_accuracy_mean": _delta_by_task(candidate, baseline),
    }


def _prior_section(prior: dict, *keys: str) -> dict:
    section = next(
        (prior[key] for key in keys if isinstance(prior.get(key), dict)),
        None,
    )
    if not isinstance(section, dict):
        names = ", ".join(repr(key) for key in keys)
        raise SystemExit(
            f"summary_ab.json is missing required object ({names}); "
            "run the full phase5o A/B summariser first"
        )
    return section


def main() -> int:
    prior = _load_json(ROOT / "summary_ab.json")
    fifo = _prior_section(prior, "fifo")
    floor2 = _prior_section(
        prior,
        "age_diversity_oldest_floor",
        "age-diversity-oldest-floor",
    )
    floor2["final_task_accuracy_mean"] = _per_task_mean(floor2.get("results", []))
    fifo["final_task_accuracy_mean"] = _per_task_mean(fifo.get("results", []))

    floor = _aggregate_floor()
    prefix = f"floor{FLOOR}"
    summary = {
        "experiment": f"phase5o_age_diversity_oldest_floor_{prefix}_nofifo",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper_target_avg_accuracy": PAPER_TARGET,
        f"{prefix}_crosses_paper_target": (
            floor["avg_accuracy_mean"] is not None
            and floor["avg_accuracy_mean"] >= PAPER_TARGET
        ),
        f"{prefix}_delta_to_paper_target": (
            floor["avg_accuracy_mean"] - PAPER_TARGET
            if floor["avg_accuracy_mean"] is not None
            else None
        ),
        "fifo_baseline": fifo,
        "floor2_baseline": floor2,
        f"{prefix}_candidate": floor,
        f"{prefix}_vs_fifo": _comparison(floor, fifo),
        f"{prefix}_vs_floor2": _comparison(floor, floor2),
    }
    try:
        (ROOT / SUMMARY_OUT).write_text(json.dumps(summary, indent=2, sort_keys=True))
    except OSError as exc:
        raise SystemExit(f"could not write {SUMMARY_OUT}: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
