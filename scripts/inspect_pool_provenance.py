#!/usr/bin/env python3
"""Inspect NCTL pool-provenance JSON emitted by run_split_mnist.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK_NAMES = {
    "1": "T1(0,1)",
    "2": "T2(2,3)",
    "3": "T3(4,5)",
    "4": "T4(6,7)",
    "5": "T5(8,9)",
    "-1": "unknown",
}


def _fmt_hist(hist: dict[str, int] | None) -> str:
    if not hist:
        return "{}"
    parts = []
    total = sum(int(v) for v in hist.values())
    for key in sorted(hist, key=lambda k: int(k)):
        value = int(hist[key])
        pct = 100.0 * value / total if total else 0.0
        parts.append(f"{TASK_NAMES.get(key, key)}={value} ({pct:.1f}%)")
    return ", ".join(parts)


def _fmt_acc(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.1f}%" for v in values) + "]"


def build_report(payload: dict[str, Any], source: Path) -> str:
    prov = payload.get("pool_provenance")
    if not prov:
        raise KeyError(f"{source} does not contain pool_provenance")

    acc_matrix = payload.get("acc_matrix", [])
    final = acc_matrix[-1] if acc_matrix else []
    diag = [acc_matrix[i][i] for i in range(min(len(acc_matrix), 5))]

    lines: list[str] = []
    lines.append(f"Pool provenance report: {source}")
    lines.append("=" * 80)
    lines.append(
        "Config: "
        f"dataset={payload.get('dataset')} seed={payload.get('seed')} "
        f"nodes={payload.get('nodes')} lr={payload.get('lr')} "
        f"pool={payload.get('pool_capacity')} min_segment={payload.get('min_segment')} "
        f"posterior_temp={payload.get('posterior_temp')}"
    )
    lines.append(
        f"Accuracy: avg={payload.get('avg_accuracy'):.2f}% "
        f"forgetting={payload.get('avg_forgetting'):.2f}%"
    )
    if final:
        lines.append(f"Final row T1..T5: {_fmt_acc(final)}")
    if diag:
        lines.append(f"Diagonal T1..T5:  {_fmt_acc(diag)}")
    lines.append("")

    lines.append("Overall final pool composition")
    lines.append("-" * 80)
    lines.append(f"Task histogram:        {_fmt_hist(prov.get('task_histogram'))}")
    lines.append(f"Append/evict counts:   {prov.get('event_counts', {})}")
    lines.append(f"Evicted task counts:   {_fmt_hist(prov.get('evicted_task_counts'))}")
    lines.append("")

    layers = prov.get("layers", [])
    for i, layer in enumerate(layers):
        lines.append(f"Layer {i}")
        lines.append("-" * 80)
        lines.append(
            f"Pool occupancy:        {layer.get('pool_size_total')} / "
            f"{layer.get('pool_capacity_total')}"
        )
        lines.append(f"Task histogram:        {_fmt_hist(layer.get('pool_task_histogram'))}")
        lines.append(f"Level histogram:       {layer.get('pool_level_histogram', {})}")
        lines.append(f"Append/evict counts:   {layer.get('event_counts', {})}")
        lines.append(f"Evicted task counts:   {_fmt_hist(layer.get('evicted_task_counts'))}")
        lines.append("Slot task histograms:")
        for slot, hist in enumerate(layer.get("slot_task_histograms", [])):
            lines.append(f"  slot {slot}: {_fmt_hist(hist)}")
        lines.append("")

    task_hist = prov.get("task_histogram", {})
    late = int(task_hist.get("4", 0)) + int(task_hist.get("5", 0))
    total = sum(int(v) for v in task_hist.values())
    lines.append("Interpretation")
    lines.append("-" * 80)
    if total:
        lines.append(
            f"Late-task slots (T4+T5): {late}/{total} "
            f"({100.0 * late / total:.1f}%)."
        )
    missing = [TASK_NAMES[str(i)] for i in range(1, 6) if str(i) not in task_hist]
    if missing:
        lines.append("Absent final-pool tasks: " + ", ".join(missing) + ".")
    lines.append(
        "This is expected for FIFO pool replacement: final retention is dominated "
        "by the latest tasks unless UPDATEMODELPOOL refines/skips redundant "
        "same-task segments."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text())
    report = build_report(payload, args.json_path)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
