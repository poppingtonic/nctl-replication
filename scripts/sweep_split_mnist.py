#!/usr/bin/env python3
"""Drive run_split_mnist.py across the planned NCTL hyperparameter sweep.

Default sweep follows the runbook in the agent notes for
forget-me-not-h2u.30.8: 5 seeds × {paper defaults, min_segment=4096,
min_segment=4096 + pool=32}.  Each run emits a JSON file; this driver
aggregates them and prints a comparison table.

Usage:
    python sweep_split_mnist.py mnist            # full default sweep
    python sweep_split_mnist.py mnist --quick    # 2 seeds × default config only
    python sweep_split_mnist.py mnist --configs ms32-p8 ms4096-p8

Each run is invoked as a subprocess so a kernel crash in one configuration
does not poison the others; subprocess.run uses check=False for the same
reason and the failure is recorded in the summary table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = SCRIPT_DIR / "run_split_mnist.py"
RESULTS_DIR = SCRIPT_DIR / "sweep_results"


@dataclass
class Config:
    name: str
    extra_args: list[str]


DEFAULT_CONFIGS = {
    # Baseline + paper-style hyperparameters
    "ms32-p8": Config("ms32-p8", []),
    "ms4096-p8": Config("ms4096-p8", ["--min-segment", "4096"]),
    "ms4096-p32": Config("ms4096-p32", ["--min-segment", "4096", "--pool", "32"]),
    "ms8192-p32": Config("ms8192-p32", ["--min-segment", "8192", "--pool", "32"]),
    # Sweep 1: smaller min_segment values stress-test bounded-pool churn.
    "ms16-p8": Config("ms16-p8", ["--min-segment", "16"]),
    "ms8-p8": Config("ms8-p8", ["--min-segment", "8"]),
    "ms4-p8": Config("ms4-p8", ["--min-segment", "4"]),
    "ms1-p8": Config("ms1-p8", ["--min-segment", "1"]),
    # Sweep 2: posterior temperature at sparse close, to test whether
    # mechanism C (posterior dilution) explains the ms4096 collapse.
    # ms4096-p8 is the base; varying only the temperature isolates the effect.
    "ms4096-p8-t0.1": Config(
        "ms4096-p8-t0.1",
        ["--min-segment", "4096", "--posterior-temp", "0.1"],
    ),
    "ms4096-p8-t0.01": Config(
        "ms4096-p8-t0.01",
        ["--min-segment", "4096", "--posterior-temp", "0.01"],
    ),
    "ms4096-p8-t0.001": Config(
        "ms4096-p8-t0.001",
        ["--min-segment", "4096", "--posterior-temp", "0.001"],
    ),
    # Combined experiment: paper-style sparse close + low temperature + bigger pool.
    "ms4096-p32-t0.01": Config(
        "ms4096-p32-t0.01",
        [
            "--min-segment", "4096",
            "--pool", "32",
            "--posterior-temp", "0.01",
        ],
    ),
    # Diagnostic configs: posterior-temp 0 collapses the posterior-weighted
    # mixture in the CUDA kernel to 0.5*fresh + 0.5*mean(pool); see
    # _posterior_mixture_prob.  This isolates mixture math from segment
    # boundary and pool-management changes.
    "ms32-p8-t0": Config(
        "ms32-p8-t0",
        ["--posterior-temp", "0"],
    ),
    "ms32-p8-t0.001": Config(
        "ms32-p8-t0.001",
        ["--posterior-temp", "0.001"],
    ),
    "ms32-p8-t0.01": Config(
        "ms32-p8-t0.01",
        ["--posterior-temp", "0.01"],
    ),
    "ms4096-p8-t0": Config(
        "ms4096-p8-t0",
        ["--min-segment", "4096", "--posterior-temp", "0"],
    ),
    "ms4096-p8-t0.0001": Config(
        "ms4096-p8-t0.0001",
        ["--min-segment", "4096", "--posterior-temp", "0.0001"],
    ),
    # Min-segment ladder at posterior_temp=0 (kernel collapses to
    # 0.5*fresh + 0.5*mean(pool)).  Insert-frequency is the only varying axis.
    # Close counts on the actual Split-MNIST task sizes (depth=15) per-task:
    #   ms=256   -> [49, 47, 44, 48, 46]   sum 234
    #   ms=512   -> [24, 24, 22, 24, 23]   sum 117
    #   ms=1024  -> [12, 12, 11, 12, 11]   sum 58
    #   ms=2048  -> [ 6,  6,  5,  6,  6]   sum 29
    # Pool capacity 8 saturates only above ~ms=1024 here.
    "ms256-p8-t0": Config(
        "ms256-p8-t0",
        ["--min-segment", "256", "--posterior-temp", "0"],
    ),
    "ms512-p8-t0": Config(
        "ms512-p8-t0",
        ["--min-segment", "512", "--posterior-temp", "0"],
    ),
    "ms1024-p8-t0": Config(
        "ms1024-p8-t0",
        ["--min-segment", "1024", "--posterior-temp", "0"],
    ),
    "ms2048-p8-t0": Config(
        "ms2048-p8-t0",
        ["--min-segment", "2048", "--posterior-temp", "0"],
    ),
    # Two t=1 controls at the same min_segment so we can A/B
    # mixture-math vs insert-frequency cleanly on the GPU host.
    "ms256-p8": Config(
        "ms256-p8",
        ["--min-segment", "256"],
    ),
    "ms2048-p8": Config(
        "ms2048-p8",
        ["--min-segment", "2048"],
    ),
    # Paper-style FMN §3.3 UPDATEMODELPOOL heuristic: refine/skip/add.
    # Thresholds are exposed for tuning because the papers specify the tests
    # but not universal alpha/beta constants for this GPU Split-MNIST setup.
    "ms2048-p8-t0-paper-a0-b0": Config(
        "ms2048-p8-t0-paper-a0-b0",
        [
            "--min-segment", "2048",
            "--posterior-temp", "0",
            "--pool-update-policy", "paper",
            "--pool-alpha", "0",
            "--pool-beta", "0",
        ],
    ),
    "ms2048-p8-t0-paper-a0-b-10": Config(
        "ms2048-p8-t0-paper-a0-b-10",
        [
            "--min-segment", "2048",
            "--posterior-temp", "0",
            "--pool-update-policy", "paper",
            "--pool-alpha", "0",
            "--pool-beta", "-10",
        ],
    ),
    "ms2048-p8-t0-paper-a-5-b0": Config(
        "ms2048-p8-t0-paper-a-5-b0",
        [
            "--min-segment", "2048",
            "--posterior-temp", "0",
            "--pool-update-policy", "paper",
            "--pool-alpha", "-5",
            "--pool-beta", "0",
        ],
    ),
    "ms2048-p8-t0-paper-a5-b0": Config(
        "ms2048-p8-t0-paper-a5-b0",
        [
            "--min-segment", "2048",
            "--posterior-temp", "0",
            "--pool-update-policy", "paper",
            "--pool-alpha", "5",
            "--pool-beta", "0",
        ],
    ),
    "ms2048-p8-t0-paper-a-1-b0": Config(
        "ms2048-p8-t0-paper-a-1-b0",
        [
            "--min-segment", "2048",
            "--posterior-temp", "0",
            "--pool-update-policy", "paper",
            "--pool-alpha", "-1",
            "--pool-beta", "0",
        ],
    ),
    "ms2048-p8-t0-paper-a1-b0": Config(
        "ms2048-p8-t0-paper-a1-b0",
        [
            "--min-segment", "2048",
            "--posterior-temp", "0",
            "--pool-update-policy", "paper",
            "--pool-alpha", "1",
            "--pool-beta", "0",
        ],
    ),
    "ms2048-p8-t0-paper-a2-b0": Config(
        "ms2048-p8-t0-paper-a2-b0",
        [
            "--min-segment", "2048",
            "--posterior-temp", "0",
            "--pool-update-policy", "paper",
            "--pool-alpha", "2",
            "--pool-beta", "0",
        ],
    ),
    "ms2048-p8-t0-paper-a0-b5": Config(
        "ms2048-p8-t0-paper-a0-b5",
        [
            "--min-segment", "2048",
            "--posterior-temp", "0",
            "--pool-update-policy", "paper",
            "--pool-alpha", "0",
            "--pool-beta", "5",
        ],
    ),
    "ms2048-p8-t0-paper-a0-b10": Config(
        "ms2048-p8-t0-paper-a0-b10",
        [
            "--min-segment", "2048",
            "--posterior-temp", "0",
            "--pool-update-policy", "paper",
            "--pool-alpha", "0",
            "--pool-beta", "10",
        ],
    ),
}


def run_one(
    dataset: str,
    seed: int,
    cfg: Config,
    base_args: list[str],
    json_path: Path,
    log_path: Path,
) -> dict:
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        dataset,
        "--seed",
        str(seed),
        "--json-out",
        str(json_path),
        "--quiet",
        *base_args,
        *cfg.extra_args,
    ]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
        )
    except FileNotFoundError as exc:
        return {
            "status": "spawn_failed",
            "error": str(exc),
            "seed": seed,
            "config": cfg.name,
            "wall_time": time.time() - t0,
        }

    log_path.write_text(
        f"$ {' '.join(cmd)}\nreturncode={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n"
    )

    if result.returncode != 0 or not json_path.exists():
        return {
            "status": "failed",
            "returncode": result.returncode,
            "seed": seed,
            "config": cfg.name,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "wall_time": time.time() - t0,
        }

    payload = json.loads(json_path.read_text())
    payload["status"] = "ok"
    payload["config"] = cfg.name
    payload["wall_time"] = time.time() - t0
    return payload


def aggregate(runs: list[dict]) -> dict[str, dict]:
    by_config: dict[str, list[dict]] = {}
    for r in runs:
        by_config.setdefault(r["config"], []).append(r)
    summary = {}
    for name, items in by_config.items():
        oks = [r for r in items if r.get("status") == "ok"]
        accs = [r["avg_accuracy"] for r in oks]
        fgts = [r["avg_forgetting"] for r in oks]
        times = [r["total_time"] for r in oks]
        summary[name] = {
            "n_ok": len(oks),
            "n_failed": len(items) - len(oks),
            "avg_accuracy_mean": statistics.mean(accs) if accs else None,
            "avg_accuracy_stdev": statistics.stdev(accs) if len(accs) > 1 else None,
            "avg_accuracy_min": min(accs) if accs else None,
            "avg_accuracy_max": max(accs) if accs else None,
            "avg_forgetting_mean": statistics.mean(fgts) if fgts else None,
            "avg_forgetting_stdev": statistics.stdev(fgts) if len(fgts) > 1 else None,
            "total_time_mean": statistics.mean(times) if times else None,
            "per_seed_accuracy": [(r["seed"], r["avg_accuracy"]) for r in oks],
        }
    return summary


def print_summary(summary: dict[str, dict], dataset: str) -> None:
    print()
    print(f"=== Split-{dataset.upper()} NCTL sweep summary ===")
    print(
        f"{'config':<14} {'n_ok':>4} {'acc mean':>9} {'acc std':>8} "
        f"{'acc min':>8} {'acc max':>8} {'fgt mean':>9} {'time (s)':>9}"
    )
    paper_target = 95.07
    def fmt_pct(v):
        return f"{v:.2f}%" if v is not None else "   --"

    def fmt_num(v, suffix=""):
        return f"{v:.2f}{suffix}" if v is not None else "   --"

    for name, row in summary.items():
        mean = row["avg_accuracy_mean"]
        std = row["avg_accuracy_stdev"]
        gap = paper_target - mean if mean is not None else None
        n_ok = row["n_ok"]
        acc_min = row["avg_accuracy_min"]
        acc_max = row["avg_accuracy_max"]
        fgt_mean = row["avg_forgetting_mean"]
        t_mean = row["total_time_mean"]
        print(
            f"{name:<14} {n_ok:>4} "
            f"{fmt_pct(mean):>9} "
            f"{fmt_num(std):>8} "
            f"{fmt_pct(acc_min):>8} "
            f"{fmt_pct(acc_max):>8} "
            f"{fmt_pct(fgt_mean):>9} "
            f"{fmt_num(t_mean):>9}"
        )
        if gap is not None:
            seeds = row["per_seed_accuracy"]
            print(
                f"{'':<14}    seeds={seeds}  "
                f"gap-to-paper={gap:+.2f} pp"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="mnist")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "ms32-p8",
            "ms16-p8",
            "ms8-p8",
            "ms4-p8",
            "ms1-p8",
            "ms4096-p8-t0.1",
            "ms4096-p8-t0.01",
            "ms4096-p8-t0.001",
            "ms4096-p32-t0.01",
        ],
        choices=sorted(DEFAULT_CONFIGS.keys()),
        help="Subset of the default sweep to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorthand for --seeds 1 2 --configs ms32-p8",
    )
    parser.add_argument(
        "--nodes",
        default="50-25-1",
        help="Forwarded to run_split_mnist.py.",
    )
    parser.add_argument("--lr", default="0.001")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Per-run JSON and stdout/stderr are written here.",
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds = [1, 2]
        args.configs = ["ms32-p8"]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    base_args = ["--nodes", args.nodes, "--lr", args.lr]
    runs = []
    plan = [(seed, DEFAULT_CONFIGS[name]) for name in args.configs for seed in args.seeds]
    print(
        f"Planned {len(plan)} runs across {len(args.configs)} config(s) "
        f"× {len(args.seeds)} seed(s); results in {args.results_dir}",
        flush=True,
    )

    for i, (seed, cfg) in enumerate(plan, 1):
        stamp = f"{args.dataset}_{cfg.name}_seed{seed}"
        json_path = args.results_dir / f"{stamp}.json"
        log_path = args.results_dir / f"{stamp}.log"
        print(
            f"[{i}/{len(plan)}] config={cfg.name:<12} seed={seed} ... ",
            end="",
            flush=True,
        )
        r = run_one(args.dataset, seed, cfg, base_args, json_path, log_path)
        runs.append(r)
        if r["status"] == "ok":
            print(
                f"acc={r['avg_accuracy']:.2f}%  fgt={r['avg_forgetting']:.2f}%  "
                f"({r['wall_time']:.1f}s)"
            )
        else:
            print(f"FAILED ({r['status']}); see {log_path}")

    summary = aggregate(runs)
    print_summary(summary, args.dataset)
    summary_path = args.results_dir / f"{args.dataset}_summary.json"
    summary_path.write_text(json.dumps({"runs": runs, "summary": summary}, indent=2))
    print(f"\nFull summary written to {summary_path}")


if __name__ == "__main__":
    main()
