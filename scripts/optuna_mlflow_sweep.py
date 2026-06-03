#!/usr/bin/env python3
"""Optuna + MLflow hyperparameter sweep API for Split-MNIST NCTL runs.

This module is intentionally importable without Optuna or MLflow installed so
unit tests and non-sweep tooling can still use the search-space and command
construction helpers.  The optional dependencies are imported only when a study
is actually executed.

Example:
    python optuna_mlflow_sweep.py mnist \
      --n-trials 20 \
      --seeds 1 2 3 \
      --tracking-uri sqlite:///sweep_results/mlflow.db \
      --experiment split-mnist-nctl \
      --study-name nctl-alpha-beta \
      --storage sqlite:///sweep_results/nctl-alpha-beta.db
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = SCRIPT_DIR / "run_split_mnist.py"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "sweep_results" / "optuna_mlflow"

CLI_FLAG_MAP = {
    "nodes": "--nodes",
    "lr": "--lr",
    "halfspaces": "--halfspaces",
    "pool": "--pool",
    "output_pool": "--output-pool",
    "min_segment": "--min-segment",
    "ptw_depth": "--ptw-depth",
    "posterior_temp": "--posterior-temp",
    "pool_update_policy": "--pool-update-policy",
    "pool_alpha": "--pool-alpha",
    "pool_beta": "--pool-beta",
    "pool_reservoir": "--pool-reservoir",
    "chunk_size": "--chunk-size",
    "close_task_boundary": "--close-task-boundary",
    "active_state": "--active-state",
    "prediction_mode": "--prediction-mode",
}

DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "min_segment": {"type": "categorical", "choices": [1024, 2048, 4096]},
    "posterior_temp": {"type": "categorical", "choices": [0.0]},
    "pool_update_policy": {"type": "categorical", "choices": ["paper"]},
    "pool_alpha": {"type": "categorical", "choices": [-1.0, 0.0, 1.0, 2.0]},
    "pool_beta": {"type": "categorical", "choices": [0.0, 5.0]},
    "pool_reservoir": {"type": "categorical", "choices": [64, 128]},
}


@dataclass(frozen=True)
class SweepRunConfig:
    dataset: str = "mnist"
    seeds: tuple[int, ...] = (1,)
    nodes: str = "50-25-1"
    lr: float = 0.001
    metric: str = "avg_accuracy"
    direction: str = "maximize"
    results_dir: Path = DEFAULT_RESULTS_DIR
    extra_run_args: tuple[str, ...] = ()
    # Per-seed wall-clock timeout in seconds.  When set, a seed run that
    # exceeds this deadline is killed and the trial is marked failed via
    # RuntimeError, which Optuna's catch=(RuntimeError,) handler converts
    # to a failed trial without aborting the study (Phase 5G Commit A).
    seed_timeout_s: float | None = None


@dataclass(frozen=True)
class TrialResult:
    value: float
    metrics: dict[str, float]
    params: dict[str, Any]
    artifacts: list[Path]
    duration_s: float


def validate_run_config(run_config: SweepRunConfig) -> None:
    if not run_config.seeds:
        raise ValueError("at least one seed is required")
    if run_config.direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")
    if not run_config.metric:
        raise ValueError("metric must be non-empty")
    if run_config.lr <= 0:
        raise ValueError("learning rate must be positive")
    if run_config.seed_timeout_s is not None and run_config.seed_timeout_s <= 0:
        raise ValueError("seed_timeout_s must be positive when provided")


def validate_study_args(
    n_trials: int, n_jobs: int, pruner: str, sampler: str = "tpe"
) -> None:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")
    if pruner not in {"median", "none"}:
        raise ValueError(f"unsupported pruner: {pruner}")
    if sampler not in {"tpe", "grid"}:
        raise ValueError(f"unsupported sampler: {sampler}")


def load_search_space(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return validate_search_space(DEFAULT_SEARCH_SPACE)
    payload = json.loads(path.read_text())
    if "parameters" in payload:
        payload = payload["parameters"]
    if not isinstance(payload, dict):
        raise ValueError("search space must be a JSON object or contain a 'parameters' object")
    return validate_search_space(payload)


def validate_search_space(search_space: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    unknown = sorted(set(str(k) for k in search_space) - set(CLI_FLAG_MAP))
    if unknown:
        raise ValueError(f"unknown run_split_mnist search parameters: {unknown}")
    return {str(k): _validate_param_spec(str(k), v) for k, v in search_space.items()}


def _validate_param_spec(name: str, spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"parameter {name!r} spec must be an object")
    typ = spec.get("type")
    if typ == "fixed":
        if "value" not in spec:
            raise ValueError(f"fixed parameter {name!r} needs value")
    elif typ == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"categorical parameter {name!r} needs non-empty choices")
    elif typ == "int":
        if "low" not in spec or "high" not in spec:
            raise ValueError(f"int parameter {name!r} needs low/high")
        if int(spec["low"]) > int(spec["high"]):
            raise ValueError(f"int parameter {name!r} has low > high")
        if int(spec.get("step", 1)) <= 0:
            raise ValueError(f"int parameter {name!r} step must be positive")
        if bool(spec.get("log", False)) and int(spec.get("step", 1)) != 1:
            raise ValueError(f"int parameter {name!r} cannot use log=True with step != 1")
    elif typ == "float":
        if "low" not in spec or "high" not in spec:
            raise ValueError(f"float parameter {name!r} needs low/high")
        if float(spec["low"]) > float(spec["high"]):
            raise ValueError(f"float parameter {name!r} has low > high")
        if "step" in spec and spec["step"] is not None and float(spec["step"]) <= 0:
            raise ValueError(f"float parameter {name!r} step must be positive")
        if bool(spec.get("log", False)) and spec.get("step") is not None:
            raise ValueError(f"float parameter {name!r} cannot use log=True with step")
    else:
        raise ValueError(f"parameter {name!r} has unsupported type {typ!r}")
    return dict(spec)


def suggest_params(trial: Any, search_space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        typ = spec["type"]
        if typ == "fixed":
            params[name] = spec["value"]
        elif typ == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif typ == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif typ == "float":
            kwargs: dict[str, Any] = {"log": bool(spec.get("log", False))}
            if "step" in spec and spec["step"] is not None:
                kwargs["step"] = float(spec["step"])
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                **kwargs,
            )
        else:  # validated earlier; kept for defensive direct calls
            raise ValueError(f"unsupported parameter type {typ!r} for {name!r}")
    return params




def grid_search_space(search_space: Mapping[str, Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Convert a JSON search space to an Optuna GridSampler search space.

    Fixed parameters are omitted because ``suggest_params`` does not call an
    Optuna suggest method for them.  This keeps the grid focused on actual
    variable knobs and avoids duplicate trials in small categorical grids.
    Continuous float grids require an explicit ``step``; otherwise exhaustive
    enumeration is undefined and the caller should use the TPE sampler.
    """
    grid: dict[str, list[Any]] = {}
    for name, spec in validate_search_space(search_space).items():
        typ = spec["type"]
        if typ == "fixed":
            continue
        if typ == "categorical":
            grid[name] = list(spec["choices"])
        elif typ == "int":
            if bool(spec.get("log", False)):
                raise ValueError(f"grid sampler does not support log int parameter {name!r}")
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            grid[name] = list(range(low, high + 1, step))
        elif typ == "float":
            if bool(spec.get("log", False)):
                raise ValueError(f"grid sampler does not support log float parameter {name!r}")
            if "step" not in spec or spec["step"] is None:
                raise ValueError(
                    f"grid sampler needs explicit step for float parameter {name!r}"
                )
            low = float(spec["low"])
            high = float(spec["high"])
            step = float(spec["step"])
            values: list[float] = []
            value = low
            # Include a small tolerance to avoid dropping high due to rounding.
            while value <= high + step * 1e-9:
                values.append(value)
                value += step
            grid[name] = values
        else:  # validated above
            raise ValueError(f"unsupported parameter type {typ!r} for {name!r}")
    return grid


def params_to_cli_args(params: Mapping[str, Any]) -> list[str]:
    args: list[str] = []
    unknown = sorted(set(params) - set(CLI_FLAG_MAP))
    if unknown:
        raise ValueError(f"unknown run_split_mnist parameters: {unknown}")
    for name, value in params.items():
        if value is None:
            continue
        args.extend([CLI_FLAG_MAP[name], _format_cli_value(value)])
    return args


def _format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, ".12g")
    return str(value)


def aggregate_seed_payloads(payloads: list[dict[str, Any]], metric: str) -> dict[str, float]:
    if not payloads:
        raise ValueError("cannot aggregate zero payloads")
    numeric_keys = [
        metric,
        "avg_accuracy",
        "avg_forgetting",
        "total_time",
    ]
    out: dict[str, float] = {}
    for key in numeric_keys:
        values = [float(p[key]) for p in payloads if key in p and p[key] is not None]
        if values:
            out[f"{key}_mean"] = statistics.mean(values)
            if len(values) > 1:
                out[f"{key}_stdev"] = statistics.stdev(values)
            out[f"{key}_min"] = min(values)
            out[f"{key}_max"] = max(values)
    if f"{metric}_mean" not in out:
        raise KeyError(f"metric {metric!r} was not found in any trial payload")
    return out


def build_run_command(
    python_exe: str,
    run_config: SweepRunConfig,
    params: Mapping[str, Any],
    seed: int,
    json_out: Path,
) -> list[str]:
    return [
        python_exe,
        str(RUN_SCRIPT),
        run_config.dataset,
        "--seed",
        str(seed),
        "--nodes",
        run_config.nodes,
        "--lr",
        _format_cli_value(run_config.lr),
        "--json-out",
        str(json_out.resolve() if not json_out.is_absolute() else json_out),
        "--quiet",
        *params_to_cli_args(params),
        *run_config.extra_run_args,
    ]


@dataclass
class _SeedRunResult:
    """Lightweight subprocess-result surface for the Phase 5G Commit B
    Popen+tee runner.  Mirrors the subset of ``subprocess.CompletedProcess`` that
    ``run_trial_subprocess`` consumes, plus a ``timed_out`` flag so the caller
    can distinguish a hard kill from a non-zero return.
    """
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _run_seed_with_tee(
    cmd: list[str],
    cwd: Path,
    timeout_s: float | None,
    log_path: Path,
    *,
    stream_to_parent: bool = True,
    kill_grace_s: float = 2.0,
) -> _SeedRunResult:
    """Run ``cmd`` under ``Popen`` with line-buffered stdout/stderr, tee'd to
    ``log_path`` (and optionally to the parent process's stdout) so a long
    running GPU trial's startup banner / per-task progress shows up in real
    time instead of being held in a ``capture_output=True`` buffer until the
    child exits.

    When ``timeout_s`` is set and exceeded:
      1. SIGTERM the child, wait ``kill_grace_s`` seconds for graceful exit.
      2. SIGKILL if it's still alive.
      3. Return with ``timed_out=True``; the caller is responsible for
         raising ``RuntimeError`` and writing the final TIMEOUT-tagged log.

    The implementation merges stderr into stdout (``stderr=STDOUT``) so the
    tee preserves line ordering as the operator would see it on a real
    terminal; ``_SeedRunResult.stderr`` is always ``""`` on this code path
    (kept on the dataclass for shape parity with subprocess.CompletedProcess).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"$ {' '.join(cmd)}\n"
    buf: list[str] = []
    timed_out = False
    deadline = (time.monotonic() + timeout_s) if timeout_s is not None else None

    # bufsize=1 + universal_newlines/text=True gives line-buffered text I/O.
    # stderr=STDOUT preserves on-the-wire ordering for the operator log so a
    # traceback isn't interleaved out-of-order with stdout progress lines.
    with open(log_path, "w", buffering=1, encoding="utf-8") as log_f:
        log_f.write(header)
        log_f.flush()
        proc = subprocess.Popen(  # noqa: S603 (cmd is a list[str], not shell)
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        # Use select() instead of readline() so the deadline check below can
        # fire even when the child is silent.  Without this the deadline can
        # never trip because readline() blocks until either a newline or
        # EOF arrives -- which is exactly the hung-child scenario this
        # helper exists to bound.
        import select as _select
        stdout_fd = proc.stdout.fileno()
        pending = ""  # carry-over for partial lines between selects
        POLL_INTERVAL_S = 0.1
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                remaining = (
                    deadline - time.monotonic() if deadline is not None else None
                )
                wait_for = (
                    min(POLL_INTERVAL_S, remaining)
                    if remaining is not None
                    else POLL_INTERVAL_S
                )
                if wait_for <= 0:
                    timed_out = True
                    break
                ready, _, _ = _select.select([stdout_fd], [], [], wait_for)
                if not ready:
                    # No data arrived in this poll window; loop back to
                    # re-check the deadline.  Also confirm the child is
                    # still alive so we don't spin forever on a zombie.
                    if proc.poll() is not None:
                        # Drain any final bytes that became available
                        # between the select() and the poll() race.
                        final = proc.stdout.read()
                        if final:
                            pending += final
                        break
                    continue
                # Read whatever's available; os.read on the underlying fd
                # is the non-blocking primitive we want here.
                import os as _os
                try:
                    chunk = _os.read(stdout_fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    # Child closed stdout (EOF).
                    break
                pending += chunk.decode("utf-8", errors="replace")
                # Emit complete lines; carry the partial tail to the next
                # iteration so we don't break a line mid-write.
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    line = line + "\n"
                    buf.append(line)
                    log_f.write(line)
                    if stream_to_parent:
                        sys.stdout.write(line)
                        sys.stdout.flush()
            # Emit any trailing partial line (no terminating newline)
            # captured before EOF / kill.
            if pending:
                buf.append(pending)
                log_f.write(pending)
                if stream_to_parent:
                    sys.stdout.write(pending)
                    sys.stdout.flush()
        finally:
            if timed_out and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=kill_grace_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            else:
                # Normal-exit path: wait for child to fully reap so
                # returncode is populated.
                proc.wait()
            # Close the stdout pipe explicitly so the GC doesn't emit a
            # ResourceWarning when the Popen object is collected later.
            if proc.stdout is not None:
                with contextlib.suppress(OSError):
                    proc.stdout.close()
        if timed_out:
            footer = (
                f"TIMEOUT after {timeout_s:.1f}s; "
                f"child killed (returncode={proc.returncode}).\n"
            )
            log_f.write(footer)
            log_f.flush()

    return _SeedRunResult(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout="".join(buf),
        stderr="",
        timed_out=timed_out,
    )


def run_trial_subprocess(
    trial_number: int,
    params: Mapping[str, Any],
    run_config: SweepRunConfig,
    python_exe: str = sys.executable,
    seed_callback: Any | None = None,
) -> TrialResult:
    validate_run_config(run_config)
    t0 = time.time()
    trial_dir = (Path.cwd() / run_config.results_dir).resolve() if not run_config.results_dir.is_absolute() else run_config.results_dir
    trial_dir = trial_dir / f"trial_{trial_number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    artifacts: list[Path] = []

    (trial_dir / "params.json").write_text(json.dumps(dict(params), indent=2, sort_keys=True))
    artifacts.append(trial_dir / "params.json")

    for seed in run_config.seeds:
        json_out = trial_dir / f"seed_{seed}.json"
        log_out = trial_dir / f"seed_{seed}.log"
        cmd = build_run_command(python_exe, run_config, params, seed, json_out)
        # Phase 5G Commit B: stream child stdout to both the per-seed log and
        # the parent stdout so the operator sees the startup banner / per-task
        # progress in real time.  Hung children are SIGTERM/SIGKILLed when
        # ``seed_timeout_s`` elapses, and the helper returns ``timed_out=True``.
        proc = _run_seed_with_tee(
            cmd,
            cwd=SCRIPT_DIR,
            timeout_s=run_config.seed_timeout_s,
            log_path=log_out,
        )
        artifacts.extend([json_out, log_out])
        if proc.timed_out:
            raise RuntimeError(
                f"trial {trial_number} seed {seed} timed out after "
                f"{run_config.seed_timeout_s:.1f}s; see {log_out}"
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"trial {trial_number} seed {seed} failed with return code "
                f"{proc.returncode}; see {log_out}"
            )
        if not json_out.exists():
            raise RuntimeError(f"trial {trial_number} seed {seed} did not write {json_out}")
        payload = json.loads(json_out.read_text())
        payloads.append(payload)
        if seed_callback is not None:
            seed_callback(seed, payload, list(payloads))

    metrics = aggregate_seed_payloads(payloads, run_config.metric)
    value = metrics[f"{run_config.metric}_mean"]
    duration_s = time.time() - t0
    summary = {
        "trial_number": trial_number,
        "value": value,
        "metric": run_config.metric,
        "direction": run_config.direction,
        "params": dict(params),
        "metrics": metrics,
        "seeds": list(run_config.seeds),
        "duration_s": duration_s,
    }
    summary_path = trial_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    artifacts.append(summary_path)
    return TrialResult(
        value=value,
        metrics=metrics,
        params=dict(params),
        artifacts=artifacts,
        duration_s=duration_s,
    )


def run_study(
    search_space: Mapping[str, Mapping[str, Any]],
    run_config: SweepRunConfig,
    *,
    n_trials: int,
    study_name: str,
    storage: str | None = None,
    tracking_uri: str | None = None,
    experiment: str = "split-mnist-nctl",
    sampler_seed: int | None = None,
    sampler: str = "tpe",
    timeout: int | None = None,
    n_jobs: int = 1,
    pruner: str = "median",
    fail_fast: bool = False,
):
    validate_run_config(run_config)
    validate_study_args(n_trials=n_trials, n_jobs=n_jobs, pruner=pruner, sampler=sampler)
    search_space = validate_search_space(search_space)

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is required to run studies; install optuna") from exc
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("mlflow is required to run studies; install mlflow") from exc

    run_config.results_dir.mkdir(parents=True, exist_ok=True)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    if sampler == "tpe":
        sampler_obj = (
            optuna.samplers.TPESampler(seed=sampler_seed)
            if sampler_seed is not None
            else None
        )
    elif sampler == "grid":
        sampler_obj = optuna.samplers.GridSampler(
            grid_search_space(search_space), seed=sampler_seed
        )
    else:
        raise ValueError(f"unsupported sampler: {sampler}")

    if pruner == "none":
        pruner_obj = optuna.pruners.NopPruner()
    elif pruner == "median":
        pruner_obj = optuna.pruners.MedianPruner()
    else:
        raise ValueError(f"unsupported pruner: {pruner}")
    study = optuna.create_study(
        study_name=study_name,
        direction=run_config.direction,
        storage=storage,
        load_if_exists=True,
        sampler=sampler_obj,
        pruner=pruner_obj,
    )

    def objective(trial):
        params = suggest_params(trial, search_space)
        with mlflow.start_run(run_name=f"trial-{trial.number:04d}"):
            mlflow.set_tags(
                {
                    "study_name": study_name,
                    "dataset": run_config.dataset,
                    "metric": run_config.metric,
                    "direction": run_config.direction,
                    "seeds": ",".join(str(s) for s in run_config.seeds),
                }
            )
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("nodes", run_config.nodes)
            mlflow.log_param("lr", run_config.lr)
            mlflow.log_params(params)

            def on_seed_result(seed: int, payload: dict[str, Any], seen: list[dict[str, Any]]) -> None:
                step = len(seen)
                if run_config.metric in payload:
                    interim = statistics.mean(float(p[run_config.metric]) for p in seen)
                    trial.report(interim, step=step)
                    mlflow.log_metric(f"seed_{run_config.metric}", float(payload[run_config.metric]), step=step)
                    mlflow.log_metric(f"interim_{run_config.metric}_mean", float(interim), step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned(f"pruned after seed {seed}")
                for key in ("avg_accuracy", "avg_forgetting", "total_time"):
                    if key in payload:
                        mlflow.log_metric(f"seed_{key}", float(payload[key]), step=step)

            result = run_trial_subprocess(
                trial.number, params, run_config, seed_callback=on_seed_result
            )
            for key, value in result.metrics.items():
                mlflow.log_metric(key, float(value))
            mlflow.log_metric("objective", float(result.value))
            mlflow.log_metric("duration_s", float(result.duration_s))
            for artifact in result.artifacts:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact), artifact_path=f"trial_{trial.number:04d}")
            trial.set_user_attr("metrics", result.metrics)
            trial.set_user_attr("artifacts", [str(p) for p in result.artifacts])
            return result.value

    catch = () if fail_fast else (RuntimeError,)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        catch=catch,
    )
    return study


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="mnist")
    parser.add_argument("--search-space", type=Path, default=None)
    parser.add_argument("--print-default-search-space", action="store_true")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--nodes", default="50-25-1")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--metric", default="avg_accuracy")
    parser.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--study-name", default="nctl-optuna")
    parser.add_argument("--storage", default=None, help="Optuna storage, e.g. sqlite:///sweep_results/study.db")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI, e.g. file:./mlruns")
    parser.add_argument("--experiment", default="split-mnist-nctl")
    parser.add_argument("--sampler-seed", type=int, default=None)
    parser.add_argument("--sampler", choices=["tpe", "grid"], default="tpe")
    parser.add_argument("--pruner", choices=["median", "none"], default="median")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the study on the first failed subprocess trial instead of marking it failed and continuing.",
    )
    parser.add_argument(
        "--seed-timeout-s",
        type=float,
        default=None,
        help=(
            "Per-seed wall-clock timeout in seconds.  When a child run exceeds "
            "this it is SIGKILLed and the trial is marked failed; Optuna's "
            "catch=(RuntimeError,) keeps the study alive so other trials "
            "still run.  Use to bound hung GPU trials in long sweeps."
        ),
    )
    parser.add_argument(
        "--run-arg",
        action="append",
        default=[],
        help="Extra literal argument forwarded to run_split_mnist.py; repeat for each token.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.print_default_search_space:
        print(json.dumps({"parameters": DEFAULT_SEARCH_SPACE}, indent=2))
        return 0

    search_space = load_search_space(args.search_space)
    run_config = SweepRunConfig(
        dataset=args.dataset,
        seeds=tuple(args.seeds),
        nodes=args.nodes,
        lr=args.lr,
        metric=args.metric,
        direction=args.direction,
        results_dir=args.results_dir,
        extra_run_args=tuple(args.run_arg),
        seed_timeout_s=args.seed_timeout_s,
    )
    study = run_study(
        search_space,
        run_config,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        sampler_seed=args.sampler_seed,
        sampler=args.sampler,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        pruner=args.pruner,
        fail_fast=args.fail_fast,
    )
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print(json.dumps(study.best_trial.params, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
