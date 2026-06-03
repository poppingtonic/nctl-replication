from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from optuna_mlflow_sweep import (
    SweepRunConfig,
    _run_seed_with_tee,
    _SeedRunResult,
    aggregate_seed_payloads,
    build_run_command,
    grid_search_space,
    load_search_space,
    params_to_cli_args,
    run_trial_subprocess,
    suggest_params,
    validate_run_config,
    validate_study_args,
)


class FakeTrial:
    def suggest_categorical(self, name, choices):
        return choices[-1]

    def suggest_int(self, name, low, high, step=1, log=False):
        assert not log
        return low + step

    def suggest_float(self, name, low, high, **kwargs):
        assert kwargs.get("log") is False
        return (low + high) / 2.0


def test_load_search_space_accepts_parameters_wrapper(tmp_path: Path) -> None:
    p = tmp_path / "space.json"
    p.write_text(
        json.dumps(
            {
                "parameters": {
                    "pool_alpha": {"type": "float", "low": -1, "high": 1},
                    "pool_reservoir": {"type": "int", "low": 32, "high": 128, "step": 32},
                    "pool_update_policy": {"type": "categorical", "choices": ["fifo", "paper"]},
                    "posterior_temp": {"type": "fixed", "value": 0.0},
                }
            }
        )
    )

    space = load_search_space(p)
    params = suggest_params(FakeTrial(), space)

    assert params == {
        "pool_alpha": 0.0,
        "pool_reservoir": 64,
        "pool_update_policy": "paper",
        "posterior_temp": 0.0,
    }


def test_load_search_space_rejects_unknown_parameter_early(tmp_path: Path) -> None:
    p = tmp_path / "space.json"
    p.write_text(json.dumps({"parameters": {"unknown_param": {"type": "fixed", "value": 1}}}))

    with pytest.raises(ValueError, match="unknown"):
        load_search_space(p)


def test_load_search_space_rejects_invalid_log_step_combo(tmp_path: Path) -> None:
    p = tmp_path / "space.json"
    p.write_text(
        json.dumps(
            {
                "parameters": {
                    "pool_alpha": {
                        "type": "float",
                        "low": 1e-3,
                        "high": 10.0,
                        "step": 0.1,
                        "log": True,
                    }
                }
            }
        )
    )

    with pytest.raises(ValueError, match="log=True"):
        load_search_space(p)




def test_bundled_search_spaces_are_valid_and_paper_replicate_is_in_scope() -> None:
    bundled = [
        "search_space_alpha_beta.json",
        "search_space_alpha_beta_v3.json",
        "search_space_gpu_utilization.json",
        "search_space_gpu_utilization_v2.json",
        "search_space_output_retention.json",
        "search_space_paper_replicate.json",
        "search_space_paper_diagnostic.json",
    ]
    loaded = {name: load_search_space(SCRIPTS_DIR / name) for name in bundled}

    paper = loaded["search_space_paper_replicate.json"]
    diagnostic = loaded["search_space_paper_diagnostic.json"]

    for space in (paper, diagnostic):
        assert space["pool"]["choices"] == [15, 30]
        assert space["posterior_temp"] == {"type": "fixed", "value": 1.0}
        assert space["pool_update_policy"] == {"type": "fixed", "value": "paper"}
        assert space["active_state"] == {"type": "fixed", "value": "per-level"}
        assert space["pool_alpha"]["choices"] == [0.0, 0.2]
        assert space["pool_beta"]["choices"] == [0.0, 0.06]
        assert space["pool_reservoir"]["choices"] == [10, 100]
        assert "output_pool" not in space
        assert "close_task_boundary" not in space

    assert paper["min_segment"] == {"type": "fixed", "value": 16}
    assert diagnostic["min_segment"] == {"type": "fixed", "value": 512}

def test_validate_run_config_rejects_empty_seeds_and_bad_lr(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed"):
        validate_run_config(SweepRunConfig(seeds=(), results_dir=tmp_path))
    with pytest.raises(ValueError, match="learning rate"):
        validate_run_config(SweepRunConfig(lr=0.0, results_dir=tmp_path))


def test_validate_study_args_rejects_non_positive_trials_and_jobs() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        validate_study_args(n_trials=0, n_jobs=1, pruner="median")
    with pytest.raises(ValueError, match="n_jobs"):
        validate_study_args(n_trials=1, n_jobs=0, pruner="median")
    with pytest.raises(ValueError, match="pruner"):
        validate_study_args(n_trials=1, n_jobs=1, pruner="bad")
    with pytest.raises(ValueError, match="sampler"):
        validate_study_args(n_trials=1, n_jobs=1, pruner="median", sampler="bad")



def test_grid_search_space_enumerates_only_variable_parameters() -> None:
    space = load_search_space(SCRIPTS_DIR / "search_space_paper_diagnostic.json")
    grid = grid_search_space(space)

    assert grid == {
        "pool": [15, 30],
        "pool_alpha": [0.0, 0.2],
        "pool_beta": [0.0, 0.06],
        "pool_reservoir": [10, 100],
    }
    n_combinations = 1
    for values in grid.values():
        n_combinations *= len(values)
    assert n_combinations == 16


def test_grid_search_space_rejects_unstepped_float() -> None:
    with pytest.raises(ValueError, match="explicit step"):
        grid_search_space({"pool_alpha": {"type": "float", "low": 0, "high": 1}})


def test_params_to_cli_args_maps_supported_run_split_flags() -> None:
    args = params_to_cli_args(
        {
            "pool": 32,
            "output_pool": 64,
            "min_segment": 2048,
            "posterior_temp": 0.0,
            "pool_update_policy": "paper",
            "pool_alpha": -1.0,
            "pool_beta": 5.0,
            "pool_reservoir": 128,
            "chunk_size": 1024,
            "close_task_boundary": 1,
            "active_state": "per-level",
        }
    )

    assert args == [
        "--pool",
        "32",
        "--output-pool",
        "64",
        "--min-segment",
        "2048",
        "--posterior-temp",
        "0",
        "--pool-update-policy",
        "paper",
        "--pool-alpha",
        "-1",
        "--pool-beta",
        "5",
        "--pool-reservoir",
        "128",
        "--chunk-size",
        "1024",
        "--close-task-boundary",
        "1",
        "--active-state",
        "per-level",
    ]


def test_params_to_cli_args_rejects_unknown_parameters() -> None:
    with pytest.raises(ValueError, match="unknown"):
        params_to_cli_args({"not_a_run_split_arg": 1})


def test_build_run_command_includes_fixed_seed_json_and_trial_params(tmp_path: Path) -> None:
    cfg = SweepRunConfig(
        dataset="mnist",
        seeds=(1,),
        nodes="50-25-1",
        lr=0.001,
        results_dir=tmp_path,
        extra_run_args=("--chunk-size", "128"),
    )
    cmd = build_run_command(
        "python",
        cfg,
        {"min_segment": 2048, "pool_update_policy": "paper"},
        seed=7,
        json_out=tmp_path / "trial.json",
    )

    assert cmd[:3] == ["python", str(SCRIPTS_DIR / "run_split_mnist.py"), "mnist"]
    assert cmd[cmd.index("--seed") + 1] == "7"
    assert cmd[cmd.index("--json-out") + 1] == str(tmp_path / "trial.json")
    assert Path(cmd[cmd.index("--json-out") + 1]).is_absolute()
    assert cmd[cmd.index("--min-segment") + 1] == "2048"
    assert cmd[cmd.index("--pool-update-policy") + 1] == "paper"
    assert cmd[-2:] == ["--chunk-size", "128"]


def test_aggregate_seed_payloads_computes_mean_and_bounds() -> None:
    metrics = aggregate_seed_payloads(
        [
            {"avg_accuracy": 80.0, "avg_forgetting": 20.0, "total_time": 10.0},
            {"avg_accuracy": 84.0, "avg_forgetting": 16.0, "total_time": 12.0},
        ],
        metric="avg_accuracy",
    )

    assert metrics["avg_accuracy_mean"] == pytest.approx(82.0)
    assert metrics["avg_accuracy_min"] == pytest.approx(80.0)
    assert metrics["avg_accuracy_max"] == pytest.approx(84.0)
    assert metrics["avg_forgetting_mean"] == pytest.approx(18.0)
    assert metrics["total_time_mean"] == pytest.approx(11.0)


def test_run_trial_subprocess_calls_seed_callback_and_aggregates(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, cwd, timeout_s, log_path, **kwargs):
        from optuna_mlflow_sweep import _SeedRunResult
        json_path = Path(cmd[cmd.index("--json-out") + 1])
        seed = int(cmd[cmd.index("--seed") + 1])
        json_path.write_text(
            json.dumps(
                {
                    "avg_accuracy": 80.0 + seed,
                    "avg_forgetting": 20.0 - seed,
                    "total_time": 1.0,
                }
            )
        )
        log_path.write_text("ok\n")
        return _SeedRunResult(returncode=0, stdout="ok\n", stderr="", timed_out=False)

    monkeypatch.setattr("optuna_mlflow_sweep._run_seed_with_tee", fake_run)
    seen = []
    cfg = SweepRunConfig(seeds=(1, 2), results_dir=tmp_path)

    result = run_trial_subprocess(3, {"min_segment": 2048}, cfg, seed_callback=lambda *args: seen.append(args))

    assert result.value == pytest.approx(81.5)
    assert result.metrics["avg_accuracy_mean"] == pytest.approx(81.5)
    summary_path = tmp_path / "trial_0003" / "summary.json"
    assert summary_path in result.artifacts
    summary = json.loads(summary_path.read_text())
    assert summary["value"] == pytest.approx(81.5)
    assert summary["seeds"] == [1, 2]
    assert [item[0] for item in seen] == [1, 2]
    assert len(seen[0][2]) == 1
    assert len(seen[1][2]) == 2


def test_run_trial_subprocess_resolves_relative_results_dir(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, cwd, timeout_s, log_path, **kwargs):
        from optuna_mlflow_sweep import _SeedRunResult
        calls.append((cmd, cwd))
        json_path = Path(cmd[cmd.index("--json-out") + 1])
        assert json_path.is_absolute()
        assert json_path.parent.exists()
        json_path.write_text(
            json.dumps(
                {
                    "avg_accuracy": 83.0,
                    "avg_forgetting": 17.0,
                    "total_time": 1.0,
                }
            )
        )
        log_path.write_text("ok\n")
        return _SeedRunResult(returncode=0, stdout="ok\n", stderr="", timed_out=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("optuna_mlflow_sweep._run_seed_with_tee", fake_run)
    cfg = SweepRunConfig(seeds=(1,), results_dir=Path("relative_results"))

    result = run_trial_subprocess(0, {"min_segment": 2048}, cfg)

    assert result.value == pytest.approx(83.0)
    assert (tmp_path / "relative_results" / "trial_0000" / "summary.json").exists()
    assert calls[0][1] == SCRIPTS_DIR



def test_validate_run_config_rejects_non_positive_seed_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed_timeout_s"):
        validate_run_config(
            SweepRunConfig(seeds=(1,), results_dir=tmp_path, seed_timeout_s=0.0)
        )
    with pytest.raises(ValueError, match="seed_timeout_s"):
        validate_run_config(
            SweepRunConfig(seeds=(1,), results_dir=tmp_path, seed_timeout_s=-1.0)
        )
    # None means "no limit" and must be accepted.
    validate_run_config(
        SweepRunConfig(seeds=(1,), results_dir=tmp_path, seed_timeout_s=None)
    )


def test_run_trial_subprocess_forwards_seed_timeout(monkeypatch, tmp_path: Path) -> None:
    seen_timeouts: list[float | None] = []

    def fake_run(cmd, cwd, timeout_s, log_path, **kwargs):
        from optuna_mlflow_sweep import _SeedRunResult
        seen_timeouts.append(timeout_s)
        json_path = Path(cmd[cmd.index("--json-out") + 1])
        json_path.write_text(
            json.dumps(
                {"avg_accuracy": 81.0, "avg_forgetting": 19.0, "total_time": 1.0}
            )
        )
        log_path.write_text("ok\n")
        return _SeedRunResult(returncode=0, stdout="ok\n", stderr="", timed_out=False)

    monkeypatch.setattr("optuna_mlflow_sweep._run_seed_with_tee", fake_run)
    cfg = SweepRunConfig(seeds=(1, 2), results_dir=tmp_path, seed_timeout_s=12.5)

    run_trial_subprocess(7, {"min_segment": 2048}, cfg)

    assert seen_timeouts == [12.5, 12.5]


def test_run_trial_subprocess_no_timeout_when_not_set(monkeypatch, tmp_path: Path) -> None:
    seen_timeouts: list[float | None] = []

    def fake_run(cmd, cwd, timeout_s, log_path, **kwargs):
        from optuna_mlflow_sweep import _SeedRunResult
        seen_timeouts.append(timeout_s)
        json_path = Path(cmd[cmd.index("--json-out") + 1])
        json_path.write_text(
            json.dumps(
                {"avg_accuracy": 80.0, "avg_forgetting": 20.0, "total_time": 1.0}
            )
        )
        log_path.write_text("ok\n")
        return _SeedRunResult(returncode=0, stdout="ok\n", stderr="", timed_out=False)

    monkeypatch.setattr("optuna_mlflow_sweep._run_seed_with_tee", fake_run)
    cfg = SweepRunConfig(seeds=(1,), results_dir=tmp_path)

    run_trial_subprocess(0, {"min_segment": 2048}, cfg)

    assert seen_timeouts == [None]


def test_run_trial_subprocess_raises_runtime_error_on_timeout(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, cwd, timeout_s, log_path, **kwargs):
        from optuna_mlflow_sweep import _SeedRunResult
        # Simulate what _run_seed_with_tee writes on timeout: a header line,
        # any captured stdout, then a "TIMEOUT after Xs" footer.
        log_path.write_text(
            f"$ {' '.join(cmd)}\n"
            f"partial-progress\n"
            f"TIMEOUT after {timeout_s:.1f}s; child killed (returncode=-9).\n"
        )
        return _SeedRunResult(
            returncode=-9, stdout="partial-progress\n", stderr="", timed_out=True
        )

    monkeypatch.setattr("optuna_mlflow_sweep._run_seed_with_tee", fake_run)
    cfg = SweepRunConfig(seeds=(1, 2), results_dir=tmp_path, seed_timeout_s=0.1)

    with pytest.raises(RuntimeError, match="timed out"):
        run_trial_subprocess(0, {"min_segment": 2048}, cfg)

    log_path = tmp_path / "trial_0000" / "seed_1.log"
    assert log_path.exists()
    contents = log_path.read_text()
    assert "TIMEOUT" in contents
    assert "partial-progress" in contents
    # The second seed must NOT have been launched after the first timed out.
    assert not (tmp_path / "trial_0000" / "seed_2.log").exists()



def test_run_seed_with_tee_streams_real_subprocess_output_and_reports_returncode(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    result = _run_seed_with_tee(
        [
            sys.executable,
            "-c",
            "import sys; print('alpha', flush=True); print('beta', flush=True); sys.exit(0)",
        ],
        cwd=tmp_path,
        timeout_s=10.0,
        log_path=log,
        stream_to_parent=False,
    )

    assert isinstance(result, _SeedRunResult)
    assert result.returncode == 0
    assert result.timed_out is False
    # Both stdout lines must appear in the tee'd log and in the in-memory buffer.
    assert "alpha" in result.stdout and "beta" in result.stdout
    log_contents = log.read_text()
    assert "alpha" in log_contents
    assert "beta" in log_contents
    # The header (full command line) is the first line of the log.
    assert log_contents.startswith("$ ")


def test_run_seed_with_tee_kills_hung_child_within_deadline(tmp_path: Path) -> None:
    import time as _time
    log = tmp_path / "run.log"
    t0 = _time.monotonic()
    result = _run_seed_with_tee(
        [
            sys.executable,
            "-c",
            "import time, sys; print('before-hang', flush=True); time.sleep(30); sys.exit(0)",
        ],
        cwd=tmp_path,
        timeout_s=0.5,
        log_path=log,
        stream_to_parent=False,
    )
    elapsed = _time.monotonic() - t0

    assert result.timed_out is True
    # Deadline plus the 2s kill grace plus modest scheduler slop.
    assert elapsed < 5.0, f"helper took {elapsed:.2f}s, deadline was 0.5s"
    # The child got far enough to emit one line before being killed.
    assert "before-hang" in result.stdout
    log_contents = log.read_text()
    assert "TIMEOUT after 0.5s" in log_contents
    assert "before-hang" in log_contents


def test_run_seed_with_tee_propagates_non_zero_exit(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    result = _run_seed_with_tee(
        [
            sys.executable,
            "-c",
            "import sys; print('oops', flush=True); sys.exit(7)",
        ],
        cwd=tmp_path,
        timeout_s=10.0,
        log_path=log,
        stream_to_parent=False,
    )

    assert result.returncode == 7
    assert result.timed_out is False
    assert "oops" in result.stdout
