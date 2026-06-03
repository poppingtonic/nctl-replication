# Neural Combinatorial Transfer Learning (NCTL) Parameter Reference

This document inventories the parameters used by the GPU Split-MNIST /
Split-Fashion-MNIST Neural Combinatorial Transfer Learning (NCTL) runner and
the sweep tooling around it.  It is meant to make future hyperparameter
searches portable to other continual-learning environments without
re-discovering which knobs affect the algorithm, which knobs only affect
measurement, and which knobs are diagnostics.

For a visual map of the runner, network, CUDA layer, pool, and sweep outputs,
see `docs/nctl-architecture.md`.

The direct benchmark entry point is:

```bash
python3 scripts/run_split_mnist.py mnist
```

The Optuna/MLflow wrapper is:

```bash
python3 scripts/optuna_mlflow_sweep.py mnist --search-space SEARCH.json
```

## Current Reference Recipe

The strongest task-free recipe found during the replication effort is:

```bash
python3 scripts/run_split_mnist.py mnist \
  --nodes 50-25-1 --lr 0.001 \
  --min-segment 512 --pool 80 --pool-reservoir 10 \
  --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
  --pool-evict-policy age-diversity-oldest-floor --pool-oldest-floor 6 \
  --active-state per-level --prediction-mode ptw_dp \
  --chunk-size 1024 --posterior-temp 1.0 \
  --adapt-n 50 \
  --seed 1 --json-out run.json
```

In `run-03-06-2026-17:47:27.json`, `pool_oldest_floor=6` reached 95.20%
average accuracy with 4.32% average forgetting on seed 1, 0.13 percentage
points above the paper's 95.07 target. The earlier five-seed
`pool_oldest_floor=4` sweep reached 95.043% average accuracy. The recipe is
task-free: it uses insertion indices, not benchmark task IDs.

## Run Parameters

### Dataset and Task Stream

| Parameter | Default | Search-space key | Meaning | Search guidance |
| --- | --- | --- | --- | --- |
| `dataset` | `mnist` | positional, not in Optuna key map | Dataset basename. Supported by the current runner: `mnist`, `fashion-mnist`. | Treat as an environment choice, not a within-study hyperparameter. Use separate studies because task lengths and image statistics differ. |
| task pairs | fixed | not exposed | Current stream is `(0,1),(2,3),(4,5),(6,7),(8,9)`. | For other environments, document task count, per-task lengths, and class pairing because these determine pool pressure. |
| input dimension | `784` | not exposed | Hardcoded for 28x28 image binaries. | New environments need a runner change or adapter if input shape differs. |

### Architecture and Learning

| Parameter | Default | Search-space key | Meaning | Search guidance |
| --- | --- | --- | --- | --- |
| `--nodes` | `50-25-1` | `nodes` | Hyphen-separated layer sizes. The output layer should end in `1` for binary tasks. | Scale with environment complexity and GPU memory. Larger layers multiply pool memory because each node owns a snapshot pool. |
| `--halfspaces` | `4` | `halfspaces` | Number of random halfspaces per node. The Gated Geometric Mixer (GGM) cell count is `C = 2 ** halfspaces`. | Memory and compute double with each increment. Search small categorical sets, e.g. `[3, 4, 5]`, after temporal/pool settings are stable. |
| `--lr` | `0.001` | `lr` | Per-step GGM weight learning rate. | Use log-scale floats around the reference value, e.g. `3e-4..3e-3`. Retune for new input normalization or task difficulty. |
| `--seed` | `42` | top-level Optuna arg `--seeds` | Seeds hyperplane initialization and run order artifacts. | Always report multi-seed means. Single-seed deltas can move by several tenths of a point. |

### FMN Temporal State

| Parameter | Default | Search-space key | Meaning | Search guidance |
| --- | --- | --- | --- | --- |
| `--min-segment` | `32` | `min_segment` | Minimum closed segment length admitted to the model pool. This corresponds to the paper's short-segment skip threshold. | Primary environment-scaling knob. Express it as a target number of closes per task: approximately `task_len / min_segment`. |
| `--ptw-depth` | `15` | `ptw_depth` | PTW/FMN binary depth. Bounds the maximum sequence length represented by the binary temporal hierarchy. | Must cover the stream length: `2 ** ptw_depth` should exceed total samples processed. |
| `--active-state` | `flat` | `active_state` | Active FMN segment state layout. `per-level` gives each retained PTW level independent active mixture/reservoir state. | Use `per-level` for paper-fidelity runs. |
| `--prediction-mode` | `selected_level` | `prediction_mode` | Prediction rule. `ptw_dp` uses the paper Algorithm 1 bottom-up PTW dynamic program; `selected_level` returns one selected level's conditional. | Use `ptw_dp` with `--active-state per-level` for paper-fidelity runs. |
| `--chunk-size` | `256` | `chunk_size` | Number of training samples per Python-to-kernel chunk. | Mostly a performance/memory knob. Keep fixed within accuracy sweeps unless validating chunk invariance. |
| `--close-task-boundary` | `0` | `close_task_boundary` | If `1`, force-closes the open FMN segment after each training task. | This uses task boundary knowledge. Treat as an ablation unless the target environment exposes boundaries. |

### Pool Capacity and Retention

| Parameter | Default | Search-space key | Meaning | Search guidance |
| --- | --- | --- | --- | --- |
| `--pool` | `8` | `pool` | Per-node model-pool capacity for all layers unless overridden by `--output-pool`. This is the FMN `k` memory budget. | Primary memory/accuracy knob. Size relative to admitted closes per task and number of tasks. |
| `--output-pool` | same as `--pool` | `output_pool` | Optional pool capacity for the final layer only. | Useful when output-layer retention is the bottleneck but hidden-layer pools are too expensive. |
| `--pool-update-policy` | `fifo` | `pool_update_policy` | Pool admission/update rule. `paper` enables the alpha/beta refine/skip/add heuristic; `fifo` appends/evicts without that heuristic. | Use `paper` for paper-fidelity sweeps. |
| `--pool-alpha` | `inf` | `pool_alpha` | Threshold used by `pool_update_policy=paper` to decide whether an active segment should refine an existing pool model. | Values used in successful Split-MNIST runs include `0.0`. Search jointly with `pool_beta`. |
| `--pool-beta` | `inf` | `pool_beta` | Threshold used by `pool_update_policy=paper` to decide whether to skip/add a segment to the pool. | Values used in successful Split-MNIST runs include `0.0`. |
| `--pool-reservoir` | `64` | `pool_reservoir` | Reservoir size for the paper-style pool update heuristic. | Split-MNIST reference task-free runs used `10`. Search `[10, 32, 64, 128]` when adapting to new task lengths. |
| `--pool-evict-policy` | `fifo` | use `--run-arg` in Optuna currently | Which slot is overwritten when a full pool accepts a new snapshot. Choices: `fifo`, `task-floor`, `age-diversity`, `age-bucket-floor`, `age-diversity-oldest-floor`. | For task-free paper-fidelity candidates, prefer `age-diversity-oldest-floor`. `task-floor` is diagnostic only because it uses task IDs. |
| `--pool-oldest-floor` | `2` | use `--run-arg` in Optuna currently | For `age-diversity-oldest-floor`, protects the oldest `k` slots before applying age-diversity to the remainder. Inert for other policies. | Tunes early-task retention. Split-MNIST floor `2` reached 94.60%; floor `4` reached 95.04% across five seeds; floor `6` reached 95.20% with 4.32% forgetting on the recorded seed-1 run. |

Eviction policies:

- `fifo`: drops slot 0, the oldest snapshot.
- `task-floor`: protects the sole surviving snapshot for each task ID. This
  identified retention as the lever but leaks benchmark task labels.
- `age-diversity`: evicts the insertion index with the closest temporal
  neighbour; ties evict the oldest tied slot.
- `age-bucket-floor`: protects log-age buckets, then evicts FIFO within an
  overfull bucket. On Split-MNIST it degenerated to FIFO in the tested regime.
- `age-diversity-oldest-floor`: protects the oldest `pool_oldest_floor` slots,
  then applies age-diversity. This is the current best task-free retention rule.

### Posterior and Mixture Weighting

| Parameter | Default | Search-space key | Meaning | Search guidance |
| --- | --- | --- | --- | --- |
| `--posterior-temp` | `1.0` | `posterior_temp` | Temperature for current-segment posterior weighting. `1.0` is the standard Bayesian conditional; lower values flatten or damp posterior saturation. | Keep `1.0` for current paper-fidelity runs. Use temperature sweeps only to diagnose posterior saturation or unstable long segments. |

### Evaluation

| Parameter | Default | Search-space key | Meaning | Search guidance |
| --- | --- | --- | --- | --- |
| `--adapt-n` | `50` | not in Optuna key map | Number of test samples used for per-task adaptation before measuring task accuracy. | Default `50` matches the paper and acceptance recipe. |
| `--eval-suffix-from` | unset | not in Optuna key map | Common evaluation offset. When set, evaluates on `test[eval_suffix_from:]` instead of `test[adapt_n:]`, with values below `adapt_n` raised to `adapt_n`. | Use for fairness when comparing `adapt_n` values. |

### Output, Profiling, and Operator Controls

| Parameter | Default | Search-space key | Meaning | Search guidance |
| --- | --- | --- | --- | --- |
| `--json-out` | unset | generated by sweep tools | Writes run metrics, per-task accuracy matrix, and pool provenance to JSON. | Always set in sweeps. Summary scripts depend on it. |
| `--quiet` | `false` | generated by sweep tools | Suppresses per-task stdout and prints compact JSON when `--json-out` is set. | Use for automated sweeps. |
| `--profile-timings` | unset | not in Optuna key map | Writes per-section timing JSON from the profiler. | Use on short diagnostic runs; it adds wrapper overhead. |
| `--profile-timings-sync-cuda` | `false` | not in Optuna key map | Synchronizes CUDA around profiled sections. | Use only for timing attribution. It can add large overhead. |

## Optuna / MLflow Sweep Parameters

`optuna_mlflow_sweep.py` wraps `run_split_mnist.py` and averages the selected
metric across seeds for each trial.

| Sweep parameter | Default | Meaning |
| --- | --- | --- |
| `--search-space` | built-in default | JSON file with a top-level `parameters` object or a direct parameter object. |
| `--print-default-search-space` | off | Prints the built-in Optuna search space. |
| `--n-trials` | `20` | Number of Optuna trials. For grid sampling, must cover the full grid to exhaust it. |
| `--timeout` | unset | Study-level wall-clock timeout in seconds. |
| `--n-jobs` | `1` | Parallel Optuna trials. Use carefully on a single GPU. |
| `--seeds` | `1` | Seeds run per trial. Use multiple seeds before trusting small deltas. |
| `--nodes` | `50-25-1` | Fixed architecture forwarded to every trial. |
| `--lr` | `0.001` | Fixed learning rate forwarded to every trial. |
| `--metric` | `avg_accuracy` | Trial objective key from the run JSON. |
| `--direction` | `maximize` | Optuna objective direction. |
| `--results-dir` | `scripts/sweep_results/optuna_mlflow` | Child JSON/log/artifact output directory. |
| `--study-name` | `nctl-optuna` | Optuna/MLflow study name. |
| `--storage` | unset | Optional Optuna storage, e.g. SQLite. |
| `--tracking-uri` | unset | Optional MLflow tracking URI. |
| `--experiment` | `split-mnist-nctl` | MLflow experiment name. |
| `--sampler-seed` | unset | Seed for the Optuna sampler. |
| `--sampler` | `tpe` | `tpe` or `grid`. |
| `--pruner` | `median` | `median` or `none`. |
| `--fail-fast` | off | Abort on first failed child run instead of marking trial failed. |
| `--seed-timeout-s` | unset | Per-seed subprocess timeout. Useful for hung GPU trials. |
| `--run-arg` | repeatable | Extra literal tokens forwarded to `run_split_mnist.py`. Use for runner flags not yet in the JSON key map, such as `--pool-evict-policy`, `--pool-oldest-floor`, `--adapt-n`, or profiling flags. |

Supported JSON search-space keys:

```text
nodes, lr, halfspaces, pool, output_pool, min_segment, ptw_depth,
posterior_temp, pool_update_policy, pool_alpha, pool_beta, pool_reservoir,
chunk_size, close_task_boundary, active_state, prediction_mode
```

Other runner parameters must currently be supplied with repeated `--run-arg`
tokens:

```bash
python3 scripts/optuna_mlflow_sweep.py mnist \
  --search-space search.json \
  --run-arg --pool-evict-policy --run-arg age-diversity-oldest-floor \
  --run-arg --pool-oldest-floor --run-arg 6
```

## Designing Searches for Other Environments

Start with environment measurements before broad random search:

1. Record task count, task order, and per-task sample counts.
2. Choose `ptw_depth` so `2 ** ptw_depth` exceeds total online samples.
3. Choose a `min_segment` ladder by closes per task, not by absolute Split-MNIST
   values.  A useful first ladder is roughly 5, 10, 20, and 40 closes per task.
4. Choose `pool` relative to admitted closes and number of tasks.  If the pool
   cannot retain at least a few early-task snapshots, eviction policy dominates.
5. Fix paper-fidelity mechanics first: `active_state=per-level`,
   `prediction_mode=ptw_dp`, `pool_update_policy=paper`, and no task-ID-based
   eviction.
6. Run a small multi-seed control before interpreting deltas.

Recommended first-pass families:

- Memory/retention: sweep `min_segment`, `pool`, `pool_reservoir`, and the
  task-free eviction policy.
- Old-task retention: hold the best memory settings fixed and sweep
  `pool_oldest_floor` over `{0, 2, 4, 6, 8}`.
- Architecture: after temporal/pool settings are stable, sweep `nodes`,
  `halfspaces`, and `lr`.

## Paper-Fidelity Notes

Paper-faithful or intended candidate settings:

- `pool_update_policy=paper`
- `active_state=per-level`
- `prediction_mode=ptw_dp`
- task-free eviction, preferably `age-diversity-oldest-floor`
- `posterior_temp=1.0` unless explicitly diagnosing saturation

Diagnostic or environment-leaking settings:

- `pool_evict_policy=task-floor` uses benchmark task IDs and should not be
  reported as a paper-faithful result.
- `close_task_boundary=1` uses task boundary knowledge. Treat it as an ablation
  unless the target deployment supplies boundaries.
- `prediction_mode=selected_level` and `active_state=flat` are legacy
  approximations useful for regression checks, not the preferred NCTL recipe.

## Metrics to Inspect

Every serious search should record:

- `avg_accuracy` and `avg_forgetting`
- final per-task accuracy from `acc_matrix`
- `pool_provenance.task_histogram` / `pool_task_histogram`
- `pool_insert_indices` and `slot_task_histograms` for retention diagnostics
- per-seed values, not only means

Mean accuracy can hide the failure mode that dominated the Split-MNIST paper
gap: early-task starvation.  Always inspect the first task's final accuracy
when changing pool capacity, `min_segment`, or eviction policy.
