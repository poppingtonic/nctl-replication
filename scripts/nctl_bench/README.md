# NCTL Benchmark Implementation

This package contains the GPU implementation used for the Neural Combinatorial Transfer Learning (NCTL) Split-MNIST paper-replication effort in this repository.
It is the implementation layer underneath `scripts/run_split_mnist.py`:
the runner builds an `NctlNetwork`, streams the five binary Split-MNIST tasks
online, evaluates per-task adaptation/forgetting, and writes JSON artifacts for
sweep summarizers.

The current research target is the NCTL paper's Split-MNIST result:

```text
95.07% average accuracy
```

The strict reproduction path (task-free, without leaking task ids) is close but not yet formally closed:
the strongest five-seed task-free result recorded so far is
`age-diversity-oldest-floor` with `pool_oldest_floor=4`, at `95.043%` average
accuracy, 0.027 percentage points below the paper target.

## Where This Fits

```text
run_split_mnist.py
  -> NctlNetwork
     -> CudaFmnMixtureLayer stack
        -> CUDA kernels for GGM (Gated Geometric Mixer) / FMN mixture updates
        -> bounded per-node model pools
        -> PTW dynamic-programming prediction
  -> JSON metrics and pool provenance
  -> sweep_results/* summary scripts
```

For a Mermaid diagram of the architecture, see
`../../docs/nctl-architecture.md`.  For the full parameter inventory and
search guidance, see `../../docs/nctl-parameter-reference.md`.

## Important Files

- `nctl_network.py` - Main implementation. Defines `NctlNetwork`,
  `CudaFmnMixtureLayer`, segment-close bookkeeping, PTW prediction modes,
  pool update/eviction policies, serialization, snapshot/restore for
  evaluation, and provenance summaries.
- `ggm_layer.py` - Standalone/reference Bernoulli Gated Geometric Mixer (GGM)
  layer. The active NCTL
  path does not instantiate this class; equivalent GGM context, prediction,
  and update math is fused into the FMN mixture kernels.
- `gaussian_head.py` - Gaussian head utilities and DeepMind-reference parity
  support for related experiments.
- `fmn_pool.py` - CPU-side FMN/PTW pool structures used by tests and earlier
  implementations.
- `_profile.py` - Lightweight section profiler used by
  `run_split_mnist.py --profile-timings`.
- `cuda/ggm_kernel.cu` - Standalone CUDA GGM kernel. This is not called by the
  active NCTL runner; it remains a reference/legacy kernel for the unfused GGM
  path.
- `cuda/fmn_mixture_kernel.cu` - Earlier single-level FMN mixture kernel with
  inlined GGM math.
- `cuda/fmn_mixture_multilevel_kernel.cu` - Hot path for the current
  multilevel per-level-state NCTL implementation, also with inlined GGM math.
- `cuda/build_kernels.sh` and `cuda/BUILD.md` - Kernel build helpers and notes.

## Paper-Fidelity Path

Use this path when measuring against the NCTL paper rather than running a
diagnostic:

```bash
python3 scripts/run_split_mnist.py mnist \
  --nodes 50-25-1 --lr 0.001 \
  --min-segment 512 --pool 80 --pool-reservoir 10 \
  --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
  --pool-evict-policy age-diversity-oldest-floor --pool-oldest-floor 4 \
  --active-state per-level --prediction-mode ptw_dp \
  --chunk-size 1024 --posterior-temp 1.0 \
  --adapt-n 50 \
  --seed 1 --json-out run.json
```

Key paper-fidelity constraints:

- `active_state=per-level` keeps independent active mixture state for retained
  PTW levels.
- `prediction_mode=ptw_dp` uses the FMN/PTW bottom-up dynamic program instead
  of returning a selected level's conditional.
- `pool_update_policy=paper` enables the paper-style `alpha`/`beta`
  refine/skip/add model-pool heuristic.
- `posterior_temp=1.0` keeps standard Bayesian posterior weighting.
- Eviction must be task-free. `task-floor` is useful diagnostically, but it
  consumes benchmark task IDs and should not be reported as paper-faithful.
- `close_task_boundary=1` is also diagnostic unless the target environment
  exposes task boundaries to the learner.

## Current Retention Story

The main remaining implementation gap has been model-pool retention, not CUDA
throughput or basic network capacity.

The useful sequence of diagnostics was:

1. FIFO eviction reproduced the accepted control: about `90.94%` over five
   seeds with pool 80.
2. `task-floor` lifted the same recipe to about `95.46%`, showing that old-task
   retention was the lever. It is not paper-faithful because it uses task IDs.
3. `age-diversity` recovered most of the lift without task IDs, but still left
   the earliest task under-retained.
4. `age-diversity-oldest-floor` protects the oldest few snapshots and then
   applies age-diversity to the rest. With floor 4 it reached `95.043%`,
   essentially at the paper line but still slightly below it on the five-seed
   mean.

Relevant sweep harnesses live in:

```text
scripts/sweep_results/phase5m_age_diversity/
scripts/sweep_results/phase5n_age_bucket_floor/
scripts/sweep_results/phase5o_age_diversity_oldest_floor/
```

Generated JSON/log outputs are intentionally not part of the implementation
package. The harness scripts and summarizers are the reproducible artifacts.

## Core Concepts

### `NctlNetwork`

`NctlNetwork` owns the layer stack and online temporal schedule. It parses the
network topology, builds one `CudaFmnMixtureLayer` per layer, drives chunked
training, handles PTW depth/min-segment logic, exposes prediction methods, and
supports snapshot/restore during evaluation.

### `CudaFmnMixtureLayer`

Each layer owns:

- random halfspace contexts for every node;
- fresh/base weights plus remembered pool snapshots;
- active mixture weights and segment log probabilities;
- optional per-level active state for paper-fidelity PTW recursion;
- bounded model-pool metadata: task provenance, inserted level, insertion
  index, reservoir statistics, and evicted-task counters.

The CUDA kernel performs dense per-sample updates. Python handles sparse
segment-boundary events, model-pool admission, eviction, serialization, and
diagnostic summaries.

### How GGM Enters the NCTL Path

GGM means Gated Geometric Mixer.  It is still the node-local learner in NCTL,
but it is not reached through the standalone `BatchedGGMLayer` or `ggm_cuda`
module during current `run_split_mnist.py` runs.  Instead,
`CudaFmnMixtureLayer` stores one GGM weight table per node, pool slot, context,
and input feature:

```text
mixture_weights: [nodes, pool_slots + fresh_slot, contexts, input_features]
```

In the multilevel path this becomes:

```text
level_mixture_weights: [ptw_levels, nodes, pool_slots + fresh_slot, contexts, input_features]
```

The fused FMN mixture kernels perform the three GGM operations directly:

1. Compute the context by projecting side information through the layer's
   random hyperplanes and packing the signs into a context index.
2. Predict with the selected context row:
   `sigmoid(weight[context] dot logit(previous_layer_probs))`.
3. Apply the online GGM update:
   `weight += lr * (symbol - prediction) * logit(previous_layer_probs)`.

FMN then wraps those GGM slot predictions: it posterior-weights the fresh model
and remembered pool snapshots, tracks segment likelihoods, and decides which
active GGM snapshot is written back to the bounded pool at segment close.

So the active stack is:

```text
NctlNetwork
  -> CudaFmnMixtureLayer
     -> fmn_mixture_multilevel_cuda / fmn_mixture_cuda
        -> inlined GGM context + prediction + update
        -> FMN posterior mixture + segment likelihood tracking
```

not:

```text
NctlNetwork -> BatchedGGMLayer -> ggm_cuda
```

### Model Pool

The pool stores immutable snapshots of node-local GGM weights. When a segment
closes, `pool_update_policy` decides whether the segment refines, skips, or
adds a model. If the pool is full, `pool_evict_policy` selects the overwritten
slot.

Task-free eviction policies only use benchmark-agnostic metadata such as
`pool_insert_indices`. Task IDs are retained in provenance so we can diagnose
what happened after the run, but they should not steer training-time decisions
for paper-faithful results.

## Running Tests

Focused CPU tests for the pool policies and summary tooling:

```bash
python3 -m pytest scripts/tests/test_pool_evict_policy.py -q
python3 -m pytest scripts/tests/test_phase5o_age_diversity_oldest_floor_summary.py -q
python3 -m pytest scripts/tests/test_phase5o_age_diversity_oldest_floor_nofifo_summary.py -q
```

Repo-level gates:

```bash
ruff check scripts
python3 -m pytest scripts/tests -q
```

CUDA-specific tests are skipped automatically on CPU-only hosts where the test
helpers mark them as requiring CUDA kernels.

## Extending This Implementation

When changing the NCTL path, keep these boundaries clear:

- Put benchmark orchestration and command-line flags in `run_split_mnist.py`.
- Put online algorithm state, pool logic, PTW recursion, and serialization in
  `nctl_network.py`.
- Put hot per-sample math in CUDA kernels only after the Python reference path
  or CPU-focused tests make the intended behavior clear.
- Add CPU tests for policy/summary/serialization behavior; add CUDA tests only
  for kernel semantics and parity.
- Keep generated sweep outputs out of commits unless the repository explicitly
  asks for a result artifact. Commit reproducible scripts and summaries instead.

For future hyperparameter searches in other environments, start from
`../../docs/nctl-parameter-reference.md` and scale `min_segment`, `pool`, and
`pool_oldest_floor` by task length and pool pressure rather than copying
Split-MNIST values blindly.
