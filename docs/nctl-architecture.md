# Neural Combinatorial Transfer Learning (NCTL) Architecture

This document sketches the GPU Neural Combinatorial Transfer Learning (NCTL)
Split-MNIST benchmark architecture.  It is intended as a quick map for future
hyperparameter searches and implementation work; see
`docs/nctl-parameter-reference.md` for the full parameter inventory.

## End-to-End Flow

```mermaid
flowchart TD
    CLI["run_split_mnist.py<br/>CLI parameters + seed"]
    DATA["Binary MNIST / Fashion-MNIST data<br/>load_dataset() normalizes pixels"]
    TASKS["Split task stream<br/>(0,1) -> (2,3) -> (4,5) -> (6,7) -> (8,9)"]
    NET["NctlNetwork<br/>layers, PTW depth, min_segment,<br/>pool and prediction settings"]
    TRAIN["train_chunk()<br/>online mini-chunk driver"]
    MSCB["MSCB / segment-close logic<br/>decides which PTW levels close"]
    LAYERS["CudaFmnMixtureLayer stack<br/>one layer per network level"]
    KERNEL["fmn_mixture_multilevel CUDA kernel<br/>inlined GGM (Gated Geometric Mixer)<br/>context/predict/update, segment log probs,<br/>reservoirs, predictions"]
    ACTIVE["Per-level active state<br/>active mixtures + reservoirs for levels 0..d"]
    POOL["Bounded model pool per node<br/>snapshots, log probs, levels,<br/>insert indices, task-id provenance"]
    UPDATE["Pool update policy<br/>fifo or paper alpha/beta heuristic"]
    EVICT["Pool eviction policy<br/>fifo, age-diversity,<br/>age-diversity-oldest-floor, diagnostics"]
    PRED["Prediction mode<br/>selected_level or ptw_dp"]
    EVAL["evaluate_task()<br/>adapt on first adapt_n samples,<br/>score common/effective suffix"]
    JSON["Run JSON output<br/>accuracy matrix, forgetting,<br/>pool provenance, timings"]
    SWEEP["Sweep tooling<br/>shell A/B scripts or Optuna/MLflow"]

    CLI --> DATA
    DATA --> TASKS
    CLI --> NET
    TASKS --> TRAIN
    NET --> TRAIN
    TRAIN --> MSCB
    MSCB --> LAYERS
    LAYERS --> KERNEL
    KERNEL --> ACTIVE
    KERNEL --> PRED
    MSCB --> UPDATE
    ACTIVE --> UPDATE
    UPDATE --> POOL
    POOL --> EVICT
    EVICT --> POOL
    POOL --> KERNEL
    PRED --> EVAL
    NET --> EVAL
    EVAL --> JSON
    POOL --> JSON
    JSON --> SWEEP
```

## Layer and Pool Structure

```mermaid
flowchart LR
    subgraph Network["NctlNetwork"]
        direction TB
        L1["Hidden layer<br/>CudaFmnMixtureLayer"]
        L2["Hidden layer<br/>CudaFmnMixtureLayer"]
        L3["Output layer<br/>CudaFmnMixtureLayer"]
        L1 --> L2 --> L3
    end

    subgraph Layer["For each CudaFmnMixtureLayer"]
        direction TB
        NODE["Nodes"]
        FRESH["Fresh/base model slot<br/>slot = pool_capacity"]
        SNAP["Pool snapshot slots<br/>0..pool_capacity-1"]
        MIX["Mixture weights<br/>fresh + remembered snapshots"]
        RES["Reservoir / active state<br/>flat or per-level"]
        PROV["Provenance tensors<br/>task ids, levels, insert indices"]
        NODE --> FRESH
        NODE --> SNAP
        FRESH --> MIX
        SNAP --> MIX
        RES --> MIX
        SNAP --> PROV
    end

    Network --> Layer
```

## Control Loops

```mermaid
sequenceDiagram
    participant Sweep as Sweep driver
    participant Runner as run_split_mnist.py
    participant Net as NctlNetwork
    participant Layer as CudaFmnMixtureLayer
    participant CUDA as CUDA kernel
    participant Pool as Model pool

    Sweep->>Runner: choose params, seed, json_out
    Runner->>Net: construct network
    loop training tasks
        Runner->>Net: train_chunk(images, labels)
        Net->>Layer: forward/update chunk
        Layer->>CUDA: update online mixture state
        CUDA-->>Layer: predictions + segment evidence
        Net->>Layer: close eligible PTW levels
        Layer->>Pool: add/refine/skip snapshot
        Pool->>Pool: evict if full
    end
    loop evaluation tasks
        Runner->>Net: snapshot_state()
        Runner->>Net: adapt on adapt_n samples
        Runner->>Net: predict evaluation suffix
        Runner->>Net: restore_state()
    end
    Runner-->>Sweep: JSON metrics + provenance
```

## Architectural Notes

- `NctlNetwork` owns the layer stack, task-independent PTW/segment scheduling,
  and the evaluation-facing prediction path.
- `CudaFmnMixtureLayer` owns per-node hyperplanes, active mixture state,
  bounded pools, pool metadata, and provenance counters.
- The multilevel CUDA kernel performs the hot per-sample work, including
  Gated Geometric Mixer (GGM) context selection, prediction, and online weight
  update inlined inside the FMN mixture kernel. Python handles sparse boundary
  bookkeeping, pool admission, eviction, and JSON summaries.
- `active_state=per-level` plus `prediction_mode=ptw_dp` is the current
  paper-fidelity path because it gives every retained PTW level independent
  active state and mixes predictions with the Algorithm 1 dynamic program.
- Pool retention is the main lever for closing the Split-MNIST paper gap.
  Task-free policies should use insertion indices and other benchmark-agnostic
  metadata; `task-floor` is diagnostic because it consumes task IDs.
