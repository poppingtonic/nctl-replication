# CUDA Kernel Build Notes

## Environment

- GPU: NVIDIA GeForce RTX 2080 Ti (compute capability 7.5)
- Driver: 580.159.03
- System nvcc: CUDA 12.0 at `/usr/lib/nvidia-cuda-toolkit/bin/nvcc`
- PyTorch: 2.9.0+cu128 (expects CUDA 12.8 — version mismatch)
- GCC: 13.3.0 (default), 12.4.0, 11.5.0 available
- OS: Ubuntu 24.04

## The Problem

PyTorch's `torch.utils.cpp_extension.load()` JIT compiler has two issues:

1. **CUDA version mismatch**: PyTorch 2.9 expects `/usr/local/cuda-12.8/bin/nvcc` but only
   12.0 is installed. Setting `CUDA_HOME` helps but introduces issue 2.

2. **math.h circular include**: When `CUDA_HOME` points to `/usr/include` (where the CUDA
   headers live on this system), the include path creates a cycle:
   - `/usr/include/crt/math_functions.h` → `#include <cmath>`
   - GCC's `cmath` → `#include_next <math.h>`
   - `#include_next` looks for the *next* `math.h` after the current include dir
   - But `/usr/include` is already the system include dir, so there is no "next"
   - Result: `fatal error: math.h: No such file or directory`

   This affects GCC 11, 12, AND 13. It's a known issue when CUDA headers are
   installed into `/usr/include/` instead of a separate `/usr/local/cuda/include/`.

## Working Compilation Command

Compile directly with nvcc, NOT through `torch.utils.cpp_extension.load()`:

```bash
/usr/lib/nvidia-cuda-toolkit/bin/nvcc \
  --allow-unsupported-compiler \
  -ccbin=/usr/bin/g++-11 \
  -gencode=arch=compute_75,code=sm_75 \
  -O3 -std=c++17 \
  --compiler-options '-fPIC' \
  -DTORCH_EXTENSION_NAME=<module_name> \
  -DTORCH_API_INCLUDE_EXTENSION_H \
  -isystem $(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")/../../include \
  -isystem $(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")/../../include/torch/csrc/api/include \
  -isystem $(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
  --shared \
  -o <output>.so \
  <input>.cu \
  -L$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")/../../lib \
  -lc10 -ltorch -ltorch_cpu -ltorch_python
```

The key: do NOT add `-isystem /usr/include` — let nvcc find system headers through
its default search path, which avoids the `#include_next` cycle.

## Loading Pre-compiled Kernels

```python
import importlib.util
spec = importlib.util.spec_from_file_location("module_name", "/path/to/module.so")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

## Kernels

| Kernel | File | Function | Notes |
|--------|------|----------|-------|
| ggm_kernel.cu | Single-model GGM forward+update | `forward_update`, `forward_only` | 6.7μs/sample, 76 nodes |
| fmn_mixture_kernel.cu | FMN posterior mixture with pool-synced updates | `forward_update`, `forward_only` | One CUDA block per node; loops over the batch inside the block. Runtime scales with active pool slots (`M = pool_capacity + 1`). Used by `selected_level` prediction mode. |
| fmn_mixture_multilevel_kernel.cu | FMN multilevel Eq. 5 ν_j over all PTW levels + segmented PTW-DP state advance | `forward_update`, `forward_only`, `forward_update_with_resets`, `segmented_dp_state_advance` | One CUDA block per (level, node) for FMN prediction/update; single launch covers every level. Phase 5I computes the Eq. 5 conditional as a posterior-weighted per-slot mixture using `segment_log_probs` and `posterior_temp` (`1.0` is the paper Bayes conditional; `0.0` recovers the old unweighted 1/2 fresh + 1/2 pool-mean mix). Phase 5H-2 adds one block per node for segmented PTW-DP event-state advancement. Used by `prediction_mode=ptw_dp` when the multilevel kernel is built; otherwise the Python wrapper falls back to L per-level launches / Python DP state walking. |

### Current `fmn_mixture_kernel.cu` Python interface

The FMN mixture kernel no longer uses a fixed uniform pool average. It consumes
per-model segment log-likelihoods and a posterior temperature so that
`run_split_mnist.py --posterior-temp ...` can interpolate between a hard uniform
pool/fresh blend (`0`) and Bayesian posterior weighting (`>0`).

`forward_update(...)` arguments:

```python
forward_update(
    z_batch,             # [B, D] float32 CUDA
    p_prev_batch,        # [B, K_in] float32 CUDA
    symbols,             # [B] int32 CUDA
    mixture_weights,     # [N, M, C, K_in] float32 CUDA; fresh slot is M-1
    hyperplanes,         # [N, H, D] float32 CUDA
    hp_bias,             # [N, H] float32 CUDA
    pool_sizes,          # [N] int32 CUDA; active pool slots per node
    segment_log_probs,   # [N, M] float32 CUDA; current-segment log p(data|model)
    model_log_probs,     # [N, M] float32 CUDA; lifetime diagnostic log p(data|model)
    lr,                  # float
    posterior_temp,      # float
)
```

`forward_only(...)` arguments:

```python
forward_only(
    z_batch,
    p_prev_batch,
    mixture_weights,
    hyperplanes,
    hp_bias,
    pool_sizes,
    segment_log_probs,
    posterior_temp,
)
```

Both functions return `[B, N]` predictions/probabilities. `forward_update` also
updates every active pool slot plus the fresh slot in-place and accumulates both
`segment_log_probs` and `model_log_probs`.

### `fmn_mixture_multilevel_kernel.cu` Python interface (Phase 5E)

This kernel batches the per-level FMN `forward_update` / `forward_only`
launches into one CUDA call.  State tensors carry an extra leading `L`
axis; `hyperplanes` and `hp_bias` stay level-independent because the
half-space context is shared across levels.

`forward_update(...)` arguments:

```python
forward_update(
    z_batch,             # [B, D] float32 CUDA
    p_prev_batch,        # [B, K_in] float32 CUDA
    symbols,             # [B] int32 CUDA
    mixture_weights,     # [L, N, M, C, K_in] float32 CUDA
    hyperplanes,         # [N, H, D] float32 CUDA
    hp_bias,             # [N, H] float32 CUDA
    pool_sizes,          # [L, N] int32 CUDA
    segment_log_probs,   # [L, N, M] float32 CUDA
    model_log_probs,     # [L, N, M] float32 CUDA
    lr,                  # float
    posterior_temp,      # float; 1.0 = paper Eq. 5 Bayes conditional, 0.0 = unweighted legacy
)
```

`forward_only(...)` arguments:

```python
forward_only(
    z_batch,             # [B, D] float32 CUDA
    p_prev_batch,        # [B, K_in] float32 CUDA
    mixture_weights,     # [L, N, M, C, K_in] float32 CUDA
    hyperplanes,         # [N, H, D] float32 CUDA
    hp_bias,             # [N, H] float32 CUDA
    pool_sizes,          # [L, N] int32 CUDA
    segment_log_probs,   # [L, N, M] float32 CUDA; current-segment log p(data|slot)
    posterior_temp,      # float
)
```

Both return `[L, B, N]` predictions of ν_j(x_obs | x_<t) (training path)
or ν_j(x_t=1 | x_<t) (read-only path).  Phase 5I computes the conditional
ratio implied by the FMN paper's Eq. 5 joint mixture: prior 1/2 on the fresh
base measure and prior `(1/2)/pool_size` on each active pool slot, multiplied
by `exp(posterior_temp * segment_log_probs)`.  Use `posterior_temp=1.0` for
the paper-faithful Bayes conditional; `posterior_temp=0.0` deliberately
recovers the pre-5I unweighted conditional used for regression tests.

`forward_update_with_resets(...)` extends `forward_update` with two inputs:

```python
forward_update_with_resets(
    z_batch, p_prev_batch, symbols,
    mixture_weights,
    pool_snapshots,       # [N, pool_capacity, C, K_in] float32 CUDA
    hyperplanes, hp_bias,
    pool_sizes,
    close_mask,           # [L, B] bool CUDA; reset active level before sample b
    segment_log_probs,
    model_log_probs,
    lr,
    posterior_temp,
)
```

It returns `[L, B, N]` predictions and applies close-only active-state resets
inside the same batch launch.

`segmented_dp_state_advance(...)` is the Phase 5H-2 PTW-DP state helper:

```python
segmented_dp_state_advance(
    cum_padded,           # [L, B+1, N] float64 CUDA
    event_offsets,        # [E] int32 CUDA, sorted unique offsets
    event_close_mask,     # [E, L] bool CUDA
    event_seg_idx,        # [E] int32 CUDA; segment seed index or -1
    trailing_seg_idx,     # int; trailing segment seed index or -1
    state_nu,             # [L, N] float64 CUDA, in/out
    state_w,              # [L, N] float64 CUDA, in/out
    state_b,              # [L, N] float64 CUDA, in/out
    num_segments,         # int
)
```

It returns `(seg_nu0, seg_w0, seg_b0)`, each `[S, L, N]` float64 CUDA, and
mutates `state_nu/state_w/state_b` to the post-chunk state.  The Python wrapper
uses these segment seeds to build `[L, B, N]` tensors for the vectorised convex
PTW combine while avoiding the old Python event loop over close boundaries.

## Build Script

```bash
#!/bin/bash
# Build all CUDA kernels
NVCC=/usr/lib/nvidia-cuda-toolkit/bin/nvcc
TORCH_INC=$(python3 -c "import torch; p=torch.utils.cmake_prefix_path; print(p+'/../../include')")
TORCH_LIB=$(python3 -c "import torch; p=torch.utils.cmake_prefix_path; print(p+'/../../lib')")
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
CACHE=~/.cache/torch_extensions/py310_cu128

for kernel in ggm_kernel fmn_mixture_kernel fmn_mixture_multilevel_kernel; do
    name=${kernel//_kernel/}_cuda
    mkdir -p $CACHE/$name
    $NVCC --allow-unsupported-compiler -ccbin=/usr/bin/g++-11 \
        -gencode=arch=compute_75,code=sm_75 -O3 -std=c++17 \
        --compiler-options '-fPIC' \
        -DTORCH_EXTENSION_NAME=$name -DTORCH_API_INCLUDE_EXTENSION_H \
        -isystem $TORCH_INC -isystem $TORCH_INC/torch/csrc/api/include \
        -isystem $PY_INC --shared \
        -o $CACHE/$name/$name.so ${kernel}.cu \
        -L$TORCH_LIB -lc10 -ltorch -ltorch_cpu -ltorch_python
    echo "Built $name → $CACHE/$name/$name.so"
done
```
