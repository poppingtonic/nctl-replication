#!/usr/bin/env python3
"""Pre-flight GPU memory check before launching pool=160.

Builds an NctlNetwork(50-25-1, pool_capacity=160) and reports CUDA
allocated/reserved.  Aborts with non-zero exit if reserved > 8 GiB
(handoff: pool=200 OOMs in forward_only due to predictions buffer;
8 GiB is a conservative ceiling on the 10.5 GiB device after accounting
for the per-chunk predictions and PTW DP scratch buffers).
"""
from __future__ import annotations

import sys

import torch

if not torch.cuda.is_available():
    print("cuda not available -- this script only runs on the GPU host",
          file=sys.stderr)
    raise SystemExit(2)

# Local import; assumes invocation from repo root with PYTHONPATH including
# rust-fmn/scripts.
sys.path.insert(0, "rust-fmn/scripts")
from nctl_bench.nctl_network import NctlNetwork  # noqa: E402

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

net = NctlNetwork(
    nodes_per_layer=[50, 25, 1],
    n_inputs=784,
    halfspaces=4,
    pool_capacity=160,
    lr=0.001,
    min_segment=512,
    pool_update_policy="paper",
    pool_alpha=0.0,
    pool_beta=0.0,
    pool_reservoir_size=10,
    active_state_mode="per-level",
    prediction_mode="ptw_dp",
    chunk_size=1024,
    posterior_temp=1.0,
)
del net  # ensure tensor lifetimes flush
torch.cuda.synchronize()
alloc = torch.cuda.memory_allocated() / (1 << 30)
reserved = torch.cuda.memory_reserved() / (1 << 30)
peak = torch.cuda.max_memory_reserved() / (1 << 30)
print(f"pool=160 allocated={alloc:.2f} GiB reserved={reserved:.2f} GiB "
      f"peak_reserved={peak:.2f} GiB")
if peak > 8.0:
    print("ABORT: peak reserved > 8 GiB; pool=160 unsafe on this device",
          file=sys.stderr)
    raise SystemExit(1)
print("OK: pool=160 safe to launch")
