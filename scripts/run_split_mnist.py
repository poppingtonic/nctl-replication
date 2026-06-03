#!/usr/bin/env python3
"""Split-MNIST / Split-Fashion-MNIST NCTL GPU benchmark.

Usage:
    python run_split_mnist.py [mnist|fashion-mnist]
    python run_split_mnist.py mnist --nodes 50-25-1 --lr 0.001

Notes on hyperparameters (see docs/forget-me-not-process-paper.txt §3.3):
  * ``--min-segment`` corresponds to the paper's ``2**c`` threshold for
    skipping UPDATEMODELPOOL on short segments.  Defaults (32) are far too
    aggressive for ~12k-sample tasks; recommended sweep 1024..8192.
  * ``--pool`` is the FMN ``k`` parameter.  With 5 tasks plus base, 8 is the
    minimum useful capacity; 16..32 leaves room before oldest-eviction
    churn starts.
  * ``--seed`` matters.  Paper reports 95.07 ± 0.02 averaged over many
    task sequences (§5.2); single-seed numbers vary by several percentage
    points on this non-i.i.d. stream.
"""

import os

os.environ.setdefault("CUDA_HOME", "/usr")

import argparse
import json
import math
import struct
import time
from pathlib import Path

import torch
from nctl_bench import (
    DEFAULT_POOL_OLDEST_FLOOR,
    POOL_EVICT_POLICIES,
    NctlNetwork,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def load_dataset(path):
    with open(path, "rb") as f:
        buf = f.read()
    n = struct.unpack("<I", buf[0:4])[0]
    img_size = struct.unpack("<I", buf[4:8])[0]
    assert img_size == 784
    labels, images = [], []
    offset = 8
    for _ in range(n):
        labels.append(buf[offset])
        offset += 1
        images.append(list(buf[offset : offset + img_size]))
        offset += img_size
    images = torch.tensor(images, dtype=torch.float32) / 255.0
    labels = torch.tensor(labels, dtype=torch.long)
    mean = images.mean(dim=0)
    std = images.std(dim=0).clamp(min=1e-8)
    images = (images - mean) / std
    return images.to(DEVICE), labels.to(DEVICE)


def filter_task(images, labels, a, b):
    mask = (labels == a) | (labels == b)
    return images[mask], (labels[mask] == b).to(torch.int32)


def evaluate_task(
    net,
    test_imgs,
    test_lbls,
    adapt_n=50,
    *,
    in_place=True,
    eval_offset: int | None = None,
):
    """Adapt on first ``adapt_n`` test samples, predict on the rest.

    With ``in_place=True`` (Phase 5I-D default), the adaptation runs on
    ``net`` directly after a CPU snapshot of every mutable tensor, then
    the snapshot restores the original state.  This keeps GPU peak
    memory at the training peak instead of 2x (clone()-based path) and
    unlocks larger ``--pool`` values on the same GPU.

    Setting ``in_place=False`` falls back to the legacy ``net.clone()``
    path; useful when the snapshot mechanism is unavailable (legacy
    state dicts or test harnesses that lack snapshot_state).

    ``eval_offset`` (bd 30.12 fairness fix): when provided, the evaluation
    subset is ``test[eval_offset:]`` instead of ``test[adapt_n:]``.  This
    lets a sweep over different ``adapt_n`` values compare on a common
    test suffix (e.g. set ``eval_offset = max(adapt_n_grid)`` to ensure
    every cell evaluates on identical samples).  ``eval_offset`` must be
    >= ``adapt_n`` so the adapt window is never re-used as eval.
    """
    n = len(test_imgs)
    n_adapt = min(adapt_n, n)
    eval_start = n_adapt if eval_offset is None else max(int(eval_offset), n_adapt)
    if eval_start >= n:
        # No test samples left after the (possibly larger) eval offset.
        if in_place and hasattr(net, "snapshot_state"):
            snap = net.snapshot_state()
            try:
                if n_adapt > 0:
                    net.train_chunk(
                        test_imgs[:n_adapt], test_lbls[:n_adapt]
                    )
            finally:
                net.restore_state(snap)
        return 50.0
    if in_place and hasattr(net, "snapshot_state"):
        snap = net.snapshot_state()
        try:
            if n_adapt > 0:
                net.train_chunk(test_imgs[:n_adapt], test_lbls[:n_adapt])
            eval_imgs = test_imgs[eval_start:]
            eval_lbls = test_lbls[eval_start:]
            preds = net.predict_batch(eval_imgs)
            predicted = (preds > 0.5).long()
            correct = (predicted == eval_lbls.to(DEVICE)).sum().item()
            return correct / len(eval_lbls) * 100.0
        finally:
            net.restore_state(snap)
    # Legacy clone-based path retained for back-compat.
    eval_net = net.clone()
    if n_adapt > 0:
        eval_net.train_chunk(test_imgs[:n_adapt], test_lbls[:n_adapt])
    eval_imgs = test_imgs[eval_start:]
    eval_lbls = test_lbls[eval_start:]
    preds = eval_net.predict_batch(eval_imgs)
    predicted = (preds > 0.5).long()
    correct = (predicted == eval_lbls.to(DEVICE)).sum().item()
    return correct / len(eval_lbls) * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="mnist")
    parser.add_argument("--nodes", default="50-25-1")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--halfspaces", type=int, default=4)
    parser.add_argument("--pool", type=int, default=8)
    parser.add_argument(
        "--output-pool",
        type=int,
        default=None,
        help=(
            "Optional pool capacity for the final/output layer only. "
            "Defaults to --pool. Use this to test output-layer retention "
            "without globally increasing all hidden-layer pools."
        ),
    )
    parser.add_argument("--min-segment", type=int, default=32)
    parser.add_argument("--ptw-depth", type=int, default=15)
    parser.add_argument(
        "--posterior-temp",
        type=float,
        default=1.0,
        help="Temperature scaling for the FMN current-segment posterior "
             "(1.0=standard Bayesian conditional, <1 mitigates posterior "
             "saturation in long segments).",
    )
    parser.add_argument(
        "--pool-update-policy",
        choices=["fifo", "paper"],
        default="fifo",
        help="Model-pool update rule. 'paper' enables the FMN §3.3 alpha/beta refine/skip/add heuristic.",
    )
    parser.add_argument("--pool-alpha", type=float, default=math.inf)
    parser.add_argument("--pool-beta", type=float, default=math.inf)
    parser.add_argument("--pool-reservoir", type=int, default=64)
    parser.add_argument(
        "--pool-evict-policy",
        choices=sorted(POOL_EVICT_POLICIES),
        default="fifo",
        help=(
            "Pool eviction policy when the pool is full.  'fifo' (default,"
            " paper-faithful) always drops the oldest snapshot.  'task-floor'"
            " is an opt-in diagnostic (bd 30.11) that walks slots oldest-first"
            " and protects the sole surviving snapshot of each task on a node;"
            " it leaks benchmark task identity and is NOT paper-faithful."
            " 'age-diversity' is the task-free follow-up that preserves"
            " temporal coverage using insertion indices only. 'age-bucket-floor'"
            " protects log-age buckets before falling back to FIFO."
            " 'age-diversity-oldest-floor' is age-diversity that additionally"
            " protects the --pool-oldest-floor oldest snapshots, guarding the"
            " earliest task against starvation (still task-free)."
        ),
    )
    parser.add_argument(
        "--pool-oldest-floor",
        type=int,
        default=DEFAULT_POOL_OLDEST_FLOOR,
        help=(
            "Number of oldest snapshots protected by the"
            " 'age-diversity-oldest-floor' eviction policy. Inert for every"
            " other policy."
        ),
    )
    parser.add_argument(
        "--active-state",
        choices=["flat", "per-level"],
        default="flat",
        help=(
            "Active FMN segment state layout. 'per-level' gives each retained "
            "PTW level independent active mixture weights/reservoirs while "
            "sharing the global model pool."
        ),
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["selected_level", "ptw_dp"],
        default="selected_level",
        help=(
            "Per-step predictive mixture.  'selected_level' returns the "
            "prediction-level's per-level conditional (legacy diagnostic). "
            "'ptw_dp' uses paper Algorithm 1's bottom-up w_j/b_j PTW "
            "dynamic-programming recursion to mix per-level conditionals.  "
            "ptw_dp requires --active-state=per-level."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--close-task-boundary",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "If 1, force-close the currently open FMN segment after each "
            "training task. This commits task-tail segments before the next "
            "task starts and avoids cross-task segment contamination under "
            "large --min-segment settings."
        ),
    )
    parser.add_argument(
        "--adapt-n",
        type=int,
        default=50,
        help=(
            "Number of test samples consumed by evaluate_task's adapt step"
            " before per-task accuracy is measured.  Default 50 matches"
            " the NCTL paper and the 30.9 acceptance recipe.  bd 30.12"
            " sweeps {10, 50, 200, 1000}; use --eval-suffix-from to keep"
            " the evaluation subset identical across sweep cells."
        ),
    )
    parser.add_argument(
        "--eval-suffix-from",
        type=int,
        default=None,
        help=(
            "Common-suffix evaluation offset (bd 30.12 fairness fix)."
            " When set, evaluate_task tests on test[eval_suffix_from:]"
            " instead of test[adapt_n:], so multiple adapt_n values can"
            " be compared on identical evaluation samples.  Must be >="
            " --adapt-n; values below adapt_n are silently raised."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write per-run summary JSON to this path (for sweep aggregation).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-task stdout; useful when driven by sweep_split_mnist.py.",
    )
    parser.add_argument(
        "--profile-timings",
        type=str,
        default=None,
        help=(
            "Enable the Phase 5H-1 hotspot profiler and write per-section "
            "timing JSON to this path at the end of the run.  When set, the "
            "summary table is also printed to stderr.  Adds a small wrapper "
            "around every instrumented method but no compute work; expect "
            "~5--15%% overhead at depth=15."
        ),
    )
    parser.add_argument(
        "--profile-timings-sync-cuda",
        action="store_true",
        help=(
            "When --profile-timings is set, also call torch.cuda.synchronize() "
            "around every bracketed section so GPU dispatch latency does not "
            "leak between sections.  Adds noticeable per-section overhead "
            "(milliseconds per call); use for diagnostic runs only."
        ),
    )
    args = parser.parse_args()

    # Phase 5H-1: configure the global profiler BEFORE NctlNetwork is built
    # so the constructor's instrumented methods are captured if they ever
    # get decorated.  Currently we instrument hot training-path methods
    # only; cold ctor work is unprofiled.
    if args.profile_timings is not None:
        from nctl_bench._profile import configure_profiler, reset_profiler  # noqa: PLC0415
        reset_profiler()
        configure_profiler(
            enabled=True, sync_cuda=bool(args.profile_timings_sync_cuda)
        )

    def log(msg=""):
        if not args.quiet:
            print(msg, flush=True)

    data_dir = Path(__file__).parent.parent / "data"
    layer_sizes = [int(x) for x in args.nodes.split("-")]

    log(f"=== Split-{args.dataset.upper()} NCTL GPU Benchmark ===")
    log(f"Device: {DEVICE}")
    log(f"Network: {args.nodes} = {sum(layer_sizes)} nodes")
    output_pool = args.output_pool if args.output_pool is not None else args.pool

    log(
        f"C={2**args.halfspaces}, LR={args.lr}, Pool={args.pool}, "
        f"OutputPool={output_pool}, "
        f"MinSeg={args.min_segment}, PTWDepth={args.ptw_depth}, "
        f"PosteriorTemp={args.posterior_temp}, PoolPolicy={args.pool_update_policy}, "
        f"Alpha={args.pool_alpha}, Beta={args.pool_beta}, Reservoir={args.pool_reservoir}, "
        f"EvictPolicy={args.pool_evict_policy}, "
        f"OldestFloor={args.pool_oldest_floor}, "
        f"ActiveState={args.active_state}, "
        f"PredictionMode={args.prediction_mode}, "
        f"Chunk={args.chunk_size}, CloseTaskBoundary={args.close_task_boundary} "
        f"Seed={args.seed}"
    )
    log()

    train_imgs, train_lbls = load_dataset(str(data_dir / f"{args.dataset}_train.bin"))
    test_imgs, test_lbls = load_dataset(str(data_dir / f"{args.dataset}_test.bin"))
    log(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")

    net = NctlNetwork(
        layer_sizes=layer_sizes,
        input_dim=784,
        num_halfspaces=args.halfspaces,
        lr=args.lr,
        pool_capacity=args.pool,
        output_pool_capacity=args.output_pool,
        min_segment=args.min_segment,
        ptw_depth=args.ptw_depth,
        posterior_temp=args.posterior_temp,
        pool_update_policy=args.pool_update_policy,
        pool_alpha=args.pool_alpha,
        pool_beta=args.pool_beta,
        pool_reservoir_size=args.pool_reservoir,
        pool_evict_policy=args.pool_evict_policy,
        pool_oldest_floor=args.pool_oldest_floor,
        active_state_mode=args.active_state.replace("-", "_"),
        prediction_mode=args.prediction_mode,
        device=DEVICE,
        seed=args.seed,
    )
    log(f"Nodes: {net.num_nodes()}\n")

    # Phase 5G Commit B startup banner.  Prints unconditionally (even under
    # --quiet) so a sweep operator can confirm that the multilevel CUDA
    # extension loaded with the Phase 5F D-step 2 ``forward_update_with_resets``
    # binding (otherwise train_chunk silently falls back to the slow legacy
    # path with no log signal).  See COOKBOOK Phase 5G entries.
    try:
        from nctl_bench.nctl_network import _get_fmn_multilevel_cuda  # noqa: PLC0415
        ml = _get_fmn_multilevel_cuda()
        has_with_resets = (
            ml is not None and hasattr(ml, "forward_update_with_resets")
        )
        print(
            "STARTUP "
            f"multilevel_loaded={ml is not None} "
            f"has_with_resets={has_with_resets} "
            f"chunk_size={args.chunk_size} "
            f"ptw_depth={args.ptw_depth} "
            f"min_segment={args.min_segment} "
            f"active_state={args.active_state} "
            f"prediction_mode={args.prediction_mode}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        # Banner must never abort the run.  If the import fails we still
        # print enough to let the operator know.
        print(f"STARTUP banner failed: {exc!r}", flush=True)

    test_sets = [filter_task(test_imgs, test_lbls, a, b) for a, b in TASKS]
    acc_matrix = [[0.0] * 5 for _ in range(5)]
    per_task_loss = []
    per_task_pool = []
    per_task_time = []
    per_task_provenance = []
    start = time.time()

    for ti, (ca, cb) in enumerate(TASKS):
        log(f"--- Task {ti + 1} ({ca} vs {cb}) ---")
        imgs, lbls = filter_task(train_imgs, train_lbls, ca, cb)
        n = len(imgs)
        log(f"  Training on {n} samples...")

        t0 = time.time()
        total_loss = 0.0
        # Process in chunks for FMN segment management
        for start_idx in range(0, n, args.chunk_size):
            end_idx = min(start_idx + args.chunk_size, n)
            chunk_z = imgs[start_idx:end_idx]
            chunk_s = lbls[start_idx:end_idx]
            chunk_loss = net.train_chunk(chunk_z, chunk_s, task_id=ti + 1)
            total_loss += chunk_loss

        if args.close_task_boundary:
            net.close_open_segment(task_id=ti + 1, global_index=net.index)
        elapsed = time.time() - t0
        per_task_loss.append(total_loss / max(n, 1))
        per_task_pool.append(int(net.total_pool_size()))
        per_task_time.append(elapsed)
        # Snapshot pool provenance immediately after each task to expose
        # task-distinct retention without rerunning.  Cheap: walks pool
        # slots once per layer.
        prov_now = net.provenance_summary()
        per_task_provenance.append(
            {
                "task_histogram": dict(prov_now["task_histogram"]),
                "event_counts": dict(prov_now["event_counts"]),
                "evicted_task_counts": dict(prov_now["evicted_task_counts"]),
            }
        )
        th = prov_now["task_histogram"]
        ec = prov_now["event_counts"]
        log(
            f"  Avg loss: {total_loss / n:.4f} nats ({elapsed:.1f}s)"
        )
        log(
            f"  Pool: {net.total_pool_size()} snapshots; tasks=" + str(dict(sorted(th.items())))
        )
        log(
            "  Events: " + str(dict(sorted(ec.items())))
        )

        for ei in range(ti + 1):
            acc_matrix[ti][ei] = evaluate_task(
                net,
                *test_sets[ei],
                adapt_n=args.adapt_n,
                eval_offset=args.eval_suffix_from,
            )

        accs = " ".join(
            f"[{TASKS[j][0]}v{TASKS[j][1]}]={acc_matrix[ti][j]:.1f}%"
            for j in range(ti + 1)
        )
        log(f"  Accuracy: {accs}\n")

    total_time = time.time() - start
    avg_acc = sum(acc_matrix[4]) / 5.0
    fgt = (
        sum(
            max(acc_matrix[t][j] for t in range(j, 5)) - acc_matrix[4][j]
            for j in range(4)
        )
        / 4.0
    )

    log("=== Results ===\n")
    header = "".join(f"  T{j + 1}({TASKS[j][0]},{TASKS[j][1]})" for j in range(5))
    log(f"{'':>12}{header}")
    for t in range(5):
        row = f"After T{t + 1}: "
        for j in range(5):
            row += f"  {acc_matrix[t][j]:>6.1f}%" if j <= t else "       --"
        log(row)

    log(f"\nAverage Accuracy:   {avg_acc:.2f}%")
    log(f"Average Forgetting: {fgt:.2f}%")
    log(f"Total time:         {total_time:.1f}s")
    log("\n| Method               | Avg Accuracy | Avg Forgetting |")
    log("|----------------------|--------------|----------------|")
    log(f"| NCTL (this run)      | {avg_acc:>11.2f}% | {fgt:>13.2f}% |")
    log("| Naive                |       19.62% |        99.28%  |")
    log("| EWC (lambda=580k)    |       43.74% |        53.04%  |")
    log("| Replay (buffer=200)  |       88.45% |        11.29%  |")
    log("| Joint Training       |       96.97% |         0.00%  |")
    log("| NCTL (paper, 50-25-1)|       95.07% |            --  |")

    if args.json_out:
        payload = {
            "dataset": args.dataset,
            "seed": args.seed,
            "nodes": args.nodes,
            "lr": args.lr,
            "halfspaces": args.halfspaces,
            "pool_capacity": args.pool,
            "output_pool_capacity": output_pool,
            "min_segment": args.min_segment,
            "ptw_depth": args.ptw_depth,
            "posterior_temp": args.posterior_temp,
            "pool_update_policy": args.pool_update_policy,
            "pool_alpha": args.pool_alpha,
            "pool_beta": args.pool_beta,
            "pool_reservoir": args.pool_reservoir,
            "pool_evict_policy": args.pool_evict_policy,
            "pool_oldest_floor": args.pool_oldest_floor,
            "active_state": args.active_state,
            "prediction_mode": args.prediction_mode,
            "chunk_size": args.chunk_size,
            "close_task_boundary": bool(args.close_task_boundary),
            "acc_matrix": acc_matrix,
            "avg_accuracy": avg_acc,
            "avg_forgetting": fgt,
            "per_task_loss": per_task_loss,
            "per_task_pool": per_task_pool,
            "per_task_provenance": per_task_provenance,
            "per_task_time": per_task_time,
            "pool_provenance": net.provenance_summary(),
            "total_time": total_time,
            "device": str(DEVICE),
            "torch_version": torch.__version__,
        }
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2))
        if args.quiet:
            print(
                json.dumps(
                    {
                        "seed": args.seed,
                        "min_segment": args.min_segment,
                        "pool": args.pool,
                        "output_pool": output_pool,
                        "avg_accuracy": avg_acc,
                        "avg_forgetting": fgt,
                        "pool_update_policy": args.pool_update_policy,
                        "pool_alpha": args.pool_alpha,
                        "pool_beta": args.pool_beta,
                        "pool_evict_policy": args.pool_evict_policy,
                        "pool_oldest_floor": args.pool_oldest_floor,
                        "active_state": args.active_state,
                        "prediction_mode": args.prediction_mode,
                        "total_time": total_time,
                        "close_task_boundary": bool(args.close_task_boundary),
                        "json_out": args.json_out,
                    }
                )
            )


    if args.profile_timings is not None:
        from nctl_bench._profile import format_profile_table, write_profile_json  # noqa: PLC0415
        prof_path = Path(args.profile_timings)
        write_profile_json(prof_path)
        # Also echo a sorted table to stderr so the operator sees it
        # without having to open the JSON.  stderr because stdout is
        # often parsed by sweep callers (see run_split_mnist --quiet
        # JSON line at the tail of main()).
        import sys as _sys  # noqa: PLC0415
        print(
            f"\n=== PROFILE TIMINGS (top 30) -> {prof_path} ===\n"
            + format_profile_table(top=30),
            file=_sys.stderr,
            flush=True,
        )

if __name__ == "__main__":
    main()
