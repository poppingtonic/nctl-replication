"""CPU unit tests for the Gaussian GLN head (bd 31).

Coverage:
  * API surface parity with the Bernoulli ``BatchedGGMLayer``
    (predict / update / snapshot / load / reset / clone).
  * Numerical correctness of the geometric-mixing predict path against
    a from-scratch NumPy reference that re-derives mu_out / tau_out
    independently.
  * Numerical correctness of the one-step weight update + log-density
    accumulation against a from-scratch NumPy reference.
  * Multi-step convergence on a synthetic stationary stream (the
    learned mean must approach the true target's mean).
  * Snapshot / restore round-trip is bitwise.
  * Edge cases: input precision floor/ceiling, weight clamps,
    init shape validation, invalid construction args.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from nctl_bench.gaussian_head import BatchedGaussianGGMLayer  # noqa: E402

# ---------------------------------------------------------------------------
# Reference implementation (NumPy; independent of the torch code).
# ---------------------------------------------------------------------------

def _np_predict(
    w_active: np.ndarray,            # [N, K]
    mus: np.ndarray,                 # [K]
    taus: np.ndarray,                # [K]
) -> tuple[np.ndarray, np.ndarray]:
    """Reference predict (precision-space geometric mixing)."""
    w_tau = w_active * taus[None, :]
    tau_out = np.maximum(w_tau.sum(axis=-1), 1e-6)
    mu_out = (w_tau * mus[None, :]).sum(axis=-1) / tau_out
    return mu_out, tau_out


def _np_grad(
    mus: np.ndarray,                 # [K]
    taus: np.ndarray,                # [K]
    mu_out: np.ndarray,              # [N]
    tau_out: np.ndarray,             # [N]
    y: float,
) -> np.ndarray:
    """Reference per-(node, input) NLL gradient.

    Returns a ``[N, K]`` array; broadcasts the [K] input stats over N.
    """
    term1 = taus[None, :] * (mus[None, :] - mu_out[:, None]) * (mu_out - y)[:, None]
    term2 = 0.5 * taus[None, :] * (
        ((y - mu_out) ** 2)[:, None] - (1.0 / tau_out)[:, None]
    )
    return term1 + term2


# ---------------------------------------------------------------------------
# API surface.
# ---------------------------------------------------------------------------

def _layer(seed: int = 7, **overrides) -> BatchedGaussianGGMLayer:
    kwargs = {
        "num_nodes": 3,
        "num_inputs": 4,
        "input_dim": 5,
        "num_halfspaces": 2,
        "lr": 0.05,
        "device": torch.device("cpu"),
        "seed": seed,
    }
    kwargs.update(overrides)
    return BatchedGaussianGGMLayer(**kwargs)


def test_constructor_initialises_weights_to_uniform() -> None:
    layer = _layer()
    expected = 1.0 / 4
    assert torch.allclose(layer.weights, torch.full_like(layer.weights, expected))


def test_constructor_rejects_zero_inputs() -> None:
    with pytest.raises(ValueError, match="num_inputs must be positive"):
        _layer(num_inputs=0)


def test_snapshot_restore_round_trip_is_bitwise() -> None:
    layer = _layer()
    # Train a bit so the weights are no longer the trivial init.
    rng = np.random.default_rng(0)
    for _ in range(10):
        mus = torch.as_tensor(np.asarray(rng.normal(size=4))).float()
        taus = torch.as_tensor(np.asarray(np.abs(rng.normal(size=4))) + 0.5).float()
        z = torch.as_tensor(np.asarray(rng.normal(size=5))).float()
        layer.update(mus, taus, z, float(rng.normal()))
    snap = layer.snapshot_weights()
    layer.reset_weights()
    # Reset must produce the uniform init again.
    expected = 1.0 / 4
    assert torch.allclose(layer.weights, torch.full_like(layer.weights, expected))
    # And load_weights must restore the trained snapshot exactly.
    layer.load_weights(snap)
    assert torch.equal(layer.weights, snap)


def test_load_weights_rejects_shape_mismatch() -> None:
    layer = _layer()
    with pytest.raises(ValueError, match="shape mismatch"):
        layer.load_weights(torch.zeros(2, 2, 2))


def test_clone_decouples_weights_but_shares_gates() -> None:
    layer = _layer()
    c = layer.clone()
    assert c.weights is not layer.weights
    assert c.hyperplanes is layer.hyperplanes      # shared, read-only
    assert c.hyperplane_bias is layer.hyperplane_bias
    c.weights[0, 0, 0] = 99.0
    assert layer.weights[0, 0, 0] != 99.0


# ---------------------------------------------------------------------------
# Predict path parity with the NumPy reference.
# ---------------------------------------------------------------------------

def test_predict_matches_numpy_reference() -> None:
    layer = _layer()
    rng = np.random.default_rng(123)
    mus = rng.normal(size=4).astype(np.float32)
    taus = (np.abs(rng.normal(size=4)) + 0.3).astype(np.float32)
    z = rng.normal(size=5).astype(np.float32)

    mu_t, tau_t = layer.predict(
        torch.as_tensor(np.asarray(mus)), torch.as_tensor(np.asarray(taus)), torch.as_tensor(np.asarray(z))
    )

    # NumPy reference: compute contexts the same way the layer does, then mix.
    hp = np.asarray(layer.hyperplanes.detach().cpu().tolist(), dtype=np.float32)
    bias = np.asarray(layer.hyperplane_bias.detach().cpu().tolist(), dtype=np.float32)
    dots = np.einsum("nhd,d->nh", hp, z) + bias
    bits = (dots >= 0).astype(np.int64)
    powers = 2 ** np.arange(layer.num_halfspaces)
    contexts = (bits * powers).sum(axis=-1)
    n_idx = np.arange(layer.num_nodes)
    w_active = np.asarray(layer.weights.detach().cpu().tolist(), dtype=np.float32)[n_idx, contexts]
    mu_ref, tau_ref = _np_predict(w_active, mus, taus)

    assert np.allclose(np.asarray(mu_t.tolist(), dtype=np.float32), mu_ref, atol=1e-5)
    assert np.allclose(np.asarray(tau_t.tolist(), dtype=np.float32), tau_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Update path parity with the NumPy reference.
# ---------------------------------------------------------------------------

def test_update_step_matches_numpy_reference() -> None:
    layer = _layer(lr=0.07)
    rng = np.random.default_rng(456)
    mus_np = rng.normal(size=4).astype(np.float32)
    taus_np = (np.abs(rng.normal(size=4)) + 0.4).astype(np.float32)
    z_np = rng.normal(size=5).astype(np.float32)
    y = float(rng.normal())

    # Snapshot the layer state we'll need for the NumPy expectation.
    hp = np.asarray(layer.hyperplanes.detach().cpu().tolist(), dtype=np.float32)
    bias = np.asarray(layer.hyperplane_bias.detach().cpu().tolist(), dtype=np.float32)
    dots = np.einsum("nhd,d->nh", hp, z_np) + bias
    bits = (dots >= 0).astype(np.int64)
    powers = 2 ** np.arange(layer.num_halfspaces)
    contexts = (bits * powers).sum(axis=-1)
    n_idx = np.arange(layer.num_nodes)
    w_before = np.asarray(layer.weights.detach().cpu().tolist(), dtype=np.float32).copy()
    w_active = w_before[n_idx, contexts]

    # Run one update through the torch layer.
    mu_t, tau_t = layer.update(
        torch.as_tensor(np.asarray(mus_np)),
        torch.as_tensor(np.asarray(taus_np)),
        torch.as_tensor(np.asarray(z_np)),
        y,
    )

    # NumPy expectation: predict, then SGD step.
    mu_ref, tau_ref = _np_predict(w_active, mus_np, taus_np)
    grad = _np_grad(mus_np, taus_np, mu_ref, tau_ref, y)
    w_active_after = np.clip(w_active - 0.07 * grad, 1e-6, 200.0)

    # 1) The return value is the PRE-update predictive (mu_out, tau_out).
    assert np.allclose(np.asarray(mu_t.tolist(), dtype=np.float32), mu_ref, atol=1e-5)
    assert np.allclose(np.asarray(tau_t.tolist(), dtype=np.float32), tau_ref, atol=1e-5)

    # 2) Only the active context row of each node should have been updated.
    w_after = np.asarray(layer.weights.detach().cpu().tolist(), dtype=np.float32)
    for n in range(layer.num_nodes):
        c = int(contexts[n])
        assert np.allclose(w_after[n, c], w_active_after[n], atol=1e-5)
        # The non-active rows must be byte-identical to before.
        for cc in range(layer.num_contexts):
            if cc == c:
                continue
            assert np.array_equal(w_after[n, cc], w_before[n, cc])

    # 3) log_block_prob must have been incremented by the pre-update log p.
    log_p_ref = 0.5 * (np.log(tau_ref) - np.log(2.0 * np.pi)) \
        - 0.5 * tau_ref * (mu_ref - y) ** 2
    assert np.allclose(np.asarray(layer.log_block_prob.detach().cpu().tolist(), dtype=np.float32), log_p_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Convergence sanity: the learned mean drifts toward a stationary target.
# ---------------------------------------------------------------------------

def test_learns_to_predict_stationary_target_mean() -> None:
    """Drive the same (mus, taus, z) repeatedly with a fixed y; mu_out -> y."""
    layer = _layer(num_nodes=1, num_inputs=4, num_halfspaces=1, lr=0.2)
    rng = np.random.default_rng(11)
    # Use inputs whose precision-weighted mean is NOT already equal to y;
    # the learner has to move weights to close the gap.
    input_mus = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    input_taus = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    z = torch.as_tensor(np.asarray(rng.normal(size=5))).float()
    y = 1.0
    # With all input means at 0, mu_out = (sum w * tau * 0) / (...) = 0.
    # So the learner cannot reduce (mu_out - y) by tweaking weights; the
    # gradient still moves *precision* toward a higher value via term2,
    # which shrinks log p sharply.  Switch to inputs that DO span the
    # target: mus = [-1, 1, -1, 1].  Then the right precision-weighting
    # over the positive inputs reduces (mu_out - 1)^2.
    input_mus = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float32)
    for _ in range(400):
        layer.update(input_mus, input_taus, z, y)
    mu_final, _ = layer.predict(input_mus, input_taus, z)
    # After 400 steps the output should be much closer to y=1 than to the
    # untrained baseline of 0 (precision-weighted average of [-1,1,-1,1]).
    assert mu_final.item() > 0.5, f"mu_final={mu_final.item()} did not move toward y=1"


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------

def test_input_precision_floor_prevents_zero_tau_out() -> None:
    """tau_in=0 must be clamped above the floor so tau_out > 0."""
    layer = _layer()
    mus = torch.zeros(4)
    taus = torch.zeros(4)   # would zero out tau_out without the floor
    z = torch.zeros(5)
    mu_out, tau_out = layer.predict(mus, taus, z)
    assert torch.all(tau_out > 0)
    assert torch.isfinite(mu_out).all()


def test_weight_clamp_keeps_weights_in_range() -> None:
    """Huge gradients must not push weights out of [_MIN_WEIGHT, _MAX_WEIGHT]."""
    layer = _layer(lr=1e9)        # absurd learning rate
    mus = torch.tensor([1.0, 2.0, 3.0, 4.0])
    taus = torch.tensor([1.0, 1.0, 1.0, 1.0])
    z = torch.zeros(5)
    layer.update(mus, taus, z, y=-1000.0)   # huge residual
    # Every weight in the active row must still be inside the clamp range.
    assert torch.all(layer.weights >= 1e-6 - 1e-8)
    assert torch.all(layer.weights <= 200.0 + 1e-5)
