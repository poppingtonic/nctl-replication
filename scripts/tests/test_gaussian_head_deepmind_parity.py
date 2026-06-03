"""Parity test against the DeepMind reference Gaussian GLN (bd 31 verification).

The reference is at:
  /home/muhia/ai-safety-research/deepmind-research/gated_linear_networks/
  gaussian.py (jax + haiku; Apache-2.0)

We cannot import the reference directly because it depends on jax + haiku
+ chex + tensorflow_probability which are not in this repo's pyproject.
Instead, we re-implement the reference's _inference_fn math in pure NumPy
following the file exactly, then assert our BatchedGaussianGGMLayer
agrees on the predict path.

The two known intentional deltas vs the DeepMind reference -- positive-only
weights vs the DeepMind hard projection, and `>= 0` vs `> 0` context
sign convention -- are exercised explicitly here so future readers know
the deviations are deliberate and bounded.
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
# DeepMind reference math (transcribed from gaussian.py:_inference_fn lines
# 81-100 of the v0 release).  No projection -- we pick inputs that stay
# inside the bounds so the projection is a no-op on the reference and the
# only comparison is the core mixing formula.
# ---------------------------------------------------------------------------

def _deepmind_inference(
    w_active: np.ndarray,   # [N, K]  (used_weights for each node)
    mu_in: np.ndarray,      # [K]
    sigma_sq_in: np.ndarray,  # [K]
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce gaussian.py:_inference_fn lines 96-99 exactly.

        sigma_sq_out = 1. / jnp.sum(used_weights / sigma_sq_in)
        mu_out = sigma_sq_out * jnp.sum((used_weights * mu_in) / sigma_sq_in)
    """
    sigma_sq_out = 1.0 / (w_active / sigma_sq_in[None, :]).sum(axis=-1)
    mu_out = sigma_sq_out * ((w_active * mu_in[None, :]) / sigma_sq_in[None, :]).sum(axis=-1)
    return mu_out, sigma_sq_out


def _layer_predict_arrays(layer: BatchedGaussianGGMLayer, z_np: np.ndarray):
    """Pull the active weight rows out of the layer for the given side info.

    Mirrors the contexts the layer would compute internally so we can
    feed identical weights to the DeepMind reference math.
    """
    hp = np.asarray(layer.hyperplanes.detach().cpu().tolist(), dtype=np.float32)
    bias = np.asarray(layer.hyperplane_bias.detach().cpu().tolist(), dtype=np.float32)
    dots = np.einsum("nhd,d->nh", hp, z_np) + bias
    bits = (dots >= 0).astype(np.int64)              # same sign convention as ours
    powers = 2 ** np.arange(layer.num_halfspaces)
    contexts = (bits * powers).sum(axis=-1)
    weights = np.asarray(layer.weights.detach().cpu().tolist(), dtype=np.float32)
    n_idx = np.arange(layer.num_nodes)
    return weights[n_idx, contexts]                  # [N, K]


# ---------------------------------------------------------------------------
# Direct parity: same w, same (mu, sigma_sq) -> same (mu_out, sigma_sq_out).
# ---------------------------------------------------------------------------

def test_predict_matches_deepmind_inference_formula() -> None:
    """Our precision-space mixing must equal DeepMind's variance-space mixing.

    Algebraically identical when ``tau = 1/sigma_sq`` and weights are
    positive enough to keep both denominators bounded away from zero.
    """
    layer = BatchedGaussianGGMLayer(
        num_nodes=4,
        num_inputs=6,
        input_dim=5,
        num_halfspaces=2,
        lr=0.05,
        device=torch.device("cpu"),
        seed=314,
    )
    rng = np.random.default_rng(42)
    mu_in = rng.normal(size=6).astype(np.float32)
    sigma_sq_in = (np.abs(rng.normal(size=6)) + 0.3).astype(np.float32)
    z = rng.normal(size=5).astype(np.float32)

    # Our layer: precision form.
    tau_in = 1.0 / sigma_sq_in
    mu_t, tau_t = layer.predict(
        torch.as_tensor(np.asarray(mu_in), dtype=torch.float32),
        torch.as_tensor(np.asarray(tau_in), dtype=torch.float32),
        torch.as_tensor(np.asarray(z), dtype=torch.float32),
    )
    mu_ours = np.asarray(mu_t.tolist(), dtype=np.float32)
    sigma_sq_ours = 1.0 / np.asarray(tau_t.tolist(), dtype=np.float32)

    # DeepMind reference: variance form on the same active weights.
    w_active = _layer_predict_arrays(layer, z)
    mu_dm, sigma_sq_dm = _deepmind_inference(w_active, mu_in, sigma_sq_in)

    np.testing.assert_allclose(mu_ours, mu_dm, atol=1e-5)
    np.testing.assert_allclose(sigma_sq_ours, sigma_sq_dm, atol=1e-5)


def test_predict_parity_holds_on_random_weights() -> None:
    """Same parity must hold for arbitrary (non-uniform) weight matrices."""
    layer = BatchedGaussianGGMLayer(
        num_nodes=8,
        num_inputs=12,
        input_dim=7,
        num_halfspaces=3,
        lr=0.05,
        device=torch.device("cpu"),
        seed=99,
    )
    rng = np.random.default_rng(2026)

    # Replace the uniform-init weights with a random positive matrix.
    random_w = np.abs(rng.normal(size=tuple(layer.weights.shape))).astype(np.float32) + 0.05
    layer.load_weights(torch.as_tensor(np.asarray(random_w), dtype=torch.float32))

    for _ in range(5):
        mu_in = rng.normal(size=12).astype(np.float32)
        sigma_sq_in = (np.abs(rng.normal(size=12)) + 0.5).astype(np.float32)
        z = rng.normal(size=7).astype(np.float32)
        tau_in = 1.0 / sigma_sq_in

        mu_t, tau_t = layer.predict(
            torch.as_tensor(np.asarray(mu_in), dtype=torch.float32),
            torch.as_tensor(np.asarray(tau_in), dtype=torch.float32),
            torch.as_tensor(np.asarray(z), dtype=torch.float32),
        )
        mu_ours = np.asarray(mu_t.tolist(), dtype=np.float32)
        sigma_sq_ours = 1.0 / np.asarray(tau_t.tolist(), dtype=np.float32)

        w_active = _layer_predict_arrays(layer, z)
        mu_dm, sigma_sq_dm = _deepmind_inference(w_active, mu_in, sigma_sq_in)

        np.testing.assert_allclose(mu_ours, mu_dm, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(sigma_sq_ours, sigma_sq_dm, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Update path: gradient direction agrees with autodiff on DeepMind's NLL.
# We don't have jax in this env; instead we run a finite-difference check
# on the closed-form gradient our update uses, treating the DeepMind log-loss
# as the ground truth.
# ---------------------------------------------------------------------------

def _deepmind_log_loss(
    w_active: np.ndarray,
    mu_in: np.ndarray,
    sigma_sq_in: np.ndarray,
    y: float,
) -> np.ndarray:
    """- log N(y | mu_out, sigma_sq_out) per node, using the DeepMind mixer."""
    mu_out, sigma_sq_out = _deepmind_inference(w_active, mu_in, sigma_sq_in)
    return 0.5 * np.log(2.0 * np.pi * sigma_sq_out) + 0.5 * (y - mu_out) ** 2 / sigma_sq_out


def test_update_gradient_matches_finite_difference_of_deepmind_log_loss() -> None:
    """Our closed-form gradient = numerical gradient of DeepMind's log_loss.

    This is the rigorous correctness check: if our hand-derived update
    rule is correct, then a small finite-difference perturbation of
    each weight should change the DeepMind log-loss by exactly the
    dot product with our gradient.
    """
    layer = BatchedGaussianGGMLayer(
        num_nodes=2,
        num_inputs=4,
        input_dim=3,
        num_halfspaces=1,
        lr=0.03,
        device=torch.device("cpu"),
        seed=7,
    )
    rng = np.random.default_rng(11)
    mu_in = rng.normal(size=4).astype(np.float64)
    sigma_sq_in = (np.abs(rng.normal(size=4)) + 0.5).astype(np.float64)
    z = rng.normal(size=3).astype(np.float64)
    y = 0.42
    tau_in = 1.0 / sigma_sq_in

    # Active weight rows BEFORE the update.
    w_active_before = _layer_predict_arrays(layer, z.astype(np.float32)).astype(np.float64)

    # Our closed-form gradient is the (w_before - w_after) / lr scaled by sign.
    # Run one update step and back out the gradient we applied:
    _ = layer.update(
        torch.as_tensor(np.asarray(mu_in.astype(np.float32)), dtype=torch.float32),
        torch.as_tensor(np.asarray(tau_in.astype(np.float32)), dtype=torch.float32),
        torch.as_tensor(np.asarray(z.astype(np.float32)), dtype=torch.float32),
        float(y),
    )
    w_active_after = _layer_predict_arrays(layer, z.astype(np.float32)).astype(np.float64)
    our_grad = (w_active_before - w_active_after) / layer.lr  # [N, K]

    # Finite-difference gradient of the DeepMind log_loss w.r.t each weight.
    eps = 1e-5
    n, k = w_active_before.shape
    fd_grad = np.zeros((n, k))
    for ni in range(n):
        for ki in range(k):
            w_plus = w_active_before.copy()
            w_plus[ni, ki] += eps
            w_minus = w_active_before.copy()
            w_minus[ni, ki] -= eps
            loss_plus = _deepmind_log_loss(w_plus, mu_in, sigma_sq_in, y)[ni]
            loss_minus = _deepmind_log_loss(w_minus, mu_in, sigma_sq_in, y)[ni]
            fd_grad[ni, ki] = (loss_plus - loss_minus) / (2 * eps)

    # The two gradients must agree within finite-difference tolerance.
    # Tolerance accounts for float32 noise in the layer state vs float64 FD.
    np.testing.assert_allclose(our_grad, fd_grad, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Document the two intentional deltas vs the DeepMind reference.
# ---------------------------------------------------------------------------

def test_documented_delta_weight_clamp_is_positive_only() -> None:
    """We clamp weights to [1e-6, 200].  DeepMind clamps to [-1e3, 1e3] plus
    a differentiable projection.  Both ensure ``tau_out > 0`` for valid
    Gaussian outputs; ours is more conservative (drops negative-weight
    expressiveness in exchange for not needing the projection).
    """
    layer = BatchedGaussianGGMLayer(
        num_nodes=1, num_inputs=1, input_dim=1, num_halfspaces=1,
        lr=1e9, device=torch.device("cpu"), seed=0,
    )
    # Force a huge negative gradient via an absurd lr * residual.
    layer.update(
        torch.tensor([1.0]), torch.tensor([1.0]),
        torch.zeros(1),
        y=-1e6,
    )
    # Clamp is at MIN_WEIGHT=1e-6.  In particular, no weight is negative.
    assert (layer.weights >= 0).all()
    assert (layer.weights >= 1e-6 - 1e-8).all()


def test_documented_delta_context_sign_uses_geq_not_strict_gt() -> None:
    """We compute bits with ``>= 0``; DeepMind uses ``> hyperplane_bias``.

    On the measure-zero set ``<hp, z> + bias == 0`` they assign different
    contexts.  In practice, hp is L2-normalised random Gaussian noise and
    z is real-valued input, so equality has probability zero -- but this
    test pins the convention so future readers know which one to use
    when porting.
    """
    layer = BatchedGaussianGGMLayer(
        num_nodes=1, num_inputs=1, input_dim=1, num_halfspaces=1,
        lr=0.01, device=torch.device("cpu"), seed=0,
    )
    # Force <hp, z> + bias == 0 exactly by choosing z=0 and bias=0.
    z = torch.zeros(1)
    contexts = layer._compute_contexts(z)
    # With >= 0, bits[0] = 1, so context = 1.  DeepMind's > would give 0.
    assert contexts.item() == 1
