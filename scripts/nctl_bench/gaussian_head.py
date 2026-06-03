"""Gaussian Gated Linear Network head (bd forget-me-not-h2u.31).

A batched Gaussian GLN node layer that mirrors the public API of the
Bernoulli ``BatchedGGMLayer`` (``predict``, ``update``, ``snapshot_weights``,
``load_weights``, ``reset_weights``, ``clone``).  The downstream consumers
in h2u.34 (Numerai), h2u.32 (Fistful/Atari), and h2u.33 (SAE) can plug
this in wherever the Bernoulli head is wired today, by swapping
``input_probs / symbol`` (Bernoulli scalars) for
``(input_mus, input_taus) / y_target`` (continuous Gaussian).

Math (precision parameterisation; sticks closer to numerical stability
than the natural-parameter form in Veness et al. 2021):

  * Each input k emits a Gaussian with mean mu_k and precision tau_k
    (tau_k = 1 / sigma_k^2; strictly positive).
  * Per context c, the node has K non-negative mixing weights w_{c,k}.
  * Output predictive distribution under geometric mixing:
      tau_out         = sum_k  w_{c,k} * tau_k
      tau_out * mu_out= sum_k  w_{c,k} * tau_k * mu_k
      mu_out          = (sum_k w_{c,k} tau_k mu_k) / tau_out
  * Online natural-gradient step on the Gaussian NLL w.r.t. the weights:
      d NLL / d w_{c,k} = tau_k * (mu_k - mu_out) * (mu_out - y)
                         + (tau_k / 2) * ((y - mu_out)^2 - 1 / tau_out)
    Sign of each term checked against the NumPy reference parity test.
  * After the step we clamp w_{c,k} >= eps (positive weights ensure
    tau_out > 0; the paper uses softplus reparam, but a clamp is
    simpler and the parity test pins exact agreement either way).

API parity notes vs ``BatchedGGMLayer``:

  * ``predict(input_mus, input_taus, z) -> (mu_out, tau_out)`` instead of
    Bernoulli's ``predict(input_probs, z) -> p``.
  * ``update(input_mus, input_taus, z, y) -> (mu_out_before, tau_out_before)``
    instead of ``update(input_probs, z, symbol) -> p_sym``.
  * ``log_block_prob`` accumulates Gaussian log p(y | mu_out, tau_out) per
    node, matching the Bernoulli surface 1:1 for downstream FMN bookkeeping.

This is a CPU/PyTorch reference implementation; the CUDA kernel parity
is intentionally deferred to a follow-up (per the recorded suggestion on
h2u.31).  Downstream h2u.34.2 only needs the algorithmic head to be
correct; throughput tuning can land later without breaking the wrapper.

Reference
~~~~~~~~~

The canonical DeepMind implementation lives at
``deepmind-research/gated_linear_networks/gaussian.py`` (jax + haiku;
Apache-2.0).  Our predict path is mathematically identical to its
``_inference_fn`` -- the only delta is the precision vs variance
parameterisation:

    ours (tau-space):       tau_out = sum_k w_k * tau_k
    DeepMind (sigma_sq):    sigma_sq_out = 1 / sum_k (w_k / sigma_sq_k)

Setting ``tau_k = 1 / sigma_sq_k`` makes the two formulas literally
equal; ``test_gaussian_head_deepmind_parity.py`` pins this in tests.
Our update is derived in closed form against the Gaussian NLL the
DeepMind reference computes via ``jax.grad`` of a TFP log-prob;
``test_update_gradient_matches_finite_difference_of_deepmind_log_loss``
verifies the two gradients agree.

Two intentional deviations (also pinned by tests):

1. **Weight clamp.**  We use positive-only weights ``[1e-6, 200]``;
   DeepMind uses signed weights ``[-1e3, 1e3]`` plus a differentiable
   hard projection to keep ``sigma_sq_out`` bounded in
   ``[0.5, 1e5]``.  Both choices keep ``tau_out > 0``; ours drops the
   negative-weight expressiveness in exchange for not needing the
   projection (simpler gradient, no special cases at the boundary).
2. **Context sign.**  We compute bits with ``>= 0``; DeepMind uses
   ``> hyperplane_bias``.  On the measure-zero set where the gate
   evaluates to exactly the bias the two disagree; with L2-normalised
   random gates and real-valued side info this case has probability
   zero, but the convention is pinned so future ports stay consistent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Numerical guards.  Chosen to match the Bernoulli head's spirit (small
# enough not to mask real signal, large enough to keep weights and
# precisions inside the float32 range under the worst-case input stream).
_MIN_WEIGHT = 1e-6      # keeps tau_out > 0 even when one input dominates
_MAX_WEIGHT = 200.0     # mirrors BatchedGGMLayer.MAX_WEIGHT
_MIN_TAU = 1e-6         # input precision floor
_MAX_TAU = 1e6          # input precision ceiling
_2PI = 6.283185307179586


class BatchedGaussianGGMLayer:
    """A layer of N batched Gaussian GLN nodes (CPU/PyTorch reference).

    Tensor shapes:

      * ``hyperplanes``       : ``[N, H, D]``  random gating projections
      * ``hyperplane_bias``   : ``[N, H]``     gating bias
      * ``weights``           : ``[N, C, K]``  non-negative mixing weights
        per-(node, context, input).  Initialised to ``1/K`` so the
        un-trained predictor returns the precision-weighted average of
        its inputs -- a sensible non-informative prior in precision
        space.
      * ``log_block_prob``    : ``[N]``        accumulated Gaussian log p(y).
    """

    def __init__(
        self,
        num_nodes: int,
        num_inputs: int,
        input_dim: int,
        num_halfspaces: int,
        lr: float,
        device: torch.device,
        seed: int = 42,
    ):
        if num_inputs <= 0:
            raise ValueError(f"num_inputs must be positive; got {num_inputs}")
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.input_dim = input_dim
        self.num_halfspaces = num_halfspaces
        self.num_contexts = 2**num_halfspaces
        self.lr = float(lr)
        self.device = device

        gen = torch.Generator(device="cpu").manual_seed(seed)
        hp = torch.randn(num_nodes, num_halfspaces, input_dim, generator=gen)
        hp = F.normalize(hp, dim=-1)
        self.hyperplanes = hp.to(device)
        self.hyperplane_bias = torch.zeros(num_nodes, num_halfspaces, device=device)

        # Weights init at 1/K -- precision-weighted average of the
        # incoming Gaussians.  Strictly positive so tau_out > 0 from
        # step 0.
        init = 1.0 / float(num_inputs)
        self.weights = torch.full(
            (num_nodes, self.num_contexts, num_inputs),
            init,
            device=device,
        )

        self.log_block_prob = torch.zeros(num_nodes, device=device)

    # --- internal helpers --------------------------------------------------

    def _compute_contexts(self, z: torch.Tensor) -> torch.Tensor:
        """Context index in ``[0, C)`` per node from side information z."""
        dots = torch.einsum("nhd,d->nh", self.hyperplanes, z) + self.hyperplane_bias
        bits = (dots >= 0).long()
        powers = (2 ** torch.arange(self.num_halfspaces, device=self.device)).long()
        return (bits * powers).sum(dim=-1)

    def _select_weights(self, contexts: torch.Tensor) -> torch.Tensor:
        """Active weight row ``[N, K]`` for the per-node contexts."""
        n_idx = torch.arange(self.num_nodes, device=self.device)
        return self.weights[n_idx, contexts]

    @staticmethod
    def _mix(
        w: torch.Tensor,
        mus: torch.Tensor,
        taus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Geometric mixing in precision space.

        Args:
            w: ``[N, K]`` non-negative mixing weights.
            mus: ``[K]`` input means.
            taus: ``[K]`` input precisions, strictly positive.

        Returns:
            ``(mu_out, tau_out)`` each ``[N]``.
        """
        w_tau = w * taus.unsqueeze(0)                       # [N, K]
        tau_out = w_tau.sum(dim=-1).clamp(min=_MIN_TAU)     # [N]
        mu_out = (w_tau * mus.unsqueeze(0)).sum(dim=-1) / tau_out
        return mu_out, tau_out

    # --- public API --------------------------------------------------------

    def predict(
        self,
        input_mus: torch.Tensor,
        input_taus: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched forward pass for all N nodes.

        Args:
            input_mus: ``[K]`` input means (from the previous layer).
            input_taus: ``[K]`` input precisions, strictly positive.
            z: ``[D]`` side information.

        Returns:
            ``(mu_out, tau_out)`` each ``[N]``.  Read-only call; no
            weight update.
        """
        taus = input_taus.clamp(_MIN_TAU, _MAX_TAU)
        contexts = self._compute_contexts(z)
        w = self._select_weights(contexts)
        return self._mix(w, input_mus, taus)

    def update(
        self,
        input_mus: torch.Tensor,
        input_taus: torch.Tensor,
        z: torch.Tensor,
        y: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass + per-node SGD step on Gaussian NLL.

        Returns the ``(mu_out, tau_out)`` computed BEFORE the weight
        update (so the caller can score against ``y`` consistently with
        the recursion happening upstream).  ``log_block_prob`` is
        accumulated with the pre-update predictive log-density.
        """
        taus = input_taus.clamp(_MIN_TAU, _MAX_TAU)
        contexts = self._compute_contexts(z)               # [N]
        w = self._select_weights(contexts)                  # [N, K]
        mu_out, tau_out = self._mix(w, input_mus, taus)    # [N], [N]

        # Pre-update predictive log-density: log N(y | mu_out, 1/tau_out)
        #   = 0.5 * log(tau_out / 2pi) - 0.5 * tau_out * (y - mu_out)^2
        if not isinstance(y, torch.Tensor):
            y_t = torch.as_tensor(float(y), device=self.device)
        else:
            y_t = y.to(self.device)
        residual = mu_out - y_t                             # [N]
        log_p = 0.5 * (torch.log(tau_out) - torch.log(torch.tensor(_2PI, device=self.device))) \
            - 0.5 * tau_out * residual * residual
        self.log_block_prob += log_p

        # Gradient of NLL w.r.t. w_{c,k}.  Each node has its own context,
        # so the update is row-selected.  Broadcast input dimensions over
        # the [N, K] active block:
        #   g_{n,k} = tau_k * (mu_k - mu_out_n) * (mu_out_n - y)
        #             + 0.5 * tau_k * ((y - mu_out_n)^2 - 1/tau_out_n)
        mu_diff = input_mus.unsqueeze(0) - mu_out.unsqueeze(-1)   # [N, K]
        term1 = taus.unsqueeze(0) * mu_diff * (mu_out - y_t).unsqueeze(-1)
        term2 = 0.5 * taus.unsqueeze(0) * (
            (y_t - mu_out).square().unsqueeze(-1) - (1.0 / tau_out).unsqueeze(-1)
        )
        grad = term1 + term2                                # [N, K]

        new_w = (w - self.lr * grad).clamp(_MIN_WEIGHT, _MAX_WEIGHT)
        n_idx = torch.arange(self.num_nodes, device=self.device)
        self.weights[n_idx, contexts] = new_w

        return mu_out, tau_out

    # --- snapshot / restore parity with Bernoulli head --------------------

    def snapshot_weights(self) -> torch.Tensor:
        return self.weights.detach().clone()

    def load_weights(self, weights: torch.Tensor) -> None:
        if weights.shape != self.weights.shape:
            raise ValueError(
                f"load_weights shape mismatch: got {tuple(weights.shape)}, "
                f"expected {tuple(self.weights.shape)}"
            )
        self.weights = weights.clone()

    def reset_weights(self) -> None:
        init = 1.0 / float(self.num_inputs)
        self.weights.fill_(init)
        self.log_block_prob.zero_()

    def clone(self) -> BatchedGaussianGGMLayer:
        c = BatchedGaussianGGMLayer.__new__(BatchedGaussianGGMLayer)
        c.num_nodes = self.num_nodes
        c.num_inputs = self.num_inputs
        c.input_dim = self.input_dim
        c.num_halfspaces = self.num_halfspaces
        c.num_contexts = self.num_contexts
        c.lr = self.lr
        c.device = self.device
        c.hyperplanes = self.hyperplanes
        c.hyperplane_bias = self.hyperplane_bias
        c.weights = self.weights.clone()
        c.log_block_prob = self.log_block_prob.clone()
        return c


__all__ = ["BatchedGaussianGGMLayer"]
