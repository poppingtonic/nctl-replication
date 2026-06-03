"""Batched Gated Geometric Mixer layer on GPU.

All N nodes in a layer computed in a single batched matmul.
No Python loops over nodes in the forward or update pass.
"""


import torch
import torch.nn.functional as F

EPS = 1e-7
MAX_WEIGHT = 200.0


class BatchedGGMLayer:
    """A layer of N Gated Geometric Mixer nodes, fully batched on GPU.

    Shapes:
        hyperplanes: [N, H, D]  — random gating projections (frozen)
        hyperplane_bias: [N, H] — gating bias (frozen)
        weights: [N, C, K]     — learnable mixing weights
        where H = num_halfspaces, C = 2^H, D = input_dim, K = num_inputs
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
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.input_dim = input_dim
        self.num_halfspaces = num_halfspaces
        self.num_contexts = 2**num_halfspaces
        self.lr = lr
        self.device = device

        gen = torch.Generator(device="cpu").manual_seed(seed)

        # Random gating: [N, H, D] normalized per hyperplane
        hp = torch.randn(num_nodes, num_halfspaces, input_dim, generator=gen)
        hp = F.normalize(hp, dim=-1)
        self.hyperplanes = hp.to(device)

        # Gating bias: [N, H]
        self.hyperplane_bias = torch.zeros(num_nodes, num_halfspaces, device=device)

        # Weights: [N, C, K] init to zero → output = sigmoid(0) = 0.5
        self.weights = torch.zeros(
            num_nodes, self.num_contexts, num_inputs, device=device
        )

        # Per-node accumulated log block probability
        self.log_block_prob = torch.zeros(num_nodes, device=device)

    def _compute_contexts(self, z: torch.Tensor) -> torch.Tensor:
        """Compute context index for each node.

        Args:
            z: [D] side information vector

        Returns:
            [N] long tensor of context indices in [0, C)
        """
        # [N, H, D] @ [D] → [N, H]
        dots = torch.einsum("nhd,d->nh", self.hyperplanes, z) + self.hyperplane_bias
        bits = (dots >= 0).long()  # [N, H]
        # Encode as integer: bit[0]*1 + bit[1]*2 + bit[2]*4 + ...
        powers = (2 ** torch.arange(self.num_halfspaces, device=self.device)).long()
        return (bits * powers).sum(dim=-1)  # [N]

    def _select_weights(self, contexts: torch.Tensor) -> torch.Tensor:
        """Select active weight row for each node.

        Args:
            contexts: [N] context indices

        Returns:
            [N, K] active weight vectors
        """
        # Advanced indexing: weights[n, contexts[n], :]
        return self.weights[torch.arange(self.num_nodes, device=self.device), contexts]

    def predict(self, input_probs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Batched forward pass for all N nodes.

        Args:
            input_probs: [K] input probabilities from previous layer
            z: [D] side information

        Returns:
            [N] predicted P(symbol=1) for each node
        """
        contexts = self._compute_contexts(z)  # [N]
        w = self._select_weights(contexts)  # [N, K]
        logits = torch.logit(input_probs.clamp(EPS, 1 - EPS))  # [K]
        # [N, K] @ [K] → [N]  (batch dot product via matmul)
        return torch.sigmoid((w * logits).sum(dim=-1))

    def update(
        self, input_probs: torch.Tensor, z: torch.Tensor, symbol: int
    ) -> torch.Tensor:
        """Batched forward + gradient step for all N nodes.

        Args:
            input_probs: [K] input probabilities
            z: [D] side information
            symbol: 0 or 1

        Returns:
            [N] predictive P(symbol) for each node (before weight update)
        """
        contexts = self._compute_contexts(z)  # [N]
        w = self._select_weights(contexts)  # [N, K]
        logits = torch.logit(input_probs.clamp(EPS, 1 - EPS))  # [K]

        p1 = torch.sigmoid((w * logits).sum(dim=-1))  # [N]
        p_sym = p1 if symbol == 1 else 1.0 - p1  # [N]

        # Accumulate log block probability
        self.log_block_prob += torch.log(p_sym.clamp(min=1e-300))

        # Online gradient descent: w += lr * (target - p1) * logits
        error = float(symbol) - p1  # [N]
        # [N, 1] * [K] → [N, K]
        delta = self.lr * error.unsqueeze(-1) * logits.unsqueeze(0)

        # Scatter-add into the correct context rows
        # weights[n, contexts[n], :] += delta[n, :]
        n_idx = torch.arange(self.num_nodes, device=self.device)
        self.weights[n_idx, contexts] = (self.weights[n_idx, contexts] + delta).clamp(
            -MAX_WEIGHT, MAX_WEIGHT
        )

        return p_sym

    def snapshot_weights(self) -> torch.Tensor:
        """Return a detached copy of the weight tensor [N, C, K]."""
        return self.weights.detach().clone()

    def load_weights(self, weights: torch.Tensor):
        """Load weights from a snapshot."""
        self.weights = weights.clone()

    def reset_weights(self):
        """Reset all weights to zero."""
        self.weights.zero_()
        self.log_block_prob.zero_()

    def clone(self) -> "BatchedGGMLayer":
        """Deep copy of this layer."""
        c = BatchedGGMLayer.__new__(BatchedGGMLayer)
        c.num_nodes = self.num_nodes
        c.num_inputs = self.num_inputs
        c.input_dim = self.input_dim
        c.num_halfspaces = self.num_halfspaces
        c.num_contexts = self.num_contexts
        c.lr = self.lr
        c.device = self.device
        c.hyperplanes = self.hyperplanes  # shared, read-only
        c.hyperplane_bias = self.hyperplane_bias  # shared, read-only
        c.weights = self.weights.clone()
        c.log_block_prob = self.log_block_prob.clone()
        return c
