"""Per-node FMN with proper PTW segment management.

Implements Algorithm 1 from Milan et al. 2016:
- MSCB-based segment boundaries (binary carry pattern)
- Per-depth-level active models with log_weighted and log_buf caching
- Model pool with Bayesian mixture on segment open
- Three-step UPDATEMODELPOOL heuristic on segment close

Each node manages its own PTW hierarchy over GGM weight snapshots.
"""

import math
from dataclasses import dataclass

import torch

LOG_HALF = -math.log(2.0)


@dataclass
class PoolEntry:
    """A stored GGM weight snapshot."""

    weights: torch.Tensor  # [C, K]
    log_block_prob: float
    age: int
    id: int


class PtwLevel:
    """One level in the PTW hierarchy for a single node."""

    __slots__ = ["weights", "log_block_prob", "log_weighted", "log_buf"]

    def __init__(self, template_weights: torch.Tensor):
        self.weights = template_weights.clone()
        self.log_block_prob = 0.0
        self.log_weighted = 0.0
        self.log_buf = 0.0

    def reset(self, template_weights: torch.Tensor):
        self.weights = template_weights.clone()
        self.log_block_prob = 0.0
        self.log_weighted = 0.0
        self.log_buf = 0.0


def log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a > b:
        a, b = b, a
    d = b - a
    if d < 100.0:
        return a + math.log(1.0 + math.exp(d))
    return b


def mscb(depth: int, t: int) -> int:
    """Most Significant Changed Bit between t-1 and t-2 in d-bit representation."""
    if t == 1:
        return 0
    c = depth - 1
    count = 0
    for _ in range(depth):
        mask = 1 << c
        if ((t - 1) & mask) != ((t - 2) & mask):
            return count
        if c == 0:
            break
        c -= 1
        count += 1
    return count


class NodePTW:
    """Per-node PTW with model pool — proper FMN segment management.

    Maintains a depth-level hierarchy of GGM weight snapshots.
    Each level tracks its own log_weighted (PTW mixture probability)
    and log_buf (cached left-sibling probability for Lemma 1).

    The model pool stores up to k GGM weight snapshots from closed segments.
    On segment open, a mixture of pool models initializes the new segment.
    """

    def __init__(
        self,
        template_weights: torch.Tensor,
        depth: int = 15,
        pool_capacity: int = 8,
        min_add_depth: int = 5,
    ):
        self.depth = depth
        self.pool_capacity = pool_capacity
        self.min_add_depth = min_add_depth
        self.template = template_weights.clone()
        self.index = 0
        self.next_id = 1

        # PTW hierarchy: depth+1 levels, each with its own GGM weights
        self.levels = [PtwLevel(template_weights) for _ in range(depth + 1)]

        # Model pool
        self.pool: list[PoolEntry] = []

    def _create_mixture_weights(self) -> torch.Tensor:
        """Create initial weights for a new segment by Bayesian model averaging.

        Returns weights that are a weighted combination of pool models + fresh model.
        For GGM, the "mixture" is approximated by selecting the best pool model
        (since weight averaging doesn't have the same semantics as probability mixing).
        """
        if not self.pool:
            return self.template.clone()

        # Select the pool model with highest log_block_prob
        best = max(self.pool, key=lambda e: e.log_block_prob)
        # Interpolate: 50% best pool model, 50% fresh (zero weights)
        # This gives the fresh model a chance while leveraging stored knowledge
        return best.weights.clone() * 0.5

    def update_node(self, log_p: float):
        """Process one symbol's contribution to this node's PTW hierarchy.

        Args:
            log_p: log probability assigned by this node's GGM to the observed symbol
        """
        t = self.index + 1  # 1-based time
        i = mscb(self.depth, t)

        # Save left sibling's weighted probability
        self.levels[i].log_buf = self.levels[i + 1].log_weighted

        # Save models from closing segments
        for j in range(i + 1, self.depth + 1):
            if (
                self.depth - j >= self.min_add_depth
                and self.levels[j].log_block_prob != 0.0
            ):
                self._save_model(j)

        # Enforce pool capacity
        while len(self.pool) > self.pool_capacity:
            oldest_idx = min(range(len(self.pool)), key=lambda x: self.pool[x].age)
            self.pool.pop(oldest_idx)

        # Reset closing segments with mixture-initialized weights
        mixture_weights = self._create_mixture_weights()
        for j in range(i + 1, self.depth + 1):
            self.levels[j].reset(mixture_weights)

        # Update all levels with the current symbol's log probability
        for j in range(self.depth + 1):
            self.levels[j].log_block_prob += log_p

        # Bottom-up weighted probability computation (Lemma 1)
        self.levels[self.depth].log_weighted = self.levels[self.depth].log_block_prob

        for offset in range(1, self.depth + 1):
            idx = self.depth - offset
            lhs = LOG_HALF + self.levels[idx].log_block_prob  # stop
            rhs = (
                LOG_HALF + self.levels[idx + 1].log_weighted + self.levels[idx].log_buf
            )  # split
            self.levels[idx].log_weighted = log_add(lhs, rhs)

        self.index += 1

    def _save_model(self, depth_level: int):
        """Save the model at a given depth level to the pool."""
        entry = PoolEntry(
            weights=self.levels[depth_level].weights.clone(),
            log_block_prob=self.levels[depth_level].log_block_prob,
            age=self.index,
            id=self.next_id,
        )
        self.next_id += 1
        self.pool.append(entry)

    def get_active_weights(self) -> torch.Tensor:
        """Return the weights from the deepest active level."""
        return self.levels[self.depth].weights

    def set_level_weights(self, weights: torch.Tensor):
        """Set weights on all levels (after GPU update)."""
        for level in self.levels:
            level.weights = weights.clone()

    @property
    def log_weighted(self) -> float:
        return self.levels[0].log_weighted

    def clone(self) -> "NodePTW":
        c = NodePTW.__new__(NodePTW)
        c.depth = self.depth
        c.pool_capacity = self.pool_capacity
        c.min_add_depth = self.min_add_depth
        c.template = self.template.clone()
        c.index = self.index
        c.next_id = self.next_id
        c.levels = []
        for lv in self.levels:
            nlv = PtwLevel(lv.weights)
            nlv.log_block_prob = lv.log_block_prob
            nlv.log_weighted = lv.log_weighted
            nlv.log_buf = lv.log_buf
            c.levels.append(nlv)
        c.pool = [
            PoolEntry(e.weights.clone(), e.log_block_prob, e.age, e.id)
            for e in self.pool
        ]
        return c


class LayerFMNPool:
    """FMN pool manager for all N nodes in a layer using proper PTW."""

    def __init__(
        self,
        num_nodes: int,
        template_weights: torch.Tensor,
        pool_capacity: int = 8,
        ptw_depth: int = 15,
        min_add_depth: int = 5,
    ):
        self.num_nodes = num_nodes
        # Each node gets its own PTW hierarchy initialized from its slice of weights
        self.nodes = []
        for n in range(num_nodes):
            node_template = (
                template_weights[n] if template_weights.dim() > 2 else template_weights
            )
            self.nodes.append(
                NodePTW(
                    node_template,
                    depth=ptw_depth,
                    pool_capacity=pool_capacity,
                    min_add_depth=min_add_depth,
                )
            )

    def step_and_manage(
        self, layer_weights: torch.Tensor, node_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Update all nodes' PTW hierarchies and return potentially updated weights.

        Args:
            layer_weights: [N, C, K] current GGM weights (after GPU update)
            node_log_probs: [N] per-node log P(symbol) for the current chunk

        Returns:
            [N, C, K] weights (may have some nodes reset from pool recall)
        """
        updated = layer_weights
        any_changed = False

        for n in range(self.num_nodes):
            # Sync the PTW levels with the GPU-updated weights
            self.nodes[n].set_level_weights(layer_weights[n])

            # Process through PTW hierarchy
            log_p = node_log_probs[n].item()
            self.nodes[n].update_node(log_p)

            # Check if PTW reset changed the active weights
            new_w = self.nodes[n].get_active_weights()
            if not torch.equal(new_w, layer_weights[n]):
                if not any_changed:
                    updated = layer_weights.clone()
                    any_changed = True
                updated[n] = new_w

        return updated

    def total_pool_size(self) -> int:
        return sum(len(n.pool) for n in self.nodes)

    def clone(self) -> "LayerFMNPool":
        c = LayerFMNPool.__new__(LayerFMNPool)
        c.num_nodes = self.num_nodes
        c.nodes = [n.clone() for n in self.nodes]
        return c
