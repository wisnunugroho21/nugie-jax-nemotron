"""
Minimal Sparse Mixture-of-Experts (MoE) in JAX/Flax NNX.

This module follows the Nemotron-style MoE idea in a simple educational form:
- Router picks top-k routed experts per token
- Shared experts are always active
- Router uses sigmoid scores
- Experts use Squared-ReLU activation
- Includes a load-balancing auxiliary loss for routed experts

The implementation is intentionally explicit (loop-based) for readability.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class MoEExpert(nnx.Module):
    """
    A single feed-forward expert with Squared-ReLU activation.

    Structure:
        x -> Linear(d_model, hidden_dim) -> SquaredReLU -> Linear(hidden_dim, d_model)
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        hidden_dim: int,
        use_bias: bool = False,
    ):
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.fc1 = nnx.Linear(
            self.d_model, self.hidden_dim, use_bias=use_bias, rngs=rngs
        )
        self.fc2 = nnx.Linear(
            self.hidden_dim, self.d_model, use_bias=use_bias, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Squared-ReLU as used in the Nemotron paper.
        h = self.fc1(x)
        h = jax.nn.relu(h)
        h = h * h
        return self.fc2(h)


class SparseMoE(nnx.Module):
    """
    Minimal sparse top-k MoE with optional shared experts.

    Routing details:
    - A router projects each token to expert scores.
    - Routed experts: only top-k are used per token.
    - Shared experts: always active and combined for every token.

    This version keeps logic simple and avoids optimization tricks.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        use_bias: bool = False,
    ):
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.expert_hidden_dim = expert_hidden_dim

        assert self.num_experts > 0, "num_experts must be > 0"
        assert self.top_k > 0, "top_k must be > 0"
        assert self.top_k <= self.num_experts, "top_k must be <= num_experts"
        assert self.num_shared_experts >= 0, "num_shared_experts must be >= 0"

        # Router predicts scores for both routed and shared experts.
        self.router = nnx.Linear(
            self.d_model,
            self.num_experts + self.num_shared_experts,
            use_bias=use_bias,
            rngs=rngs,
        )

        # Routed experts (sparse top-k selection).
        for i in range(self.num_experts):
            setattr(
                self,
                f"routed_expert_{i}",
                MoEExpert(
                    d_model=self.d_model,
                    hidden_dim=self.expert_hidden_dim,
                    use_bias=use_bias,
                    rngs=rngs,
                ),
            )

        # Shared experts (always active).
        for i in range(self.num_shared_experts):
            setattr(
                self,
                f"shared_expert_{i}",
                MoEExpert(
                    d_model=self.d_model,
                    hidden_dim=self.expert_hidden_dim,
                    use_bias=use_bias,
                    rngs=rngs,
                ),
            )

    def _collect_routed_outputs(self, x_flat: jax.Array) -> jax.Array:
        """
        Runs all routed experts and stacks their outputs.

        Args:
            x_flat: (num_tokens, d_model)
        Returns:
            routed_outputs: (num_tokens, num_experts, d_model)
        """
        outputs = []
        for i in range(self.num_experts):
            expert = getattr(self, f"routed_expert_{i}")
            outputs.append(expert(x_flat))
        return jnp.stack(outputs, axis=1)

    def _collect_shared_outputs(self, x_flat: jax.Array) -> jax.Array:
        """
        Runs all shared experts and stacks their outputs.

        Args:
            x_flat: (num_tokens, d_model)
        Returns:
            shared_outputs: (num_tokens, num_shared_experts, d_model)
        """
        if self.num_shared_experts == 0:
            return jnp.zeros((x_flat.shape[0], 0, self.d_model), dtype=x_flat.dtype)

        outputs = []
        for i in range(self.num_shared_experts):
            expert = getattr(self, f"shared_expert_{i}")
            outputs.append(expert(x_flat))
        return jnp.stack(outputs, axis=1)

    def _load_balancing_aux_loss(
        self, routed_scores: jax.Array, topk_indices: jax.Array
    ) -> jax.Array:
        """
        Computes a simple load-balancing auxiliary loss for routed experts.

        We combine two signals per expert:
        1) Dispatch fraction: how often tokens are actually sent to that expert.
        2) Mean router probability: how much probability mass the router gives it.

        The product encourages experts to receive both routing probability and
        actual token assignments in a more balanced way.

        Args:
            routed_scores: Sigmoid router scores for routed experts,
                           shape (num_tokens, num_experts)
            topk_indices: Top-k expert indices per token,
                          shape (num_tokens, top_k)
        Returns:
            aux_loss: Scalar load-balancing loss.
        """
        num_tokens = routed_scores.shape[0]

        # Convert scores into per-token normalized probabilities.
        routed_probs = routed_scores / (
            jnp.sum(routed_scores, axis=-1, keepdims=True) + 1e-6
        )

        # Build a binary dispatch mask from top-k assignments.
        dispatch_mask = jnp.zeros_like(routed_scores)
        token_ids = jnp.arange(num_tokens)[:, None]
        dispatch_mask = dispatch_mask.at[token_ids, topk_indices].set(1.0)

        # Each token contributes total mass 1.0 across its selected experts.
        dispatch_fraction = jnp.mean(dispatch_mask / self.top_k, axis=0)
        mean_router_prob = jnp.mean(routed_probs, axis=0)

        # Switch-style balancing form, scaled by number of experts.
        aux_loss = self.num_experts * jnp.sum(dispatch_fraction * mean_router_prob)
        return aux_loss

    def __call__(
        self, x: jax.Array, return_aux_loss: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """
        Args:
            x: (batch, seqlen, d_model)
            return_aux_loss: If True, also returns routed expert balancing loss.
        Returns:
            y: (batch, seqlen, d_model)
            aux_loss (optional): scalar load-balancing loss
        """
        batch, seqlen, d_model = x.shape
        assert d_model == self.d_model, "Input d_model does not match MoE config"

        # Flatten tokens for per-token routing.
        num_tokens = batch * seqlen
        x_flat = jnp.reshape(x, (num_tokens, d_model))

        # Router scores (sigmoid gating as in the paper).
        router_scores = jax.nn.sigmoid(self.router(x_flat))

        routed_scores = router_scores[:, : self.num_experts]
        shared_scores = router_scores[:, self.num_experts :]

        # Select top-k routed experts per token.
        topk_values, topk_indices = jax.lax.top_k(routed_scores, self.top_k)

        # Load-balancing auxiliary loss only uses routed experts.
        aux_loss = self._load_balancing_aux_loss(routed_scores, topk_indices)

        # Build sparse top-k gate matrix for routed experts.
        routed_gates = jnp.zeros_like(routed_scores)
        token_ids = jnp.arange(num_tokens)[:, None]
        routed_gates = routed_gates.at[token_ids, topk_indices].set(topk_values)

        # Normalize routed gates to keep token output scale stable.
        routed_gates = routed_gates / (
            jnp.sum(routed_gates, axis=-1, keepdims=True) + 1e-6
        )

        # Run all routed experts, then apply sparse gates.
        routed_outputs = self._collect_routed_outputs(x_flat)
        routed_mix = jnp.sum(routed_outputs * routed_gates[:, :, None], axis=1)

        # Shared experts are always active and softly combined.
        if self.num_shared_experts > 0:
            shared_outputs = self._collect_shared_outputs(x_flat)
            shared_gates = shared_scores / (
                jnp.sum(shared_scores, axis=-1, keepdims=True) + 1e-6
            )
            shared_mix = jnp.sum(shared_outputs * shared_gates[:, :, None], axis=1)

            # Combine routed and shared pathways.
            y_flat = 0.5 * (routed_mix + shared_mix)
        else:
            y_flat = routed_mix

        y = jnp.reshape(y_flat, (batch, seqlen, d_model))
        if return_aux_loss:
            return y, aux_loss
        return y
