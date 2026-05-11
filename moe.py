"""
Minimal Sparse Mixture-of-Experts (MoE) in JAX/Flax NNX.

Based on Nemotron 3 Nano (arXiv:2512.20848, §2.1).

Key design choices from the paper:

1. Granular routed experts (DeepSeekMoE style, Dai et al. 2024):
   Each "base" expert is split into finer-grained smaller experts.
   Total routed experts = num_experts * granularity_factor,
   each with hidden_dim = expert_hidden_dim / granularity_factor.
   This improves expert specialization without changing total parameter count.

2. Shared experts:
   Always-on FFN experts that run for every token, unconditionally.
   They provide a stable shared capacity outside of routing competition.
   Shared experts keep the full expert_hidden_dim.

3. Sigmoid gating (Nemotron-specific, unlike most MoE models that use softmax):
   Gate scores are produced independently per expert via sigmoid.
   This means experts do NOT compete with each other for probability mass.
   After top-k selection, selected scores are renormalized to sum to 1
   so that the combined output has a stable scale.

4. Squared-ReLU activation inside each expert FFN:
   relu(x)^2 — a stronger nonlinearity than plain ReLU.

5. Standard GShard/Switch-style load-balancing auxiliary loss (Lepikhin et al. 2020):
   Encourages tokens to be spread evenly across routed experts.
   Uses raw sigmoid scores (not normalized) for the mean routing signal,
   consistent with the independent-score nature of sigmoid gating.

6. No bias on any linear layers (per paper).

The implementation is explicit and loop-based for readability, not performance.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class MoEExpert(nnx.Module):
    """
    A single FFN expert with Squared-ReLU activation.

    The Nemotron 3 Nano paper specifies squared-ReLU for all expert FFNs.
    The computation is:
        h = fc1(x)          # expand to hidden dimension
        h = relu(h) ** 2    # squared-ReLU: zero negatives, then square
        out = fc2(h)        # compress back to model dimension

    Squaring after ReLU amplifies large activations more than small ones,
    which acts as a stronger gate and improves expert specialization.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        hidden_dim: int,
        use_bias: bool = False,  # Paper: no bias on linear layers
    ):
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Gate (up) projection: expand token to the expert's hidden dimension.
        self.fc1 = nnx.Linear(
            self.d_model, self.hidden_dim, use_bias=use_bias, rngs=rngs
        )
        # Down projection: compress back to the model dimension.
        self.fc2 = nnx.Linear(
            self.hidden_dim, self.d_model, use_bias=use_bias, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.fc1(x)
        # Squared-ReLU: first zero out negatives with ReLU, then square.
        # relu(x)^2 creates a sparser, more non-linear activation than plain ReLU.
        h = jax.nn.relu(h)
        h = h * h  # element-wise square
        return self.fc2(h)


class SparseMoE(nnx.Module):
    """
    Sparse MoE layer matching the Nemotron 3 Nano design (arXiv:2512.20848, §2.1).

    --- Granular experts (DeepSeekMoE style) ---
    Instead of a few large experts, we use many small, fine-grained experts.
    Given `num_experts` base experts and a `granularity_factor` g:
      - Actual routed expert count = num_experts * g
      - Each routed expert hidden dim = expert_hidden_dim / g
      - Top-k = top_k * g   (if scale_top_k_with_granularity=True)
    Total FLOPs per token stays the same, but with more diverse expert paths.
    In Nemotron 3 Nano: 128 total routable experts, 6 activated per token.

    --- Shared experts ---
    `num_shared_experts` always-on FFN experts run on every token.
    They are NOT subject to routing — they always contribute to the output.
    Their outputs are summed and added to the routed path output.
    Shared experts keep the full expert_hidden_dim (not reduced by granularity).
    In Nemotron 3 Nano: 2 shared experts.

    --- Sigmoid routing (Nemotron-specific) ---
    Router logits -> sigmoid -> top-k selection.
    With softmax (most MoEs): experts compete; picking one raises another's cost.
    With sigmoid: scores are independent; each expert is scored on its own merit.
    After top-k, selected scores are renormalized to sum to 1 for stable output scale.

    --- Load-balancing auxiliary loss ---
    Without regularization, the router learns to always pick a few "easy" experts
    (expert collapse). The load-balancing loss penalizes this by rewarding
    even token distribution across all routed experts.

    Args:
        granularity_factor:
            1 = standard MoE (no granularity).
            >1 = each base expert is split into this many smaller experts.
        scale_top_k_with_granularity:
            True (default): effective top-k = top_k * granularity_factor.
            False: keep top-k fixed regardless of granularity.
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
        granularity_factor: int = 1,
        scale_top_k_with_granularity: bool = True,
    ):
        self.d_model = d_model

        # Keep base values for reference.
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.expert_hidden_dim = expert_hidden_dim

        self.granularity_factor = granularity_factor
        self.scale_top_k_with_granularity = scale_top_k_with_granularity

        assert self.num_experts > 0, "num_experts must be > 0"
        assert self.top_k > 0, "top_k must be > 0"
        assert self.top_k <= self.num_experts, "top_k must be <= num_experts"
        assert self.num_shared_experts >= 0, "num_shared_experts must be >= 0"
        assert self.granularity_factor > 0, "granularity_factor must be > 0"

        # Total fine-grained routed experts after granular splitting.
        # e.g. 16 base experts * 8 granularity_factor = 128 (as in Nemotron 3 Nano).
        self.num_routed_experts = self.num_experts * self.granularity_factor

        # Scale top-k proportionally so the same fraction of capacity is activated.
        # e.g. top_k=1 with granularity=6 → select 6 out of 128 experts.
        if self.scale_top_k_with_granularity:
            self.routed_top_k = self.top_k * self.granularity_factor
        else:
            self.routed_top_k = self.top_k

        assert self.routed_top_k <= self.num_routed_experts, (
            "effective routed top-k must be <= num_routed_experts"
        )

        # Granular routed experts are narrower to keep total parameter count stable.
        # e.g. expert_hidden_dim=1856, granularity=8 → each expert hidden = 232.
        self.routed_expert_hidden_dim = max(
            1, self.expert_hidden_dim // self.granularity_factor
        )

        # Shared experts keep the full hidden dimension.
        # They're meant to model general token features, so they stay large.
        self.shared_expert_hidden_dim = self.expert_hidden_dim

        # Router: a single linear layer mapping each token to one logit per routed expert.
        # No bias per paper. Shared experts are NOT routed — they bypass this.
        self.router = nnx.Linear(
            self.d_model,
            self.num_routed_experts,
            use_bias=use_bias,
            rngs=rngs,
        )

        # Instantiate all fine-grained routed experts.
        for i in range(self.num_routed_experts):
            setattr(
                self,
                f"routed_expert_{i}",
                MoEExpert(
                    d_model=self.d_model,
                    hidden_dim=self.routed_expert_hidden_dim,
                    use_bias=use_bias,
                    rngs=rngs,
                ),
            )

        # Instantiate all always-on shared experts.
        for i in range(self.num_shared_experts):
            setattr(
                self,
                f"shared_expert_{i}",
                MoEExpert(
                    d_model=self.d_model,
                    hidden_dim=self.shared_expert_hidden_dim,
                    use_bias=use_bias,
                    rngs=rngs,
                ),
            )

    def _collect_routed_outputs(self, x_flat: jax.Array) -> jax.Array:
        """
        Run every routed expert on every token, then stack the results.

        For simplicity, all experts run on all tokens even if not selected.
        The routing gates (0 for non-selected experts) will zero out those
        outputs when we compute the weighted sum in __call__.

        Args:
            x_flat: (num_tokens, d_model)
        Returns:
            routed_outputs: (num_tokens, num_routed_experts, d_model)
        """
        outputs = []
        for i in range(self.num_routed_experts):
            expert = getattr(self, f"routed_expert_{i}")
            outputs.append(expert(x_flat))
        return jnp.stack(outputs, axis=1)

    def _collect_shared_outputs(self, x_flat: jax.Array) -> jax.Array:
        """
        Run every shared expert on every token, then stack the results.

        Shared experts are always active — no routing decision is made.
        Their outputs will be summed in __call__ to form a combined shared signal.

        Args:
            x_flat: (num_tokens, d_model)
        Returns:
            shared_outputs: (num_tokens, num_shared_experts, d_model),
                            or (num_tokens, 0, d_model) if there are no shared experts.
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
        Standard GShard/Switch-style load-balancing auxiliary loss (Lepikhin et al. 2020).

        Without this loss the router tends to collapse: it repeatedly picks a small
        set of "easy" experts while the rest are never used (expert collapse).

        The loss encourages balance by penalizing the combination of:
          - dispatch_fraction[i]: the fraction of tokens actually routed to expert i
          - mean_routing_score[i]: the average sigmoid score the router assigns to expert i

        If expert i is used a lot AND the router rates it highly, both terms are large,
        producing a large loss. Minimizing this pushes the router to spread tokens out.

        Formula (Lepikhin et al. 2020 / Switch Transformer):
            L_balance = num_experts * sum_i( dispatch_fraction_i * mean_routing_score_i )

        Important: we use the raw sigmoid scores (not normalized) for mean_routing_score.
        Nemotron 3 Nano uses sigmoid gating where each expert score is independent.
        Normalizing the scores (e.g. dividing by their sum) would create an artificial
        softmax-like distribution that misrepresents how sigmoid routing actually works.

        Args:
            routed_scores: sigmoid scores for all routed experts,
                           shape (num_tokens, num_routed_experts)
            topk_indices:  indices of the selected top-k experts per token,
                           shape (num_tokens, routed_top_k)
        Returns:
            aux_loss: scalar
        """
        num_tokens = routed_scores.shape[0]

        # Build a binary dispatch mask: dispatch_mask[t, i] = 1 if token t
        # was routed to expert i, 0 otherwise.
        dispatch_mask = jnp.zeros_like(routed_scores)
        token_ids = jnp.arange(num_tokens)[:, None]
        dispatch_mask = dispatch_mask.at[token_ids, topk_indices].set(1.0)

        # dispatch_fraction[i] = fraction of routing slots going to expert i.
        # Dividing by routed_top_k ensures the fractions sum to 1.0 over all experts.
        # (Each token contributes 1/routed_top_k to each of its top-k experts.)
        dispatch_fraction = jnp.mean(dispatch_mask / self.routed_top_k, axis=0)

        # mean_routing_score[i] = average sigmoid score assigned to expert i across tokens.
        # We use the raw sigmoid scores directly — they reflect the router's independent
        # judgment of each expert, without any cross-expert normalization.
        mean_routing_score = jnp.mean(routed_scores, axis=0)

        # Scale by num_experts so the loss magnitude stays roughly constant
        # regardless of how many experts there are.
        aux_loss = self.num_routed_experts * jnp.sum(
            dispatch_fraction * mean_routing_score
        )
        return aux_loss

    def __call__(
        self, x: jax.Array, return_aux_loss: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """
        Forward pass through the sparse MoE layer.

        Overview:
            1. Router assigns one sigmoid score per routed expert, per token.
            2. Top-k selection picks the best k routed experts for each token.
            3. Selected sigmoid scores are renormalized to sum to 1 (output scale stability).
            4. Weighted sum of selected expert outputs forms the routed path.
            5. All shared experts run unconditionally; their outputs are summed.
            6. Routed path + shared path = final output.

        Args:
            x: (batch, seqlen, d_model)
            return_aux_loss: if True, also return the load-balancing auxiliary loss.
                             Pass this loss coefficient * aux_loss to the optimizer.

        Returns:
            y: (batch, seqlen, d_model)
            aux_loss (optional): scalar — only returned when return_aux_loss=True
        """
        batch, seqlen, d_model = x.shape
        assert d_model == self.d_model, "Input d_model does not match MoE config"

        # MoE routing is purely token-wise, so we flatten batch and sequence together.
        num_tokens = batch * seqlen
        x_flat = jnp.reshape(x, (num_tokens, d_model))  # (num_tokens, d_model)

        # ── Routed path ────────────────────────────────────────────────────────

        # Step 1: Compute one routing logit per expert for each token.
        routed_logits = self.router(x_flat)  # (num_tokens, num_routed_experts)

        # Step 2: Apply sigmoid to get independent gate scores.
        # Unlike softmax, sigmoid does NOT create a probability distribution.
        # Each expert's score is judged independently; scores do not sum to 1.
        # This is the "sigmoid gating" described in the Nemotron 3 Nano paper.
        routed_scores = jax.nn.sigmoid(routed_logits)  # (num_tokens, num_routed_experts)

        # Step 3: Select the top-k experts for each token.
        # topk_values:  (num_tokens, routed_top_k) — their sigmoid scores
        # topk_indices: (num_tokens, routed_top_k) — which expert indices were chosen
        topk_values, topk_indices = jax.lax.top_k(routed_scores, self.routed_top_k)

        # Step 4: Build a sparse gate matrix of shape (num_tokens, num_routed_experts).
        # Non-selected experts get gate=0; selected experts get their sigmoid score.
        token_ids = jnp.arange(num_tokens)[:, None]
        routed_gates = jnp.zeros_like(routed_scores)
        routed_gates = routed_gates.at[token_ids, topk_indices].set(topk_values)

        # Step 5: Renormalize the selected gates so they sum to 1 per token.
        # Because sigmoid scores are unbounded and independent, their sum can vary.
        # Renormalizing gives a stable output scale similar to softmax weighting.
        # Non-selected experts stay at 0, so only the top-k values are affected.
        routed_gates = routed_gates / (
            jnp.sum(routed_gates, axis=-1, keepdims=True) + 1e-6
        )

        # Compute the load-balancing loss from raw scores (before renormalization).
        # This is used only during training to encourage even expert utilization.
        aux_loss = self._load_balancing_aux_loss(routed_scores, topk_indices)

        # Step 6: Run all routed experts and compute the gated weighted sum.
        # routed_outputs: (num_tokens, num_routed_experts, d_model)
        routed_outputs = self._collect_routed_outputs(x_flat)

        # routed_gates[:, :, None] broadcasts to (num_tokens, num_routed_experts, d_model).
        # Summing over the expert dimension gives the combined routed output.
        # Non-selected experts (gate=0) contribute nothing.
        routed_mix = jnp.sum(routed_outputs * routed_gates[:, :, None], axis=1)
        # routed_mix: (num_tokens, d_model)

        # ── Shared path ─────────────────────────────────────────────────────────

        if self.num_shared_experts > 0:
            # Run all shared experts on all tokens — no routing, always active.
            shared_outputs = self._collect_shared_outputs(x_flat)
            # shared_outputs: (num_tokens, num_shared_experts, d_model)

            # Sum across shared experts: each expert adds its own contribution.
            # Summing (not averaging) means multiple shared experts act as an ensemble.
            shared_mix = jnp.sum(shared_outputs, axis=1)  # (num_tokens, d_model)

            # Final output = routed path + shared path.
            y_flat = routed_mix + shared_mix
        else:
            y_flat = routed_mix

        # Restore the original (batch, seqlen, d_model) shape.
        y = jnp.reshape(y_flat, (batch, seqlen, d_model))

        if return_aux_loss:
            return y, aux_loss
        return y
