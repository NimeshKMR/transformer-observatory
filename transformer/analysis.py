"""
Reusable analysis functions for Transformer Observatory (no UI, no model access).

Every function takes the dictionary returned by ``transformer.observatory.inspect_model`` (or
plain tensors) and returns tensors, floats or lists of dictionaries, so they can be unit tested,
used from the Streamlit app, and reused by the scripts in ``experiments/``.

Conventions
    T = number of prompt tokens, H = number of heads, D = head_dim, C = d_model.
    "Stage" names are the labels used in ``result["hidden_states"]``:
        token_embedding, positional_embedding, embeddings,
        block{i}.attn_resid, block{i}.out, final_norm
    All statistics are plain measurements of the tensors of one forward pass. Nothing here
    interprets what a head or hidden dimension "means".
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch


# --------------------------------------------------------------------------- #
# Attention: pre-softmax stages
# --------------------------------------------------------------------------- #

def causal_mask(num_tokens: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Boolean [T, T] mask that is True where the key lies in the future of the query."""
    return torch.triu(
        torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=device),
        diagonal=1,
    )


def attention_stages(query: torch.Tensor, key: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Recompute the attention calculation from saved Q and K.

    ``query`` and ``key`` have shape [..., T, D] (for example [H, T, D] as stored by
    ``inspect_model``). The operations are the same, in the same order, as
    ``SelfAttention.forward``:

        raw           = Q K^T
        scaled        = raw / sqrt(D)
        masked        = scaled with every future key set to -inf
        probabilities = softmax(masked, dim=-1)

    Returns a dict with those four tensors (each [..., T, T]) plus ``mask`` ([T, T], True where
    the key is in the future).
    """
    head_dim = query.shape[-1]
    num_tokens = query.shape[-2]

    raw = query @ key.transpose(-2, -1)
    scaled = raw / (head_dim ** 0.5)
    mask = causal_mask(num_tokens, device=query.device)
    masked = scaled.masked_fill(mask, float("-inf"))
    probabilities = torch.softmax(masked, dim=-1)

    return {
        "raw": raw,
        "scaled": scaled,
        "masked": masked,
        "probabilities": probabilities,
        "mask": mask,
    }


def head_attention_stages(result: Mapping, layer_idx: int, head_idx: int) -> Dict[str, torch.Tensor]:
    """``attention_stages`` for one layer and head of an inspection result ([T, T] tensors)."""
    return attention_stages(
        result["queries"][layer_idx][head_idx],
        result["keys"][layer_idx][head_idx],
    )


def query_row(stages: Mapping[str, torch.Tensor], position: int) -> Dict[str, torch.Tensor]:
    """
    The four stages for one query position, as 1-D [T] tensors, plus ``visible``
    (True for keys the query may attend to, i.e. key index <= position).
    """
    return {
        "raw": stages["raw"][position],
        "scaled": stages["scaled"][position],
        "masked": stages["masked"][position],
        "probabilities": stages["probabilities"][position],
        "visible": ~stages["mask"][position],
    }


# --------------------------------------------------------------------------- #
# Hidden states
# --------------------------------------------------------------------------- #

_EMBEDDING_PARTS = ("token_embedding", "positional_embedding")


def residual_stream(hidden_states: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    The residual stream in execution order: embeddings, then (attn_resid, out) for every block,
    then final_norm. ``token_embedding`` and ``positional_embedding`` are its two summands, so
    they are left out. (``final_norm`` is the LayerNorm output, not a residual-stream value.)
    """
    return {k: v for k, v in hidden_states.items() if k not in _EMBEDDING_PARTS}


def describe_stage(label: str) -> str:
    """Human-readable stage name."""
    if label == "token_embedding":
        return "Token embedding"
    if label == "positional_embedding":
        return "Positional embedding"
    if label == "embeddings":
        return "Embeddings (token + position)"
    if label == "final_norm":
        return "Final LayerNorm output"
    if label.startswith("block") and label.endswith(".attn_resid"):
        return f"L{label[len('block'):-len('.attn_resid')]} · after attention"
    if label.startswith("block") and label.endswith(".out"):
        return f"L{label[len('block'):-len('.out')]} · block output"
    return label


def short_stage_label(label: str) -> str:
    """Compact stage name for plot axes."""
    if label == "embeddings":
        return "embed"
    if label == "final_norm":
        return "final LN"
    if label.startswith("block") and label.endswith(".attn_resid"):
        return f"L{label[len('block'):-len('.attn_resid')]} attn"
    if label.startswith("block") and label.endswith(".out"):
        return f"L{label[len('block'):-len('.out')]} out"
    return label


def num_layers_in(hidden_states: Mapping[str, torch.Tensor]) -> int:
    """Number of Transformer blocks recorded in ``hidden_states``."""
    return sum(1 for label in hidden_states if label.startswith("block") and label.endswith(".out"))


def vector_stats(vector: torch.Tensor) -> Dict[str, float]:
    """L2 norm, mean, (population) standard deviation, min and max of a 1-D vector."""
    vector = vector.detach().float()
    return {
        "l2_norm": float(torch.linalg.vector_norm(vector)),
        "mean": float(vector.mean()),
        "std": float(vector.std(unbiased=False)),
        "min": float(vector.min()),
        "max": float(vector.max()),
    }


def hidden_state_trajectory(
    hidden_states: Mapping[str, torch.Tensor], position: int
) -> List[Dict[str, object]]:
    """
    One row per residual-stream stage for the token at ``position``:
    shape, vector statistics, cosine similarity to the previous stage, and relative change
    ``||h_k - h_{k-1}|| / ||h_{k-1}||``. The first stage has no predecessor (``None``).

    Caveat: cosine and relative change describe geometry only; they do not measure whether a
    representation became "better". ``final_norm`` is a LayerNorm output (rescaled, re-centred).
    """
    rows: List[Dict[str, object]] = []
    previous: Optional[torch.Tensor] = None

    for label, tensor in residual_stream(hidden_states).items():
        vector = tensor[position].float()
        row: Dict[str, object] = {
            "stage": label,
            "name": describe_stage(label),
            "shape": tuple(tensor.shape),
            **vector_stats(vector),
            "cosine_to_previous": None,
            "relative_change": None,
        }
        if previous is not None:
            row["cosine_to_previous"] = float(
                torch.nn.functional.cosine_similarity(vector, previous, dim=0, eps=1e-12)
            )
            previous_norm = float(torch.linalg.vector_norm(previous))
            row["relative_change"] = (
                float(torch.linalg.vector_norm(vector - previous)) / previous_norm
                if previous_norm > 0 else math.nan
            )
        rows.append(row)
        previous = vector

    return rows


def hidden_state_matrix(
    hidden_states: Mapping[str, torch.Tensor], position: int
) -> Tuple[List[str], torch.Tensor]:
    """(stage labels, tensor [S, C]) holding the token's vector at every residual-stream stage."""
    stream = residual_stream(hidden_states)
    labels = list(stream)
    return labels, torch.stack([stream[label][position].float() for label in labels])


def activation_summary(hidden_states: Mapping[str, torch.Tensor]) -> List[Dict[str, object]]:
    """
    Per-stage statistics over all tokens of the prompt: token-norm mean/min/max and the
    mean, std, min, max of all activations.
    """
    rows = []
    for label, tensor in hidden_states.items():
        values = tensor.float()
        norms = torch.linalg.vector_norm(values, dim=-1)
        rows.append({
            "stage": label,
            "name": describe_stage(label),
            "shape": tuple(tensor.shape),
            "token_norm_mean": float(norms.mean()),
            "token_norm_min": float(norms.min()),
            "token_norm_max": float(norms.max()),
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "max": float(values.max()),
        })
    return rows


def layer_norm_input_stats(hidden_states: Mapping[str, torch.Tensor]) -> List[Dict[str, object]]:
    """
    Statistics of what each LayerNorm receives, over the feature dimension of every token.

    The model's LayerNorms (ln1 and ln2 in every block, plus the final one) subtract the per-token
    mean and divide by the per-token standard deviation. These rows report that mean and standard
    deviation for the tensor entering each LayerNorm (averaged over tokens; min/max of the std).
    """
    layers = num_layers_in(hidden_states)
    inputs: List[Tuple[str, str]] = []
    for i in range(layers):
        inputs.append((f"block{i}.ln1", "embeddings" if i == 0 else f"block{i - 1}.out"))
        inputs.append((f"block{i}.ln2", f"block{i}.attn_resid"))
    if layers > 0:
        inputs.append(("final LayerNorm", f"block{layers - 1}.out"))

    rows = []
    for name, stage in inputs:
        values = hidden_states[stage].float()
        token_mean = values.mean(dim=-1)
        token_std = values.std(dim=-1, unbiased=False)
        rows.append({
            "layernorm": name,
            "input_stage": stage,
            "input_mean_avg": float(token_mean.mean()),
            "input_std_avg": float(token_std.mean()),
            "input_std_min": float(token_std.min()),
            "input_std_max": float(token_std.max()),
        })
    return rows


def block_update_norms(hidden_states: Mapping[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """
    For every block, the per-token L2 norm of what each sub-layer adds to the residual stream:

        attention update = block{i}.attn_resid - (block input)
        MLP update       = block{i}.out        - block{i}.attn_resid

    plus the norm of the block's input stream. All tensors are [T].
    """
    out = []
    for i in range(num_layers_in(hidden_states)):
        block_input = hidden_states["embeddings" if i == 0 else f"block{i - 1}.out"].float()
        mid = hidden_states[f"block{i}.attn_resid"].float()
        block_out = hidden_states[f"block{i}.out"].float()
        out.append({
            "input_norm": torch.linalg.vector_norm(block_input, dim=-1),
            "attention_update_norm": torch.linalg.vector_norm(mid - block_input, dim=-1),
            "mlp_update_norm": torch.linalg.vector_norm(block_out - mid, dim=-1),
        })
    return out


# --------------------------------------------------------------------------- #
# Numerical diagnostics
# --------------------------------------------------------------------------- #

def _nonfinite_row(name: str, tensor: torch.Tensor) -> Dict[str, object]:
    return {
        "tensor": name,
        "shape": tuple(tensor.shape),
        "nan": int(torch.isnan(tensor).sum()),
        "inf": int(torch.isinf(tensor).sum()),
    }


def finite_report(result: Mapping) -> List[Dict[str, object]]:
    """NaN / Inf counts for every tensor stored in an inspection result."""
    rows = [
        _nonfinite_row("logits", result["logits"]),
        _nonfinite_row("probabilities", result["probabilities"]),
    ]
    for label, tensor in result.get("hidden_states", {}).items():
        rows.append(_nonfinite_row(f"hidden_states[{label}]", tensor))
    for name in ("attention_weights", "queries", "keys", "values"):
        for layer, tensor in enumerate(result[name]):
            rows.append(_nonfinite_row(f"{name}[{layer}]", tensor))
    return rows


def all_finite(report: Sequence[Mapping[str, object]]) -> bool:
    """True when ``finite_report`` found no NaN and no Inf."""
    return all(row["nan"] == 0 and row["inf"] == 0 for row in report)


def attention_checks(result: Mapping, tolerance: float = 1e-5) -> List[Dict[str, object]]:
    """
    Structural checks of the stored attention weights, one row per (layer, head):

    * ``max_row_sum_error``       max |sum_j w[i, j] - 1| over query rows
    * ``max_future_weight``       largest weight on a future key (the causal mask makes this 0)
    * ``max_reconstruction_error`` max |softmax(mask(QK^T / sqrt(D))) - stored weights|, i.e. whether
                                   the pre-softmax stages recomputed from Q and K reproduce the forward pass

    ``ok`` compares all three with ``tolerance``. The default 1e-5 is a numerical tolerance for
    float32 chosen for this tool; it is not a published standard.
    """
    rows = []
    for layer, weights in enumerate(result["attention_weights"]):
        weights = weights.float()
        num_heads, num_tokens, _ = weights.shape
        mask = causal_mask(num_tokens, device=weights.device)
        stages = attention_stages(result["queries"][layer].float(), result["keys"][layer].float())

        row_sum_error = (weights.sum(dim=-1) - 1.0).abs().amax(dim=-1)
        future = weights[:, mask]
        future_weight = future.amax(dim=-1) if future.numel() else torch.zeros(num_heads)
        reconstruction_error = (stages["probabilities"] - weights).abs().amax(dim=(-2, -1))

        for head in range(num_heads):
            row = {
                "layer": layer,
                "head": head,
                "max_row_sum_error": float(row_sum_error[head]),
                "max_future_weight": float(future_weight[head]),
                "max_reconstruction_error": float(reconstruction_error[head]),
            }
            row["ok"] = bool(
                row["max_row_sum_error"] <= tolerance
                and row["max_future_weight"] <= tolerance
                and row["max_reconstruction_error"] <= tolerance
            )
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Attention distribution measurements
# --------------------------------------------------------------------------- #

def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats of every attention row: -sum_j p_j ln p_j (0 ln 0 = 0). [..., T, T] -> [..., T]."""
    return -torch.special.xlogy(weights, weights).sum(dim=-1)


def attention_by_distance(weights: torch.Tensor) -> torch.Tensor:
    """
    Average attention mass at each distance back from the query.

    ``weights`` is [H, T, T]. Entry [h, d] is the mean over queries i >= d of w[h, i, i - d];
    d = 0 is attention to the token itself, d = 1 the previous token, and so on. Returns [H, T].
    """
    num_tokens = weights.shape[-1]
    columns = [
        torch.diagonal(weights, offset=-d, dim1=-2, dim2=-1).mean(dim=-1)
        for d in range(num_tokens)
    ]
    return torch.stack(columns, dim=-1)


def attention_summary(result: Mapping) -> List[Dict[str, Optional[float]]]:
    """
    Descriptive attention measurements, one row per (layer, head), averaged over the prompt's
    query positions:

    * ``mean_entropy``            nats; the maximum possible for row i is ln(i + 1)
    * ``mean_normalized_entropy`` entropy / ln(i + 1) for rows i >= 1 (1.0 = uniform over visible keys);
                                  ``None`` for a one-token prompt
    * ``mean_max_weight``         mean over rows of the largest weight
    * ``self_mass`` / ``prev_token_mass`` / ``first_token_mass``  mean mass on the query itself, the previous
                                  token, and token 0 (the last two use rows i >= 1; ``None`` for one token)

    These are measurements of one prompt. They do not establish a function for any head.
    """
    rows = []
    for layer, weights in enumerate(result["attention_weights"]):
        weights = weights.float()
        num_heads, num_tokens, _ = weights.shape
        entropy = attention_entropy(weights)                       # [H, T]
        max_weight = weights.amax(dim=-1)                          # [H, T]
        self_mass = torch.diagonal(weights, dim1=-2, dim2=-1)      # [H, T]

        if num_tokens >= 2:
            max_entropy = torch.log(torch.arange(2, num_tokens + 1, dtype=weights.dtype))
            normalized = (entropy[:, 1:] / max_entropy).mean(dim=-1)
            previous = torch.diagonal(weights, offset=-1, dim1=-2, dim2=-1).mean(dim=-1)
            first = weights[:, 1:, 0].mean(dim=-1)
        else:
            normalized = previous = first = None

        for head in range(num_heads):
            rows.append({
                "layer": layer,
                "head": head,
                "mean_entropy": float(entropy[head].mean()),
                "mean_normalized_entropy": None if normalized is None else float(normalized[head]),
                "mean_max_weight": float(max_weight[head].mean()),
                "self_mass": float(self_mass[head].mean()),
                "prev_token_mass": None if previous is None else float(previous[head]),
                "first_token_mass": None if first is None else float(first[head]),
            })
    return rows
