"""
Model inspection for Transformer Observatory.

``inspect_model`` runs a single forward pass over a prompt and returns the internal
quantities of that pass. Nothing in the model is modified:

* Attention weights and Q/K/V are read from the ``last_*`` attributes that
  ``SelfAttention.forward`` already stores (detached, so they never hold the autograd graph).
* Residual-stream activations are captured with temporary forward hooks that are
  removed again before the function returns. The hooks only record; they never return a
  value, so they cannot change what the model computes.
* Pre-softmax attention scores are *not* stored. They are cheap to recompute from the saved
  Q and K (see ``transformer.analysis.attention_stages``), which avoids keeping
  three extra [heads, T, T] tensors per layer.

Returned dictionary (batch dimension removed, everything on the CPU):

    prompt, token_ids, tokens, position
    logits              [T, vocab]
    probabilities       [vocab]          softmax of logits[position]
    top_predictions     list of {token_id, token, probability}
    attention_weights   list over layers of [heads, T, T]   (post-softmax)
    queries/keys/values list over layers of [heads, T, head_dim]
    hidden_states       dict label -> [T, d_model] (``positional_embedding`` and
                        ``token_embedding`` are also [T, d_model]); labels, in order:
                        token_embedding, positional_embedding, embeddings,
                        block{i}.attn_resid, block{i}.out (for each block), final_norm
"""

import threading

import torch

# The attention modules keep their last Q/K/V/weights on the module itself and the model may be
# shared between threads (Streamlit's cache_resource shares one model across sessions). Serialising
# inspections keeps each result tied to its own forward pass.
_INSPECT_LOCK = threading.Lock()


def token_string(tokenizer, token_id):
    """Text of a token id for either tokenizer (BPE: ``int2token``; char-level: ``int2char``)."""
    mapping = getattr(tokenizer, "int2token", None)
    if mapping is None:
        mapping = tokenizer.int2char
    return mapping[token_id]


def next_token_distribution(logits_row, tokenizer, k=10):
    """
    Softmax and top-k for one position.

    ``logits_row`` is the 1-D [vocab] logit vector at that position. Returns
    ``(probabilities, top_predictions)`` where ``top_predictions`` is a list of
    ``{"token_id", "token", "probability"}`` sorted by decreasing probability.
    """
    probabilities = torch.softmax(logits_row, dim=-1)
    top_probabilities, top_token_ids = torch.topk(
        probabilities,
        k=min(k, probabilities.numel())
    )
    top_predictions = [
        {
            "token_id": token_id,
            "token": token_string(tokenizer, token_id),
            "probability": probability,
        }
        for token_id, probability in zip(top_token_ids.tolist(), top_probabilities.tolist())
    ]
    return probabilities, top_predictions


class _HiddenStateRecorder:
    """
    Context manager that records the tensors that exist in ``GPT.forward``.

    Hooks are registered on entry and always removed on exit. Each hook stores a detached CPU
    copy of batch element 0 and returns ``None``, which PyTorch treats as "leave the value alone".
    """

    def __init__(self, model):
        self.model = model
        self.captured = {}
        self._handles = []

    def _save(self, label, tensor):
        if tensor.dim() == 3:  # [B, T, C] -> [T, C]
            tensor = tensor[0]
        self.captured[label] = tensor.detach().cpu().clone()

    def __enter__(self):
        model = self.model
        register = self._handles.append

        register(model.token_embedding.register_forward_hook(
            lambda module, args, output: self._save("token_embedding", output)))
        register(model.positional_embedding.register_forward_hook(
            lambda module, args, output: self._save("positional_embedding", output)))

        # Input of the first block is token_embedding + positional_embedding.
        if len(model.blocks) > 0:
            register(model.blocks[0].register_forward_pre_hook(
                lambda module, args: self._save("embeddings", args[0])))

        for index, block in enumerate(model.blocks):
            # Input of ln2 is X + attention_output (the residual stream after the attention sub-layer).
            register(block.ln2.register_forward_pre_hook(
                lambda module, args, i=index: self._save(f"block{i}.attn_resid", args[0])))
            register(block.register_forward_hook(
                lambda module, args, output, i=index: self._save(f"block{i}.out", output)))

        register(model.layer_norm.register_forward_hook(
            lambda module, args, output: self._save("final_norm", output)))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


@torch.no_grad()
def inspect_model(model, tokenizer, prompt, device, position = None):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        raise ValueError("Prompt must contain at least one token.")
    if len(token_ids) > model.context_length:
        raise ValueError(
            f"Prompt has {len(token_ids)} tokens, but the model "
            f"supports a maximum context length of {model.context_length}."
        )

    if position is None:
        position = len(token_ids) - 1

    if position < 0 or position >= len(token_ids):
        raise ValueError("Position is outside the prompt.")

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    num_tokens = len(token_ids)

    with _INSPECT_LOCK:
        with _HiddenStateRecorder(model) as recorder:
            logits = model(x)

        attention_weights = []
        queries = []
        keys = []
        values = []

        for layer, block in enumerate(model.blocks):
            weights = block.mha.last_attention_weights
            if weights is None or weights.shape[-1] != num_tokens:
                raise RuntimeError(
                    f"Layer {layer} does not hold attention state for this forward pass."
                )
            attention_weights.append(weights[0].cpu().clone())
            queries.append(block.mha.last_query[0].cpu().clone())
            keys.append(block.mha.last_key[0].cpu().clone())
            values.append(block.mha.last_value[0].cpu().clone())

    probabilities, top_predictions = next_token_distribution(
        logits[0, position, :],
        tokenizer,
        k=10
    )

    return {
            "prompt": prompt,
            "token_ids": token_ids,
            "tokens": [token_string(tokenizer, token_id) for token_id in token_ids],
            "position": position,
            "logits": logits[0].cpu().clone(),
            "probabilities": probabilities.cpu().clone(),
            "attention_weights": attention_weights,
            "queries": queries,
            "keys": keys,
            "values": values,
            "hidden_states": recorder.captured,
            "top_predictions": top_predictions
        }
