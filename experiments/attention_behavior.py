import sys
import csv
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformer.bpe import BPETokeniser
from transformer.model import GPT
from transformer.observatory import inspect_model


# ---------- Configuration ----------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = ROOT / "checkpoints" / "best_gpt_bpe.pt"
TOKENIZER_PATH = ROOT / "checkpoints" / "bpe_tokeniser.json"
OUTPUT_PATH = ROOT / "experiments" / "results" / "experiment_02_attention.csv"

NUM_LAYERS = 2
NUM_HEADS = 4

PROMPTS = [
    "To be, or not to be, that is the question:",
    "ROMEO: What light through yonder window breaks?",
    "KING: The world is full of strange and wondrous things.",
]


def load_model():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    tokenizer = BPETokeniser.load(str(TOKENIZER_PATH))

    model = GPT(
        vocab_size=len(tokenizer.token2int),
        embedding_dim=64,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        context_length=128,
    ).to(DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer


def measure_attention(weights):
    """
    weights: (heads, query_positions, key_positions)

    Computes mean attention entropy, effective positions,
    and backward distance for every head.
    """
    results = []

    num_heads, seq_len, _ = weights.shape

    for head in range(num_heads):
        head_weights = weights[head].float()

        entropies = []
        effective_positions = []
        backward_distances = []

        for i in range(1, seq_len):
            # Causal attention permits keys 0 through i.
            probs = head_weights[i, :i + 1]
            probs = probs / probs.sum().clamp_min(1e-12)

            entropy = -(
                probs * probs.clamp_min(1e-12).log()
            ).sum()

            entropies.append(entropy.item())
            effective_positions.append(math.exp(entropy.item()))

            distances = torch.arange(
                i,
                -1,
                -1,
                device=probs.device,
                dtype=probs.dtype,
            )

            backward_distances.append(
                (probs * distances).sum().item()
            )

        results.append({
            "head": head,
            "mean_entropy": sum(entropies) / len(entropies),
            "mean_effective_positions": (
                sum(effective_positions) / len(effective_positions)
            ),
            "mean_backward_distance": (
                sum(backward_distances) / len(backward_distances)
            ),
        })

    return results


def main():
    model, tokenizer = load_model()

    all_rows = []

    print("ATTENTION BEHAVIOR EXPERIMENT")
    print("=" * 75)
    print(f"Device: {DEVICE}")
    print(f"Layers: {NUM_LAYERS}")
    print(f"Heads per layer: {NUM_HEADS}")

    for prompt_id, prompt in enumerate(PROMPTS, start=1):
        inspection = inspect_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=DEVICE,
        )

        tokens = inspection["tokens"]
        attention_layers = inspection["attention_weights"]

        print(f"\nPrompt {prompt_id}: {prompt!r}")
        print(f"Tokens: {len(tokens)}")

        if len(attention_layers) != NUM_LAYERS:
            raise ValueError(
                f"Expected {NUM_LAYERS} layers, "
                f"received {len(attention_layers)}."
            )

        for layer_idx, weights in enumerate(attention_layers):
            if weights.shape[0] != NUM_HEADS:
                raise ValueError(
                    f"Layer {layer_idx}: expected {NUM_HEADS} heads, "
                    f"received {weights.shape[0]}."
                )

            if weights.shape[1] != weights.shape[2]:
                raise ValueError(
                    f"Layer {layer_idx}: attention matrix is not square."
                )

            metrics = measure_attention(weights)

            for row in metrics:
                row.update({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "layer": layer_idx,
                    "sequence_length": weights.shape[1],
                })
                all_rows.append(row)

    # Validate that every prompt/layer/head combination exists once.
    expected_rows = len(PROMPTS) * NUM_LAYERS * NUM_HEADS

    if len(all_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, got {len(all_rows)}."
        )

    combinations = [
        (r["prompt_id"], r["layer"], r["head"])
        for r in all_rows
    ]

    if len(set(combinations)) != expected_rows:
        raise RuntimeError("Duplicate prompt/layer/head combinations found.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prompt_id",
        "prompt",
        "layer",
        "head",
        "sequence_length",
        "mean_entropy",
        "mean_effective_positions",
        "mean_backward_distance",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nRESULTS — averaged over query positions")
    print("-" * 75)
    print(
        f"{'Prompt':<8}{'Layer':<8}{'Head':<8}"
        f"{'Entropy':<12}{'Eff. pos.':<12}{'Back dist.':<12}"
    )

    for row in all_rows:
        print(
            f"{row['prompt_id']:<8}"
            f"{row['layer']:<8}"
            f"{row['head']:<8}"
            f"{row['mean_entropy']:<12.3f}"
            f"{row['mean_effective_positions']:<12.2f}"
            f"{row['mean_backward_distance']:<12.2f}"
        )

    print(f"\nValidated {len(all_rows)} rows.")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
