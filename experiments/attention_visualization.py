import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformer.bpe import BPETokeniser
from transformer.model import GPT
from transformer.observatory import inspect_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = ROOT / "checkpoints" / "best_gpt_bpe.pt"
TOKENIZER_PATH = ROOT / "checkpoints" / "bpe_tokeniser.json"
OUTPUT_DIR = ROOT / "experiments" / "results" / "experiment_03"

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


def plot_attention(attention, tokens, prompt_id, prompt):
    """
    attention: (heads, sequence_length, sequence_length)
    """
    num_heads, seq_len, _ = attention.shape

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 13),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for head in range(num_heads):
        ax = axes[head]

        matrix = attention[head].detach().cpu().numpy()

        sns.heatmap(
            matrix,
            ax=ax,
            cmap="viridis",
            xticklabels=tokens,
            yticklabels=tokens,
            vmin=0,
            vmax=1,
            square=True,
            cbar=True,
        )

        ax.set_title(f"Head {head}")
        ax.set_xlabel("Key token (attended to)")
        ax.set_ylabel("Query token (doing the attending)")
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.tick_params(axis="y", labelrotation=0, labelsize=8)

    fig.suptitle(
        f"Prompt {prompt_id} — Layer attention\n{prompt}",
        fontsize=13,
    )

    return fig


def main():
    model, tokenizer = load_model()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("ATTENTION VISUALIZATION EXPERIMENT")
    print(f"Device: {DEVICE}")

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
                f"got {len(attention_layers)}"
            )

        for layer_idx, attention in enumerate(attention_layers):
            if attention.shape[0] != NUM_HEADS:
                raise ValueError(
                    f"Layer {layer_idx}: expected {NUM_HEADS} heads, "
                    f"got {attention.shape[0]}"
                )

            fig = plot_attention(
                attention=attention,
                tokens=tokens,
                prompt_id=prompt_id,
                prompt=prompt,
            )

            output_path = OUTPUT_DIR / (
                f"prompt_{prompt_id}_layer_{layer_idx}.png"
            )

            fig.savefig(output_path, dpi=160)
            plt.close(fig)

            print(f"Saved: {output_path}")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
