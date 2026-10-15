import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import torch

from transformer.model import GPT
from transformer.tokenizer import Tokeniser
from transformer.bpe import BPETokeniser

DATA_PATH = ROOT / "data" / "tiny-shakespeare.txt"
CHAR_CHECKPOINT = ROOT / "checkpoints" / "best_gpt.pt"
BPE_CHECKPOINT = ROOT / "checkpoints" / "best_gpt_bpe.pt"
BPE_TOKENIZER_PATH = ROOT / "checkpoints" / "bpe_tokeniser.json"

TRAIN_RATIO = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path):
    return torch.load(path, map_location=device, weights_only=True)


def load_model(checkpoint):
    model = GPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, token_ids, context_length, batch_size=32, num_batches=100):
    losses = []

    for _ in range(num_batches):
        starts = torch.randint(
            0,
            len(token_ids) - context_length - 1,
            (batch_size,),
        )

        x = torch.stack([
            token_ids[start:start + context_length]
            for start in starts.tolist()
        ]).to(device)

        y = torch.stack([
            token_ids[start + 1:start + context_length + 1]
            for start in starts.tolist()
        ]).to(device)

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        losses.append(loss.item())

    return sum(losses) / len(losses)


def main():
    text = DATA_PATH.read_text(encoding="utf-8")

    split = int(len(text) * TRAIN_RATIO)
    train_text = text[:split]
    val_text = text[split:]

    char_tokenizer = Tokeniser(text)
    bpe_tokenizer = BPETokeniser.load(str(BPE_TOKENIZER_PATH))

    char_ids = char_tokenizer.encode(val_text)
    bpe_ids = bpe_tokenizer.encode(val_text)

    char_checkpoint = load_checkpoint(CHAR_CHECKPOINT)
    bpe_checkpoint = load_checkpoint(BPE_CHECKPOINT)

    assert char_tokenizer.vocab_size == char_checkpoint["vocab_size"], (
        "Character tokenizer vocabulary does not match its checkpoint."
    )
    assert bpe_tokenizer.vocab_size == bpe_checkpoint["vocab_size"], (
        "BPE tokenizer vocabulary does not match its checkpoint."
    )

    char_model = load_model(char_checkpoint)
    bpe_model = load_model(bpe_checkpoint)

    char_train_count = len(char_tokenizer.encode(train_text))
    bpe_train_count = len(bpe_tokenizer.encode(train_text))

    print("\nTOKENIZER COMPARISON")
    print("-" * 45)
    print(f"Corpus characters:       {len(text):,}")
    print(f"Training characters:     {len(train_text):,}")
    print(f"Validation characters:   {len(val_text):,}")
    print()
    print(f"Character vocabulary:    {char_tokenizer.vocab_size}")
    print(f"BPE vocabulary:          {bpe_tokenizer.vocab_size}")
    print(f"Character train tokens:  {char_train_count:,}")
    print(f"BPE train tokens:        {bpe_train_count:,}")
    print(f"Character val tokens:    {len(char_ids):,}")
    print(f"BPE val tokens:          {len(bpe_ids):,}")
    print(
        f"BPE compression:         "
        f"{len(train_text) / bpe_train_count:.3f} characters/token"
    )
    print(
        f"Token-count reduction:   "
        f"{100 * (1 - bpe_train_count / char_train_count):.2f}%"
    )

    print("\nVALIDATION LOSS")
    print("-" * 45)

    char_loss = evaluate(
        char_model,
        torch.tensor(char_ids, dtype=torch.long),
        char_checkpoint["context_length"],
    )
    bpe_loss = evaluate(
        bpe_model,
        torch.tensor(bpe_ids, dtype=torch.long),
        bpe_checkpoint["context_length"],
    )

    print(f"Character model loss:    {char_loss:.4f} nats/token")
    print(f"BPE model loss:          {bpe_loss:.4f} nats/token")
    print(
        "\nImportant: these token-level losses are not directly comparable "
        "because the tokenization and vocabulary sizes differ."
    )


if __name__ == "__main__":
    main()
