
from pathlib import Path

import torch
import torch.nn as nn

from transformer.model import GPT
from transformer.bpe import BPETokeniser
from transformer.batching import get_batch

torch.manual_seed(42)

DATA_PATH = Path("data/tiny-shakespeare.txt")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
CONTEXT_LENGTH = 128
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2

LEARNING_RATE = 3e-4
MAX_STEPS = 10000
EVAL_INTERVAL = 200
EVAL_BATCHES = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



text = DATA_PATH.read_text(encoding="utf-8")

split_index = int(0.9 * len(text))
train_text = text[:split_index]
val_text = text[split_index:]

tokenizer = BPETokeniser.load(
    CHECKPOINT_DIR / "bpe_tokeniser.json"
)

train_data = torch.tensor(
    tokenizer.encode(train_text),
    dtype=torch.long
)

val_data = torch.tensor(
    tokenizer.encode(val_text),
    dtype=torch.long
)

print("Characters:", len(text))
print("Vocabulary size:", tokenizer.vocab_size)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))

model = GPT(
    vocab_size=tokenizer.vocab_size,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(device)

print("Model vocabulary size:", model.vocab_size if hasattr(model, "vocab_size") else tokenizer.vocab_size)
print("Training data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)
print("Training data dtype:", train_data.dtype)

x, y = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH)

print("Input batch shape:", x.shape)
print("Target batch shape:", y.shape)
print("Input token range:", x.min().item(), x.max().item())
print("Target token range:", y.min().item(), y.max().item())

assert x.shape == (BATCH_SIZE, CONTEXT_LENGTH)
assert y.shape == (BATCH_SIZE, CONTEXT_LENGTH)
assert x.min() >= 0 and x.max() < tokenizer.vocab_size
assert y.min() >= 0 and y.max() < tokenizer.vocab_size

with torch.no_grad():
    logits = model(x.to(device))

print("Logits shape:", logits.shape)

assert logits.shape == (BATCH_SIZE, CONTEXT_LENGTH, tokenizer.vocab_size)

print("BPE sanity checks passed.")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_fn = nn.CrossEntropyLoss()

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)


@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data)]:
        losses = torch.zeros(EVAL_BATCHES)
        for i in range(EVAL_BATCHES):
            x, y = get_batch(
                split_data,
                BATCH_SIZE,
                CONTEXT_LENGTH
            )
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = loss_fn(
                logits.reshape(-1, tokenizer.vocab_size),
                y.reshape(-1)
            )

            losses[i] = loss.item()

        results[split_name] = losses.mean().item()

    model.train()
    return results


best_val_loss = float("inf")

for step in range(1, MAX_STEPS + 1):
    if step == 1 or step % EVAL_INTERVAL == 0:
        metrics = estimate_loss()

        train_loss = metrics["train"]
        val_loss = metrics["val"]

        print(
            f"Step {step:4d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": tokenizer.vocab_size,
                    "context_length": CONTEXT_LENGTH,
                    "embedding_dim": EMBEDDING_DIM,
                    "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                    "tokenizer_type": "bpe",
                    "tokenizer_path": str(CHECKPOINT_DIR / "bpe_tokeniser.json"),
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                },
                CHECKPOINT_DIR / "best_gpt_bpe.pt"
            )

            print("  Saved new best checkpoint.")

    x, y = get_batch(
        train_data,
        BATCH_SIZE,
        CONTEXT_LENGTH
    )

    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    loss = loss_fn(
        logits.reshape(-1, tokenizer.vocab_size),
        y.reshape(-1)
    )

    

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()


metrics = estimate_loss()

print("\nTraining complete.")
print(f"Final train loss: {metrics['train']:.4f}")
print(f"Final validation loss: {metrics['val']:.4f}")
print(f"Best validation loss: {best_val_loss:.4f}")
print("Best checkpoint:", CHECKPOINT_DIR / "best_gpt_pte.pt")
