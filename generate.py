
import torch

from transformer.model import GPT
from transformer.bpe import BPETokeniser


CHECKPOINT_PATH = "checkpoints/best_gpt_bpe.pt"
TOKENISER_PATH = "checkpoints/bpe_tokeniser.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

tokenizer = BPETokeniser.load(TOKENISER_PATH)

assert tokenizer.vocab_size == checkpoint["vocab_size"], (
    "Tokenizer vocabulary size does not match the model checkpoint."
)

model = GPT(
    vocab_size=checkpoint["vocab_size"],
    context_length=checkpoint["context_length"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def encode(text):
    return tokenizer.encode(text)


def decode(token_ids):
    return tokenizer.decode(token_ids)


@torch.no_grad()
def generate(prompt, max_new_tokens=500, temperature=0.8):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    token_ids = encode(prompt)

    generated = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):
        context = generated[:, -model.context_length:]
        logits = model(context)

        next_logits = logits[:, -1, :] / temperature
        probabilities = torch.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        generated = torch.cat((generated, next_token), dim=1)

    return decode(generated[0].tolist())


print("Device:", device)
print("Vocabulary size:", tokenizer.vocab_size)
print("Checkpoint step:", checkpoint["step"])

print(generate(
    prompt="First Citizen:\n",
    max_new_tokens=500,
    temperature=0.6
))
