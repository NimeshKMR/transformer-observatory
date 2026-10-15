
from pathlib import Path
from transformer.bpe import BPETokeniser

text = Path("data/tiny-shakespeare.txt").read_text(encoding="utf-8")

split_index = int(0.9 * len(text))
train_text = text[:split_index]
val_text = text[split_index:]

tokeniser_path = Path("checkpoints/bpe_tokeniser.json")

tokeniser = BPETokeniser(train_text, num_merges=500)
tokeniser.save(tokeniser_path)

print("Tokenizer saved.")

train_tokens = tokeniser.encode(train_text)
val_tokens = tokeniser.encode(val_text)

print("Training characters:", len(train_text))
print("Validation characters:", len(val_text))
print("Training tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))
print("Vocabulary size:", tokeniser.vocab_size)

assert tokeniser.decode(train_tokens) == train_text
assert tokeniser.decode(val_tokens) == val_text

print("Round-trip tests passed.")
