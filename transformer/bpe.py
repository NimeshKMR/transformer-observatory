import json
from pathlib import Path

def count_pairs(sequences):
    pair_counts = {}
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            if pair not in pair_counts:
                pair_counts[pair] = 0
            pair_counts[pair] += 1

    return pair_counts

def merge_pair(sequences, pair):
    merged_sequences = []
    for sequence in sequences:
        merged_sequence = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair:
                merged_sequence.append(sequence[i] + sequence[i + 1])
                i += 2
            else:
                merged_sequence.append(sequence[i])
                i += 1
        merged_sequences.append(merged_sequence)
    return merged_sequences


def train_bpe(sequences, num_merges):
    sequences = [sequence.copy() for sequence in sequences]
    merges = []
    for _ in range(num_merges):
        pair_counts = count_pairs(sequences)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        sequences = merge_pair(sequences, best_pair)
        merges.append(best_pair)

    return merges, sequences

class BPETokeniser:
    def __init__(self, text, num_merges):
        self.chars = sorted(set(text))
        self.char2int = {char: token_id for token_id, char in enumerate(self.chars)}
        self.int2char = {token_id: char for char, token_id in self.char2int.items()}

        sequences = [list(text)]
        self.merges, _ = train_bpe(sequences, num_merges)

        self.vocab = self.chars.copy()

        for pair in self.merges:
            new_token = pair[0] + pair[1]

            if new_token not in self.vocab:
                self.vocab.append(new_token)

        self.token2int = {token: token_id for token_id, token in enumerate(self.vocab)}
        self.int2token = {token_id: token for token, token_id in self.token2int.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        tokens = list(text)

        for pair in self.merges:
            tokens = merge_pair([tokens], pair)[0]

        return [self.token2int[token] for token in tokens]

    def decode(self, token_ids):
        return "".join(self.int2token[token_id] for token_id in token_ids)

    def save(self, path):
        data = {
            "chars": self.chars,
            "merges": [list(pair) for pair in self.merges]
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)


    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokeniser = cls.__new__(cls)
        tokeniser.chars = data["chars"]
        tokeniser.merges = [tuple(pair) for pair in data["merges"]]

        tokeniser.vocab = tokeniser.chars.copy()

        for pair in tokeniser.merges:
            new_token = pair[0] + pair[1]

            if new_token not in tokeniser.vocab:
                tokeniser.vocab.append(new_token)

        tokeniser.token2int = {
            token: token_id
            for token_id, token in enumerate(tokeniser.vocab)
        }
        tokeniser.int2token = {
            token_id: token
            for token, token_id in tokeniser.token2int.items()
        }
        tokeniser.vocab_size = len(tokeniser.vocab)

        return tokeniser

