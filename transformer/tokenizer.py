
class Tokeniser:
    def __init__(self, text):
        self.text = text
        self.chars = sorted(set(text))
        self.char2int = {
            char: token_id
            for token_id, char in enumerate(self.chars)
        }

        self.int2char = {
            token_id: char
            for char, token_id in self.char2int.items()
        }

        self.vocab_size = len(self.char2int)

    def encode(self, text):
        try:
            return [self.char2int[char] for char in text]
        except KeyError as e:
            raise ValueError(
                f"Unknown character: {e.args[0]!r}"
            ) from e

    def decode(self, tokens):
        try:
            return "".join(self.int2char[token] for token in tokens)
        except KeyError as e:
            raise ValueError(
                f"Unknown token ID: {e.args[0]!r}"
            ) from e
