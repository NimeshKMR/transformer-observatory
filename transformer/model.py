import torch
import torch.nn as nn

from .embeddings import TokenEmbedding, PositionalEmbedding
from .transformer_block import TransformerBlock
from .normalization import LayerNorm
from .lm_head import LMHead

class GPT(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(context_length, embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embedding_dim, num_heads)
                                    for _ in range(num_layers)
                                    ])
        self.layer_norm = LayerNorm(embedding_dim)
        self.lm_head = LMHead(embedding_dim, vocab_size)

    def forward(self, token_ids):
        token_embeddings = self.token_embedding(token_ids)
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        position_embeddings = self.positional_embedding(positions)
        X = token_embeddings + position_embeddings
        for block in self.blocks:
            X = block(X)
        X_norm = self.layer_norm(X)
        lm_output = self.lm_head(X_norm)

        return lm_output

