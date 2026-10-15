import torch 
import torch.nn as nn 
from .embeddings import TokenEmbedding, PositionalEmbedding


class LMHead(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, vocab_size, bias = False)

    def forward(self, X):
        return self.projection(X)




