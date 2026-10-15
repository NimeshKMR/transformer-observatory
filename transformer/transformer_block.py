import torch 
import torch.nn as nn 
from .attention import SelfAttention
from .normalization import LayerNorm
from .mlp import MLP

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ln1 = LayerNorm(self.embedding_dim)
        self.mha = SelfAttention(embedding_dim, num_heads)
        self.ln2 = LayerNorm(self.embedding_dim)
        self.mlp = MLP(self.embedding_dim)

    def forward(self, X):
        X_norm = self.ln1(X)
        attention_output = self.mha(X_norm)
        residual_output = X + attention_output
        mlp_input = self.ln2(residual_output)
        mlp_output = self.mlp(mlp_input)
        transformer_output = residual_output + mlp_output
        return transformer_output


