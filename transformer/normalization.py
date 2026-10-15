import torch 
import torch.nn as nn 

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, X):
        return self.norm(X)
    