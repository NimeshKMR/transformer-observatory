import torch 
import torch.nn as nn 

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.last_attention_weights = None

        self.last_query = None
        self.last_key = None
        self.last_value = None

    def forward(self, X):
        B, T, C = X.shape 

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        self.last_query = Q.detach()
        self.last_key = K.detach()
        self.last_value = V.detach()

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)
        mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=X.device),
        diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        self.last_attention_weights = weights.detach()
        output = weights @ V 

        output = output.transpose(1,2).contiguous().reshape(B, T, C)

        output = self.out_proj(output)

        return output 


