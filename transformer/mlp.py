import torch 
import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 4*embedding_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(4*embedding_dim, embedding_dim)

    def forward(self, X):
        return self.fc2(self.activation(self.fc1(X)))

