import numpy as np 


class MLP:

    def __init__(self, embedding_dim):
        self.W1 = np.random.randn(embedding_dim, 4*embedding_dim)
        self.b1 = np.zeros(4*embedding_dim, )
        self.W2 = np.random.randn(4*embedding_dim, embedding_dim)
        self.b2 = np.zeros(embedding_dim, )

    def gelu(self, Y):
        G = Y/2 * (1 + np.tanh(
            np.sqrt(2/np.pi)*(Y + 0.044715*Y**3)))
        return G 

    def forward(self, X):
        Y = X @ self.W1 + self.b1
        G = self.gelu(Y)
        Z = G @ self.W2 + self.b2
        return Z 

np.random.seed(42)

X = np.random.randn(5, 4)

mlp = MLP(embedding_dim=4)

# Forward pass
Y = X @ mlp.W1 + mlp.b1
G = mlp.gelu(Y)
Z = mlp.forward(X)

print("Input X shape:", X.shape)
print("Y shape:", Y.shape)
print("GELU output G shape:", G.shape)
print("Final MLP output Z shape:", Z.shape)

print("\nY:")
print(Y)

print("\nGELU(Y):")
print(G)

print("\nFinal MLP output:")
print(Z)