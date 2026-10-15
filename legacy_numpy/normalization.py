import numpy as np 

class LayerNorm:

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim;
        self.gamma = np.ones(embedding_dim, )
        self.beta = np.zeros(embedding_dim, )
        self.epsilon = 1e-5

    def normalize(self, X):
        mu = np.mean(X, axis = 1, keepdims=True)
        std = np.std(X, axis = 1, keepdims = True)
        X = (X - mu)/(np.sqrt(std**2 + self.epsilon))
        return self.gamma * X + self.beta

X = np.random.randn(5, 4)

layer_norm = LayerNorm(embedding_dim=4)

normalized_X = layer_norm.normalize(X)

print("Input shape:", X.shape)
print("Output shape:", normalized_X.shape)

print("\nNormalized X:")
print(normalized_X)

print("\nRow means:")
print(np.mean(normalized_X, axis=1))

print("\nRow variances:")
print(np.var(normalized_X, axis=1))

print("\nGamma shape:", layer_norm.gamma.shape)
print("Beta shape:", layer_norm.beta.shape)