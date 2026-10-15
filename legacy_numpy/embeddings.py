import numpy as np 

class TokenEmbedding:

    def __init__(self, num_features, vocab_size):
        self.embedding_matrix = np.random.randn(vocab_size, num_features)

    def get_embedding(self, ids):
        return self.embedding_matrix[ids] 

class PositionalEmbedding:
    def __init__(self, num_features, context_length):
        self.embedding_matrix = np.random.randn(context_length, num_features)
    
    def get_embedding(self, positions):
         return self.embedding_matrix[positions] 


token_embedding = TokenEmbedding(
    num_features=4,
    vocab_size=10
)

positional_embedding = PositionalEmbedding(
    num_features=4,
    context_length=5
)

token_ids = np.array([2, 7, 4, 1, 9])
positions = np.array([0, 1, 2, 3, 4])

token_vectors = token_embedding.get_embedding(token_ids)
position_vectors = positional_embedding.get_embedding(positions)

X = token_vectors + position_vectors

print("Token vectors shape:", token_vectors.shape)
print("Position vectors shape:", position_vectors.shape)
print("Final X shape:", X.shape)
print(X)