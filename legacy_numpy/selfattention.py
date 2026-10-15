import numpy as np

from legacy_numpy.embeddings import PositionalEmbedding, TokenEmbedding 

class SelfAttention:

    def __init__(self, embedding_dim, head_dim, num_heads):
        self.W_Q = []
        self.W_K = []
        self.W_V = []
        self.num_heads = num_heads 
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        for i in range(num_heads):
            self.W_Q.append(np.random.randn(embedding_dim, head_dim)) 
            self.W_K.append(np.random.randn(embedding_dim, head_dim))
            self.W_V.append(np.random.randn(embedding_dim, head_dim))
        self.W_O = np.random.randn(head_dim * num_heads, embedding_dim)

    def project(self, X):
        Q = []
        K = []
        V = []
        for i in range(self.num_heads):
            Q.append(X @ self.W_Q[i])
            K.append(X @ self.W_K[i])
            V.append(X @ self.W_V[i])
        return Q, K, V

    def attention_scores(self, Q, K):
        attention_scores = []
        for i in range(self.num_heads):
            attention_scores.append(Q[i] @ K[i].T)
        return attention_scores

    def scale_scores(self, scores):
        return scores/np.sqrt(self.W_Q[0].shape[1])

    def causal_mask(self, scale_scores):
        causal_mask = np.tril(scale_scores)
        rows, cols = np.triu_indices(n = len(causal_mask[0]), k=1)
        causal_mask[rows,cols] = -np.inf 
        return causal_mask

    def softmax(self, causal_mask):
        exp_matrix = np.exp(causal_mask - np.max(causal_mask, axis = 1, keepdims=True))
        softmax_matrix = exp_matrix/np.sum(exp_matrix, axis = 1, keepdims = True)
        return softmax_matrix

    def output(self, softmax, V):
        output = []
        for i in range(self.num_heads):
            output.append(softmax[i] @ V[i])
        return output

    def concatenate(self, outputs):
        conc = np.concatenate(outputs, axis=1)
        return conc

    def output_projection(self, conc):
        return conc @ self.W_O









attention = SelfAttention(
    embedding_dim=4,
    head_dim=2,
    num_heads=2
)


# 5 tokens, 4 features each
X = np.random.randn(5, 4)

print("Input X shape:", X.shape)

# 1. Q, K, V projections
Q, K, V = attention.project(X)

for i in range(attention.num_heads):
    print(f"\n--- Head {i+1} ---")
    print("Q shape:", Q[i].shape)
    print("K shape:", K[i].shape)
    print("V shape:", V[i].shape)


# 2. Attention scores
scores = attention.attention_scores(Q, K)

for i in range(attention.num_heads):
    print(f"\n--- Head {i+1} ---")

    print("Scores shape:", scores[i].shape)

    # 3. Scale
    scaled_scores = attention.scale_scores(scores[i])
    print("Scaled scores shape:", scaled_scores.shape)

    # 4. Causal mask
    masked_scores = attention.causal_mask(scaled_scores)
    print("Masked scores:")
    print(masked_scores)

    # 5. Softmax
    attention_weights = attention.softmax(masked_scores)
    print("Attention weights shape:", attention_weights.shape)
    print("Row sums:", np.sum(attention_weights, axis=1))

    # 6. Attention output
    output = attention_weights @ V[i]
    print("Head output shape:", output.shape)


# 7. Get all head outputs
outputs = []

for i in range(attention.num_heads):
    scaled_scores = attention.scale_scores(scores[i])
    masked_scores = attention.causal_mask(scaled_scores)
    attention_weights = attention.softmax(masked_scores)

    outputs.append(attention_weights @ V[i])


# 8. Concatenate heads
conc = attention.concatenate(outputs)

print("\nConcatenated output shape:", conc.shape)


# 9. Output projection
final_output = attention.output_projection(conc)

print("Final MHA output shape:", final_output.shape)
print("\nFinal MHA output:")
print(final_output)