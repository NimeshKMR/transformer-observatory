import numpy as np 
from legacy_numpy.normalization import LayerNorm
from legacy_numpy.selfattention import SelfAttention
from legacy_numpy.mlp import MLP

class TransformerBlock:

    def __init__(self, embedding_dim, head_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim 
        self.num_heads = num_heads
        self.ln1 = LayerNorm(self.embedding_dim)
        self.mha = SelfAttention(self.embedding_dim, self.head_dim, self.num_heads)
        self.ln2 = LayerNorm(self.embedding_dim)
        self.mlp = MLP(self.embedding_dim)

    def transform(self, X):
        X_norm = self.ln1.normalize(X)
        Q, K, V = self.mha.project(X_norm)
        attention_scores = self.mha.attention_scores(Q, K)
        scaled_scores = []
        masked_scores = []
        attention_weights = []
        for i in range(self.mha.num_heads):
            scaled = self.mha.scale_scores(attention_scores[i])
            masked = self.mha.causal_mask(scaled)
            weights = self.mha.softmax(masked)
            scaled_scores.append(scaled)
            masked_scores.append(masked)
            attention_weights.append(weights)
        outputs = self.mha.output(attention_weights, V)
        concatenated = self.mha.concatenate(outputs)
        mha_output = self.mha.output_projection(concatenated)
        residual_output = X + mha_output

        mlp_input = self.ln2.normalize(residual_output)
        z = self.mlp.forward(mlp_input)

        transformer_output = residual_output + z
        return transformer_output


X = np.random.randn(5,4)
block = TransformerBlock(
    embedding_dim=4,
    head_dim=2,
    num_heads=2
)

output = block.transform(X)

print("Input shape:", X.shape)
print("Output shape:", output.shape)
print("\nInput:")
print(X)

print("\nOutput:")
print(output)

output1 = block.transform(X)
output2 = block.transform(X)

print(np.allclose(output1, output2))
