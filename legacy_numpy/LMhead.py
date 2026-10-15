import numpy as np 

class LMhead:

    def __init__(self, num_features, vocab_size):
        self.Wlm = np.random.randn(num_features, vocab_size)

    def forward(self, X):
        return X @ self.Wlm 