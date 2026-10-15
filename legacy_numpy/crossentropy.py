import numpy as np 

class CrossEntropy:
        
    def forward(self, logits, targets):
        exp_matrix = np.exp(logits - np.max(logits, axis = 1, keepdims=True))
        softmax_matrix = exp_matrix/np.sum(exp_matrix, axis = 1, keepdims = True)
        losses = np.zeros(len(targets))
        for i, target in enumerate(targets):
            losses[i] = -np.log(softmax_matrix[i, target])
            i += 1
        
        return np.mean(losses)


import numpy as np
from legacy_numpy.crossentropy import CrossEntropy

# 3 positions, vocabulary size = 4
logits = np.array([
    [2.0, 1.0, 0.0, -1.0],
    [3.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 3.0]
])

# Correct token at each position
targets = np.array([0, 1, 3])

loss_fn = CrossEntropy()

loss = loss_fn.forward(logits, targets)

print("Loss:", loss)
print("Loss type:", type(loss))