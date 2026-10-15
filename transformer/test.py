import torch 
import torch.nn as nn 
from model import GPT 


model = GPT(
    vocab_size=20,
    context_length=8,
    embedding_dim=16,
    num_heads=4,
    num_layers=2
)

token_ids = torch.randint(0, 20, (3, 5))
logits = model(token_ids)

print("Input:", token_ids.shape)
print("Logits:", logits.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))

model.zero_grad()

token_ids = torch.randint(0, 20, (3, 5))
targets = torch.randint(0, 20, (3, 5))

logits = model(token_ids)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(
    logits.reshape(-1, 20),
    targets.reshape(-1)
)

loss.backward()

print("Loss:", loss.item())
print("Token embedding gradient exists:",
      model.token_embedding.embedding.weight.grad is not None)
print("Attention Q gradient exists:",
      model.blocks[0].mha.q_proj.weight.grad is not None)
print("MLP gradient exists:",
      model.blocks[0].mlp.fc1.weight.grad is not None)
print("LM head gradient exists:",
      model.lm_head.projection.weight.grad is not None)