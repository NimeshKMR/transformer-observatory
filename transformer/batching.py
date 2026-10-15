import torch 

def get_batch(data, batch_size, context_length):
    starts = torch.randint(
        0,
        len(data) - context_length,
        (batch_size,))
    x = torch.stack([
        data[start : start + context_length]
        for start in starts])
    y = torch.stack([
        data[start + 1 : start + context_length + 1]
        for start in starts])
    return x,y
