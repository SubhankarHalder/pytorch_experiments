"""
Access Elements
"""
import torch

t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=torch.float32)


print(t.mean())
print("")
print(t.mean().item())
print("")
print(t.mean(dim=0).tolist())
