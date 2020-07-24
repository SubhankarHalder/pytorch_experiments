"""
Tensor Operations Script
"""
import torch

t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]
], dtype=torch.float32)

print(t.size())
print(t.shape)
print(len(t.shape))
print(t.numel())
print(t.reshape([1, 12]))
print(t.reshape([2, 6]))
print(t.reshape(2, 2, 3))
