"""
Broadcast
"""
import torch

t1 = torch.tensor([
    [1, 2],
    [3, 4]
], dtype=torch.float32)

t2 = torch.tensor([
    [9, 8],
    [7, 6]
], dtype=torch.float32)

t3 = t1 / 2

print(t3)

t4 = torch.rand(1, 2)
t5 = torch.rand(3, 3, 1)

print((t4+t5).shape)

