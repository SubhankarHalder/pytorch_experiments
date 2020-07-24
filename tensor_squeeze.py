"""
Squeeze operations for tensor
"""
import torch

t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]
], dtype=torch.float32)

print(t)
print("")
s = t.reshape([1, 12])
print(s)
print("")

u = s.squeeze()

print(u)
print(u.shape)
print("")
v = u.unsqueeze(dim=0)

print(v)
print(v.shape)
print("")