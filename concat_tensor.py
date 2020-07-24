"""
Concat Tensor
"""
import torch

t1 = torch.tensor([
    [1, 2],
    [3, 4]
])

t2 = torch.tensor([
    [5, 6],
    [7, 8]
])

t3 = torch.cat((t1, t2), dim=0)
t4 = torch.cat((t1, t2), dim=1)

print(t1)
print(t2)
print(t3)
print(t4)