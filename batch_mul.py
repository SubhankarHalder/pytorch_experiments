"""
Batch Mul
"""
import torch

batch = 32
n = 10
m = 20
p = 30

t1 = torch.rand((batch, n, m))
t2 = torch.rand((batch, m, p))
out = torch.bmm(t1, t2)

print(t1.shape)
print(t2.shape)
print(out.shape)