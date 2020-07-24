"""
argmax script
"""
import torch

t = torch.tensor([
    [1, 0, 0, 2],
    [0, 3, 3, 0],
    [4, 0, 0, 5]
], dtype=torch.float32)

print("Max Dim 0:", t.max(dim=0))
print("")
print("Max Dim 1:", t.max(dim=1))
print("")
print("ArgMax Dim 0:", t.argmax(dim=0))
print("")
print("ArgMax Dim 1:", t.argmax(dim=1))
print("")

