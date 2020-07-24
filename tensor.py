"""
This is a script on tensors
"""
import torch

tensor_data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    ]

t = torch.tensor(tensor_data)
print(t)
print(t.shape)

s = t.reshape(1, 9)
print(s)
print(s.shape)
