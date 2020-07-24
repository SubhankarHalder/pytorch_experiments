"""
New Tensor File
"""
import torch
import numpy as np

data = np.array([1, 2, 3])
t_1 = torch.Tensor(data)
t_2 = torch.tensor(data)
t_3 = torch.as_tensor(data)
t_4 = torch.from_numpy(data)

print(t_1)
print(t_2)
print(t_3)
print(t_4)

t_5 = torch.eye(3)
t_6 = torch.ones(2, 2)
t_7 = torch.rand(2, 2)

print("")
print(t_5)
print(t_6)
print(t_7)