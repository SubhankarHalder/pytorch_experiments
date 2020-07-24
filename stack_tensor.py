"""
Stack and flatten tensor
"""

import torch

t1 = torch.tensor([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
])

t2 = torch.tensor([
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2]
])

t3 = torch.tensor([
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3]
])

t4 = torch.stack((t1, t2, t3))

print("t1", t1)
print("t1 shape", t1.shape)
print("")
print("t2", t2)
print("t2 shape", t2.shape)
print("")
print("t3", t3)
print("t3 shape", t3.shape)
print("")
print("t4", t4)
print("t4 shape", t4.shape)
print("")

t5 = t4.unsqueeze(dim=1)
print("t5", t5)
print("t5 shape", t5.shape)
print("")

t6 = t5.flatten(start_dim=2)
print("t6", t6)
print("t6 shape", t6.shape)
print("")

