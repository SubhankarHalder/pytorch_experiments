"""
Build RGB Image
"""
import torch

r = torch.ones(1, 2, 2)

print(r)

print("")

g = r + 1 

print(g)
print("")

b = r + 2

print(b)
print("")

img = torch.cat((r,g,b), dim=0)
print(img)
print(img.shape)
print("")
