"""
Tensor Initialization
"""
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

my_tensor = torch.tensor([
    [7, 8, 9],
    [1, 2, 3]
], dtype=torch.float32, device=device, requires_grad=True)

print(f"Tensor Info {my_tensor}")
print("")
print(f"Tensor Device {my_tensor.device}")
print("")
print(f"Tensor shape {my_tensor.shape}")
print("")
print(f"Requires Grad {my_tensor.requires_grad}")
