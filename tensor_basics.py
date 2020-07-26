"""
Script for tensor basics
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

t1 = torch.linspace(-2, 2, steps=5)
print(t1)
print("")
print(t1.shape)

# y = x^2 + 2x + 1

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(x.grad)