"""
Experiment
"""
import torch
from torch.nn import Linear
from torch import nn

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)

print("w shape", w.shape)
print("b shape", b.shape)

def forward(x):
    yhat = w * x + b
    return yhat

x = torch.tensor([
    [1.0], 
    [2.0],
    [3.0]
    ])
yhat = forward(x)
print("The Prediction: ", yhat)
print("Y size", yhat.shape)

torch.manual_seed(1)

lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))

print("Python Dictionary", lr.state_dict())
print("keys:", lr.state_dict().keys())
print("values:", lr.state_dict().values())

print("weight:", lr.weight)
print("bias:", lr.bias)


x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)


x = torch.tensor([
    [1.0],
    [2.0]
])
yhat = lr(x)
print("The prediction: ", yhat)


x = torch.tensor([
    [1.0],
    [2.0],
    [3.0]
])

yhat = lr(x)
print("The prediction: ", yhat)

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

lrg = LR(1, 1)
print("The parameters: ", list(lrg.parameters()))
print("Linear Model: ", lrg.linear)


x = torch.tensor([
    [1.0]
])

yhat = lrg(x)

print("The Prediction: ", yhat)
