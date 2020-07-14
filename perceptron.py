"""
This is a script for a perceptron using Pytorch
I teach the network to learn Y = 2X + 3
"""

import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    """
    This is a class for the perceptron
    """
    def __init__(self):
        """
        Constructor Method
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward Method
        """
        x = self.fc1(x)
        x = self.relu(x)
        return x


def criterion(out, label):
    """
    Least Squares
    """
    return (label - out)**2


if __name__ == "__main__":
    net = Net()
    print(list(net.parameters()))
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    data = [(1, 5), (2, 7), (3, 9), (4, 11), (5, 13), (6, 15)]
    for epoch in range(100):
        for i, samples in enumerate(data):
            X, Y = iter(samples)
            X = torch.tensor([X], dtype=torch.float32, requires_grad=True)
            Y = torch.tensor([Y], dtype=torch.float32, requires_grad=True)
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            print(f"We have loss: {loss.data[0]}")

    print("Parameters after training: ", list(net.parameters()))
    sample_input = torch.tensor([5], dtype=torch.float32)
    net.eval()
    print(f"Prediction for test input {sample_input[0]} is {net(sample_input)[0]}")



