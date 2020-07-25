"""
Neural Net
"""
# Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create FC Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True,
                               transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train=False,
                              transform=transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=True)

# Initialize Network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # Optimizer
        optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0 
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        accuracy = (float(num_correct)/float(num_samples)) * 100
        print(f"Accuracy: {accuracy}")

    model.train()


print("Train Loader accuracy")
check_accuracy(train_loader, model)
print("Test Loader")
check_accuracy(test_loader, model)










