import torch
from torch.utils.data import Dataset
torch.manual_seed(1)


class toy_set(Dataset):
    def __init__(self, length=100, transform=None):
        self.len = length
        self.x = 2*torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len


our_dataset = toy_set()
for x, y in our_dataset:
    print(f" X: {x} and Y: {y}")        
