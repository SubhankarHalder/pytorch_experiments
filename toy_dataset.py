import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, length=50, transform=None):
        self.len = length
        self.x = 2*torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            self.transform(sample)
        return sample

    def __len__(self):
        return self.len
