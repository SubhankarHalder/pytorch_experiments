import torch
from torch.utils.data import Dataset
from torchvision import transforms
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


class add_mult():
    def __init__(self, addx=1, muly=2):
        self.addx = addx
        self.muly = muly
    
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample


class mult():
    def __init__(self, mult=100):
        self.mult = mult

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = self.mult*x
        y = self.mult*y
        sample = x, y
        return sample


data_transfrom = transforms.Compose([add_mult(), mult()])
data_set = toy_set()
print(data_transfrom(data_set[0]))

compose_data_set = toy_set(transform=data_transfrom)

for i in range(3):
    x, y = data_set[i]
    print(x, y)
    x_, y_ = compose_data_set[i]
    print(x_, y_)
