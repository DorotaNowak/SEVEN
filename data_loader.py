import torchvision.datasets as datasets
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

def load_data():
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/files/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=60000, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('/files/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=10000, shuffle=True)

    return train_loader, test_loader