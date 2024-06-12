import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets. For this tutorial,
#  we will be using a TorchVision dataset.

# Downloading training dataset from open dataset. 
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform= ToTensor(), 
)

batch_size = 64

# Create data loaders.
training_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]:",X.shape)
    print("Shape of y:",y.shape, y.dtype)
    break