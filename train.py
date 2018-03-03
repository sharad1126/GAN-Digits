#!/usr/bin/env python

import torch
from torchvision import transforms, datasets

DATA_FOLDER = './torch_data/VGAN/MNIST'
torch.set_default_tensor_type('torch.cuda.FloatTensor')
use_gpu = torch.cuda.is_available()

def mnist_data():
    compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5 ), (.5, .5, .5))
        ]
    )
    out_dir = "{}/dataset".format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

data = mnist_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)

for images, labels in data_loader:
    print(images.shape, labels.shape)
