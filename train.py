#!/usr/bin/env python

import torch
from torchvision import transforms, datasets
from networks import Discriminator, Generator
from utils import noise

DATA_FOLDER = './torch_data/VGAN/MNIST'
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

"""
# Test if dataloader is working fine
for images, labels in data_loader:
    print(images.shape, labels.shape)
"""
discriminator = Discriminator()
generator = Generator()

if use_gpu:
    discriminator = discriminator.cuda()
    generator = generator.cuda()

"""
Instantiate optimizers
"""
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)

"""
Define loss
"""
loss = torch.nn.BCELoss()
NUM_EPOCHS = 200
