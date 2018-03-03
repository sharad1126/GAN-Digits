#!/usr/bin/env python

import torch
from torchvision import transforms, datasets
from torch.autograd.variable import Variable
from networks import Discriminator, Generator
from utils import noise, images_to_vectors, vectors_to_images
from utils import real_data_target, fake_data_target
import numpy as np
import visdom

DATA_FOLDER = './torch_data/VGAN/MNIST'
use_gpu = torch.cuda.is_available()

BATCH_SIZE = 100

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
data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
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


"""
function to train discriminator
"""
def train_discriminator(optimizer, real_data, fake_data):
    # reset gradients
    optimizer.zero_grad()

    # Train on real data
    prediction_real = discriminator(real_data)
    # Calculate error and back propagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0), use_gpu=use_gpu))
    error_real.backward()

    # Train on fake data
    prediction_fake = discriminator(fake_data)
    # Calculate error and back propagate
    error_fake = loss(prediction_fake, fake_data_target(fake_data.size(0), use_gpu=use_gpu))
    error_fake.backward()

    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake


"""
Setup plotter helpers
"""
vis = visdom.Visdom()
loss_win = "loss_win"
images_win = "images_win"
loss_plot_initiated = False
"""
function to train generator
"""
def train_generator(optimizer, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # Predict
    prediction = discriminator(fake_data)
    # Compute loss
    error = loss(prediction, real_data_target(prediction.size(0), use_gpu=use_gpu))
    # Compute gradients
    error.backward()

    optimizer.step()
    return error

"""
Actual training
"""

for epoch in range(NUM_EPOCHS):
    for n_batch, (real_batch, _) in enumerate(data_loader):

        real_data = Variable(images_to_vectors(real_batch))
        if use_gpu: real_data = real_data.cuda()

        # Note: the .detach() call detaches the generator from the
        # compute graph. Important for proper computation of
        # gradients in a case like this when training multiple
        # networks as the same time.
        # Reference : https://discuss.pytorch.org/t/how-does-detach-work/2308
        #
        # This was a **fucking** headache to figure out !!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        fake_data = generator(noise(real_data.size(0))).detach()

        # train Discriminator
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # Train Generator
        fake_data = generator(noise(real_batch.size(0)))
        g_error = train_generator(g_optimizer, fake_data)

        if n_batch % 100 == 0:
            time = epoch + n_batch*1.0/len(data_loader)
            num_samples = 64
            generated_sample = vectors_to_images(fake_data[:num_samples], (28, 28)).data.cpu()
            print(generated_sample.shape)
            print(d_error.data[0], g_error.data[0], epoch, n_batch, num_batches)
            current_time = epoch+(float(n_batch)/num_batches)
            if not loss_plot_initiated:
                vis.line(
                        Y=np.array([d_error.data[0], g_error.data[0]]).reshape(1, 2),
                         X=np.array([current_time]),
                        win=loss_win,
                        )
                loss_plot_initiated = True
            else:
                vis.line(
                        Y=np.array([d_error.data[0], g_error.data[0]]).reshape(1, 2),
                         X=np.array([current_time]),
                        win=loss_win,
                        update="append"
                        )

            # Plot images
            vis.images(generated_sample, win=images_win)
