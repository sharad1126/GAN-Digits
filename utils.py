import torch
import math
from torch.autograd.variable import Variable

def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n

def images_to_vectors(images):
    """
    Flatten out the image into a Tensor
    """
    return images.view(images.size(0), -1)

def vectors_to_images(vector, shape):
    """
    Assumes gray scale image of aspect ratio 1:1
    In this case, its MNIST, so will be batch_size, 1, 28, 28
    """
    assert len(shape) == 2
    return vector.view(vector.size(0), 1, shape[0], shape[1])

"""
Data templates
"""
def real_data_target(size, use_gpu=True):
    data = Variable(torch.ones(size, 1))
    if use_gpu: return data.cuda()
    return data

def fake_data_target(size, use_gpu=True):
    data = Variable(torch.zeros(size, 1))
    if use_gpu: return data.cuda()
    return data
