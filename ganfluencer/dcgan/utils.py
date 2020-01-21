import torch.nn as nn


def initialise_weights(model):
    """ Initialises convolutional-* layers and batch normalisation layers with a Gaussian (mu=0.0, sigma=0.02).
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)