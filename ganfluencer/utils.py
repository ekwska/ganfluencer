import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


def initialise_weights(model):
    """ Initialises convolutional-* layers and batch normalisation layers with a Gaussian (mu=0.0, sigma=0.02).
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def initialise_loader(data_root, img_size, batch_size, workers):
    dataset = dset.ImageFolder(root=data_root,
                               transform=transforms.Compose([
                                   transforms.Resize((img_size, img_size)),
                                   transforms.RandomRotation((0, 360)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=workers)
    return data_loader
