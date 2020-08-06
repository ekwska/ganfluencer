"""
Training loop for classic DCGAN modified from https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py

"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import pickle
import os
from ganfluencer.dcgan.generator import Generator
from ganfluencer.dcgan.discriminator import Discriminator
from ganfluencer.bigdcgan.generator import BigGenerator
from ganfluencer.bigdcgan.discriminator import BigDiscriminator
from ganfluencer.utils import initialise_weights, initialise_loader
from torch.utils.tensorboard import SummaryWriter
import logging

logging.basicConfig(format='%(asctime)-15s:%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger()


def run_training_loop(config):
    # View config
    logger.debug(f"CONFIG: {json.dumps(config, indent=2, sort_keys=True)}")
    # Set up logging
    writer = SummaryWriter(config['log_dir'])

    # Set up device parameters
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config['n_gpu'] > 0) else "cpu")
    logger.debug(f'Device is {device}')

    # Set up data loader
    data_loader = initialise_loader(config['data_root'], config['img_size'], config['batch_size'], config['workers'])

    #  Define model to use
    model_name = config['model']

    if model_name not in ['dcgan', 'bigdcgan']:
        raise ValueError(f"{config['model']} is not a valid model. Use one of 'dcgan' or 'bigdcgan")

    # Create the generator
    if model_name == 'dcgan':
        netG = Generator(config['z_dim'], config['f_depth_gen'], config['n_channels'], config['n_gpu']).to(device)
    else:
        netG = BigGenerator(config['z_dim'], config['f_depth_gen'], config['n_channels'], config['n_gpu']).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (config['n_gpu'] > 1):
        netG = nn.DataParallel(netG, list(range(config['n_gpu'])))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(initialise_weights)
    logging.debug(netG)

    # Create the Discriminator
    if model_name == 'dcgan':
        netD = Discriminator(config['f_depth_discrim'], config['n_channels'], config['n_gpu']).to(device)
    else:
        netD = BigDiscriminator(config['f_depth_discrim'], config['n_channels'], config['n_gpu']).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (config['n_gpu'] > 1):
        netD = nn.DataParallel(netD, list(range(config['n_gpu'])))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(initialise_weights)
    logging.debug(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(config['img_size'], config['z_dim'], 1, 1, device=device)

    real_label = 0.9
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=config['lr_discrim'], betas=(config['beta_1'], config['beta_2']))
    optimizerG = optim.Adam(netG.parameters(), lr=config['lr_gen'], betas=(config['beta_1'], config['beta_2']))

    image_dir = os.path.join(config['log_dir'], 'images')
    os.makedirs(image_dir, exist_ok=True)

    # Training Loop
    logging.debug("Starting Training Loop...")
    for epoch in range(config['n_epochs']):
        errG_epoch = 0.0
        errD_epoch = 0.0
        for i, data in enumerate(data_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)

            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config['z_dim'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 100 == 0:
                logging.debug('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, config['n_epochs'], i, len(data_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            errG_epoch += errG.item()
            errD_epoch += errD.item()

        errD_epoch /= len(data_loader)
        errG_epoch /= len(data_loader)

        logging.debug("Epoch %d Epoch_Mean_Loss_G %d Epoch_Mean_Loss_D %d", epoch+1, errG_epoch, errD_epoch)
        writer.add_scalar("Loss_Discriminator", errD_epoch, epoch+1)
        writer.add_scalar("Loss_Generator", errG_epoch, epoch+1)

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()

        for img_n in range(10):
            vutils.save_image(fake[img_n, :, :, :], os.path.join(image_dir, f'{epoch+1}_{img_n}.png'),
                              normalize=True)
        vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),
                          os.path.join(image_dir, f'{epoch+1}_grid.png'))


    writer.close()
    return True


if __name__ == "__main__":
    config_fname = 'config/lemon_config.json'
    with open(config_fname) as file:
        config = json.load(file)
    run_training_loop(config)
