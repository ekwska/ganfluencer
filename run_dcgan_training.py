"""
Training loop for classic DCGAN modified from https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py

"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import pickle
from ganfluencer.dcgan.generator import Generator
from ganfluencer.dcgan.discriminator import Discriminator
from ganfluencer.bigdcgan.generator import BigGenerator
from ganfluencer.bigdcgan.discriminator import BigDiscriminator
from ganfluencer.utils import initialise_weights, initialise_loader
from torch.utils.tensorboard import SummaryWriter


def run_training_loop(config):
    # View config
    print("CONFIG: ", json.dumps(config, indent=2, sort_keys=True))
    # Set up logging
    writer = SummaryWriter(config['log_dir'])

    # Set up device parameters
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config['n_gpu'] > 0) else "cpu")

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

    # Print the model
    print(netG)

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

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(config['img_size'], config['z_dim'], 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=config['lr_discrim'], betas=(config['beta_1'], config['beta_2']))
    optimizerG = optim.Adam(netG.parameters(), lr=config['lr_gen'], betas=(config['beta_1'], config['beta_2']))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(config['n_epochs']):
        # For each batch in the data_loader
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
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, config['n_epochs'], i, len(data_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 50 == 0) or ((epoch == config['n_epochs'] - 1) and (i == len(data_loader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    return img_list


if __name__ == "__main__":
    config_fname = 'config/bigdcgan_config.json'
    with open(config_fname) as file:
        config = json.load(file)
    img_list = run_training_loop(config)
    with open('data/img_list.pkl', 'wb') as f:
        pickle.dump(img_list, f)
