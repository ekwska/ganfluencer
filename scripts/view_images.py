import pickle

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import HTML

#
with open(
    "/media/bird/Elements/ganfluencer/bigdcgan_1000_epochs/results_bigdcgan_1000_epochs_D_loss.pkl",
    "rb",
) as f:
    results_D = pickle.load(f)

with open(
    "/media/bird/Elements/ganfluencer/bigdcgan_1000_epochs/results_bigdcgan_1000_epochs_G_loss.pkl",
    "rb",
) as ff:
    results_G = pickle.load(ff)

print(results_D, results_G)
iterations = list(range(len(results_G)))
sns.set(style="darkgrid")
sns.set_palette("Set2")
sns.set_context("poster")
fig = plt.figure()
ax = sns.lineplot(x=iterations, y=results_D, lw=1.25, label="Discriminator")
# plt.show()

# fig1 = plt.figure()
ax1 = sns.lineplot(x=iterations, y=results_G, lw=1.25, label="Generator")
plt.xlabel("Iterations")
plt.ylabel("BCE Loss")
plt.title(
    "Generator v.s discriminator loss (Big-DCGAN)"
)  # You can comment this line out if you don't need title
plt.show(fig)
plt.show()
plt.close()

#
# matplotlib.rcParams['animation.embed_limit'] = 2**128
#
# img_list = results[0]
# img_list_cp = img_list[1:]
# img_list_cp = img_list_cp[0::6]
# print(len(img_list))
# print(len(img_list_cp))
# # fig = plt.figure(figsize=(10, 10))
# # plt.axis("off")
# # ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list_cp]
# # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# #
# # HTML(ani.to_jshtml())
# # plt.show()
# i = img_list_cp[0]
# print(i)
# plt.imsave('egg.jpg', np.transpose(i.numpy(), (1, 2, 0)))
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from ganfluencer.bigdcgan.discriminator import BigDiscriminator
from ganfluencer.bigdcgan.generator import BigGenerator
from ganfluencer.dcgan.discriminator import Discriminator
from ganfluencer.dcgan.generator import Generator
from ganfluencer.utils import initialise_loader, initialise_weights

#
# config_fname = 'config/bigdcgan_config.json'
# with open(config_fname) as file:
#     config = json.load(file)
#
# # View config
# print("CONFIG: ", json.dumps(config, indent=2, sort_keys=True))
# # Set up data loader
# data_loader = initialise_loader(config['data_root'], config['img_size'], config['batch_size'], config['workers'])
# device = torch.device("cuda:0" if (torch.cuda.is_available() and config['n_gpu'] > 0) else "cpu")
#
# # Grab a batch of real images from the dataloader
# real_batch = next(iter(data_loader))
#
# # Plot the real images
# plt.figure(figsize=(15,15))
# plt.axis("off")
# plt.title("Real Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
# plt.show()
