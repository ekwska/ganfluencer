import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_dim, f_depth, n_channels, n_gpu):
        """ DCGAN Generator model

        :param z_dim: Depth of feature maps
        :param f_depth: Depth of feature maps
        :param n_channels: Number of image channels (3 for RGB)
        :param n_gpu: Number of GPUs available (0 for CPU).

        """
        super(Generator, self).__init__()
        self.n_gpu = n_gpu

        # Define architecture
        self.block_0 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, f_depth * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(f_depth * 8),
            nn.ReLU(True)
        )
        self.block_1 = nn.Sequential(
            nn.ConvTranspose2d(f_depth * 8, f_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f_depth * 4),
            nn.ReLU(True)
        )
        self.block_2 = nn.Sequential(
            nn.ConvTranspose2d(f_depth * 4, f_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f_depth * 2),
            nn.ReLU(True)
        )
        self.block_3 = nn.Sequential(
            nn.ConvTranspose2d(f_depth * 2, f_depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f_depth),
            nn.ReLU(True)
        )
        self.block_4 = nn.Sequential(
            nn.ConvTranspose2d(f_depth, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.block_0(z)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        return x
