import torch.nn as nn


class BigDiscriminator(nn.Module):
    def __init__(self, f_depth, n_channels, n_gpu):
        """ DCGAN Discriminator model with 128 x 128 images (and an extra layer)

        :param f_depth: Depth of feature maps
        :param n_channels: Number of image channels (3 for RGB)
        :param n_gpu: Number of GPUs available (0 for CPU).

        """
        super(BigDiscriminator, self).__init__()
        self.n_gpu = n_gpu

        self.block_0 = nn.Sequential(
            nn.Conv2d(n_channels, f_depth, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(f_depth, f_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(f_depth * 2, f_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(f_depth * 4, f_depth * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(f_depth * 8, f_depth * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f_depth * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(f_depth * 16, 1, 1, 4, 0, bias=False), nn.Sigmoid()
        )

    def forward(self, z):
        x = self.block_0(z)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return x
