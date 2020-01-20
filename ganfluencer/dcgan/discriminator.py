import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_features, n_channels, n_gpu):
        super(Discriminator, self).__init__()
        self.n_gpu = n_gpu

        self.block_0 = nn.Sequential(
            nn.Conv2d(n_channels, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(n_features * 4, n_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(n_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward_pass(self, z):
        x = self.block_0(z)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        return x
