import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_dim, n_features, n_channels, n_gpu):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu

        # Define architecture
        self.block_0 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, n_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.ReLU(True)
        )
        self.block_1 = nn.Sequential(
            nn.ConvTranspose2d(n_features * 8, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.ReLU(True)
        )
        self.block_2 = nn.Sequential(
            nn.ConvTranspose2d(n_features * 4, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.ReLU(True)
        )
        self.block_3 = nn.Sequential(
            nn.ConvTranspose2d(n_features * 2, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.ReLU(True)
        )
        self.block_4 = nn.Sequential(
            nn.ConvTranspose2d(n_features * 4, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward_pass(self, z):
        x = self.block_0(z)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        return x
