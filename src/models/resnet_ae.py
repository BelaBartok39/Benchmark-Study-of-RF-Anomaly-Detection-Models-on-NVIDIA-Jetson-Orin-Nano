import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetAutoencoder(nn.Module):
    def __init__(self, window_size, input_channels=2, channels=32, num_blocks=3):
        super(ResNetAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            *[ResBlock1D(channels) for _ in range(num_blocks)]
        )
        enc_len = window_size // 2
        self.decoder = nn.Sequential(
            *[ResBlock1D(channels) for _ in range(num_blocks)],
            nn.ConvTranspose1d(channels, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        # global latent vector by average pooling
        lat = z.mean(dim=2)
        return out, lat
