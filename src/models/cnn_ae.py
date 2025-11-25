import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self, window_size, input_channels=2):
        super(CNNAutoencoder, self).__init__()
        # Input shape: (batch, input_channels, window_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # -> (32, window_size/2)
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, window_size/4)
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, window_size/8)
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        # Compute flatten size
        self._enc_out_dim = 128 * (window_size // 8)
        self.fc_latent = nn.Linear(self._enc_out_dim, 64)
        self.fc_expand = nn.Linear(64, self._enc_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), # -> (64, window_size/4)
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), # -> (32, window_size/2)
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, input_channels, kernel_size=4, stride=2, padding=1),  # -> (input_channels, window_size)
            nn.Tanh()
        )

    def forward(self, x):
        # x: (batch, input_channels, window_size)
        enc = self.encoder(x)
        batch = enc.size(0)
        flat = enc.view(batch, -1)
        z = self.fc_latent(flat)
        exp = self.fc_expand(z).view(batch, 128, -1)
        out = self.decoder(exp)
        return out, z
