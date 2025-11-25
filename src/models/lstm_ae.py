import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, 2, seq_len) - I/Q channels
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        enc_out, (h, c) = self.encoder(x)
        dec_out, _ = self.decoder(enc_out)
        out = dec_out.permute(0, 2, 1)
        # latent vector as last hidden state
        z = h[-1]
        return out, z
