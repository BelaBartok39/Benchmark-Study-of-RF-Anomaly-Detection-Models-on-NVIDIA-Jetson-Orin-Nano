import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_layers=[128, 64]):
        super(FeedForwardNet, self).__init__()
        layers = []
        prev = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(True))
            prev = h
        layers.append(nn.Linear(prev, 1))  # output anomaly score
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
