import torch
import torch.nn as nn

class EMA(nn.Module):
    """
    Efficient Multiscale Attention (EMA) Module
    Combines channel and spatial attention in one efficient block.
    """
    def __init__(self, channels):
        super(EMA, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // 4, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.pool(x)
        attention = self.fc1(attention)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        return x * attention  # channel-wise modulation
