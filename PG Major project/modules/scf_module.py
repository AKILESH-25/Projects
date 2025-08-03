import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=None, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, padding=k//2 if p is None else p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SCFBlock(nn.Module):
    """
    Symmetric C2f block with 1x1 -> 3x3 -> 1x1 conv layers
    """
    def __init__(self, in_channels, out_channels):
        super(SCFBlock, self).__init__()
        hidden = out_channels // 2
        self.conv1 = ConvBNAct(in_channels, hidden, k=1)
        self.conv2 = ConvBNAct(hidden, hidden, k=3)
        self.conv3 = ConvBNAct(hidden, out_channels, k=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x + x3  # residual connection
