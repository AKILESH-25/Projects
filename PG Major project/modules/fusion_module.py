import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    """
    Feature Fusion module to merge low- and high-level features
    """
    def __init__(self, in_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f1, f2):
        # Resize f2 to f1 if needed
        if f1.size() != f2.size():
            f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([f1, f2], dim=1)
        return self.relu(self.bn(self.conv1(x)))
