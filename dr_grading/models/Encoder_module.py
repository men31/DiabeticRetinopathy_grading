import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels:int=3):
        super(ConvEncoder, self).__init__()
        self.hidden_channels = 2**(in_channels // out_channels + 5)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.hidden_channels, kernel_size=(1, 1), stride=1)
        self.conv2 = nn.Conv2d(in_channels=self.hidden_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1)
        self.norm1 = nn.BatchNorm2d(self.hidden_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out

