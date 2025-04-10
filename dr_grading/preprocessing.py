import torch
from torch import nn
from torchvision.transforms import v2
import torchvision


class FourierTransform(v2.Transform):
    def __init__(self, shift=True, return_abs=True):
        super().__init__()
        self.shift = shift
        self.return_abs = return_abs

    def transform(self, inpt, params):
        # Assumes input is a 2D or 3D tensor (C, H, W) or (H, W)
        x = torch.fft.fft2(inpt)
        if self.shift:
            x = torch.fft.fftshift(x)
        return x.abs() if self.return_abs else x


class InverseFourierTransform(v2.Transform):
    def __init__(self, shift=True, return_real=True):
        super().__init__()
        self.shift = shift
        self.return_real = return_real

    def transform(self, inpt, params):
        # Assumes input is a 2D or 3D tensor (C, H, W) or (H, W)
        if self.shift:
            x = torch.fft.ifftshift(inpt)
        x = torch.fft.ifft2(x)
        return x.real if self.return_real else x
