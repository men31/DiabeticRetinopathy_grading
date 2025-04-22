from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .__utils import Identity
from .Classifier_module import ClassifierHead
from .Encoder_module import ConvEncoder
from .Unet_module import UNet


class Swin_S(nn.Module):
    def __init__(
        self, num_classes: int, transfer: bool = True, activation: str = "relu"
    ):
        super(Swin_S, self).__init__()
        self.activation = activation
        self.weight = models.Swin_S_Weights.IMAGENET1K_V1
        self.model = models.swin_s(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.head = Identity()
        # self.head = ClassifierHead(768, num_classes, [64], activation=self.activation)
        self.head = ClassifierHead(
            768, num_classes, [256, 128, 128, 64], activation=self.activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        output = self.head(output)
        return output


class Swin_V2_S(nn.Module):
    def __init__(
        self,
        num_classes: int,
        transfer: bool = True,
        activation: str = "relu",
        in_channels: int = 3,
        encoder_model: Literal["simple", "unet"] = "unet",
        return_latent: bool = False,
    ):
        super(Swin_V2_S, self).__init__()
        self.transfer = transfer
        self.num_classes = num_classes
        self.activation = activation
        self.in_channels = in_channels
        self.encoder_model = encoder_model
        self.return_latent = return_latent
        self.hidden_layers = [256, 128, 128, 64]
        self.weight = models.Swin_V2_S_Weights.IMAGENET1K_V1
        self.model = self.__create_architecture()

    def __create_architecture(self):
        model_lst = []
        if self.in_channels != 3:
            if self.encoder_model == "simple":
                enc_rgb = ConvEncoder(in_channels=self.in_channels, out_channels=3)
            elif self.encoder_model == "unet":
                enc_rgb = UNet(
                    in_channels=self.in_channels,
                    out_channels=3,
                    features=(64, 128),
                    dropout=0.3,
                )
            else:
                raise ValueError(f"[-] Not found encoder model: {self.encoder_model}")
            model_lst.append(enc_rgb)

        model = models.swin_v2_s(weights=self.weight)
        if self.transfer:
            for param in model.parameters():
                param.requires_grad = False
        model.head = Identity()
        model_lst.append(model)

        head = ClassifierHead(
            768, self.num_classes, self.hidden_layers, activation=self.activation
        )
        model_lst.append(head)

        return nn.Sequential(*model_lst)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     output = self.model(x)
    #     return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.model[0](x) # Extract latent representation from Swin
        if self.return_latent:
            return latent  # Return latent representation
        output = self.model[1](latent)  # Pass through the classifier head
        return output


class Swin_V2_B(nn.Module):
    def __init__(
        self,
        num_classes: int,
        transfer: bool = True,
        activation: str = "relu",
        in_channels: int = 3,
        encoder_model: Literal["simple", "unet"] = "unet",
        return_latent: bool = False,
    ):
        super(Swin_V2_B, self).__init__()
        self.transfer = transfer
        self.num_classes = num_classes
        self.activation = activation
        self.in_channels = in_channels
        self.encoder_model = encoder_model
        self.return_latent = return_latent
        self.hidden_layers = [256, 128, 128, 64]
        self.weight = models.Swin_V2_B_Weights.IMAGENET1K_V1
        self.model = self.__create_architecture()

    def __create_architecture(self):
        model_lst = []
        if self.in_channels != 3:
            if self.encoder_model == "simple":
                enc_rgb = ConvEncoder(in_channels=self.in_channels, out_channels=3)
            elif self.encoder_model == "unet":
                enc_rgb = UNet(
                    in_channels=self.in_channels,
                    out_channels=3,
                    features=(64, 128),
                    dropout=0.3,
                )
            else:
                raise ValueError(f"[-] Not found encoder model: {self.encoder_model}")
            model_lst.append(enc_rgb)

        model = models.swin_v2_b(weights=self.weight)
        if self.transfer:
            for param in model.parameters():
                param.requires_grad = False
        model.head = Identity()
        model_lst.append(model)

        head = ClassifierHead(
            1024, self.num_classes, self.hidden_layers, activation=self.activation
        )
        model_lst.append(head)

        return nn.Sequential(*model_lst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.model[0](x) # Extract latent representation from Swin
        if self.return_latent:
            return latent  # Return latent representation
        output = self.model[1](latent)  # Pass through the classifier head
        return output
