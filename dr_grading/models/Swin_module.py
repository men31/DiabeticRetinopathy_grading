import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .__utils import Identity
from .Classifier_module import ClassifierHead

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