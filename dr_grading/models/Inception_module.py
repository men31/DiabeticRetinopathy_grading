import torch
import torch.nn as nn
import torchvision.models as models

from .__utils import Identity
from .Classifier_module import ClassifierHead

class Inception_V3(nn.Module):
    def __init__(
        self, num_classes: int, transfer: bool = True, activation: str = "relu"
    ):
        super(Inception_V3, self).__init__()
        self.activation = activation
        self.weight = models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = Identity()
        self.head = ClassifierHead(2048, num_classes, [64], activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if self.training:
            output = output[0]
        output = self.head(output)
        return output
