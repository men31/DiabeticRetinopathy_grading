import torch
import torch.nn as nn
import torchvision.models as models

from .__utils import Identity
from .Classifier_module import ClassifierHead

class ResNet50(nn.Module):
    def __init__(
        self, num_classes: int, transfer: bool = True, activation: str = "relu"
    ):
        super(ResNet50, self).__init__()
        self.activation = activation
        self.weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=self.weights)
        # Transfer learning
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = Identity()
        self.head = ClassifierHead(2048, num_classes, [32], activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        output = self.head(output)
        return output
