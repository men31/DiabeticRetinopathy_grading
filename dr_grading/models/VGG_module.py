import torch
import torch.nn as nn
import torchvision.models as models

from .__utils import Identity
from .Classifier_module import ClassifierHead

class VGG19(nn.Module):
    def __init__(
        self, num_classes: int, transfer: bool = True, activation: str = "relu"
    ):
        super().__init__()
        self.activation = activation
        self.weight = models.VGG19_Weights.IMAGENET1K_V1
        self.model = models.vgg19(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = Identity()
        self.head = ClassifierHead(25088, num_classes, [49, 512], activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        # output = GlobalAvgPool2d(output)
        output = self.head(output)
        return output
