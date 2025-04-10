import torch
import torch.nn as nn
import torchvision.models as models

from .__utils import Identity
from .Classifier_module import ClassifierHead

class ViT16(nn.Module):
    def __init__(
        self, num_classes: int, transfer: bool = True, activation: str = "relu"
    ):
        super(ViT16, self).__init__()
        self.patch_size = 16
        self.activation = activation
        self.weight = "IMAGENET1K_V1"
        self.model = models.vit_b_16(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.heads.head = Identity()
        self.head = ClassifierHead(768, num_classes, [64], activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch Embbedding
        output = self.model(x)
        output = self.head(output)
        return output
