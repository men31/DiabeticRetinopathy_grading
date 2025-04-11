import torch.nn as nn
import torch
from typing import Optional, Union, Sequence, List, Dict, Any

from .__utils import View
from .MoE_module import Encoder_MoE_Attention_Block


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: list,
        activation: str = "relu",
    ):
        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.head = self.__create_architecture()

    def __create_architecture(self):
        layers = []
        in_features = self.in_features
        # Flatten
        layers.append(nn.Flatten())
        # Input layers & hidden layers
        for num in self.hidden_layers:
            # layers.append(nn.BatchNorm1d(in_features))
            layers.append(nn.Linear(in_features=in_features, out_features=num))
            layers.append(nn.BatchNorm1d(num))
            if self.activation == "relu":
                layers.append(nn.ReLU())
            elif self.activation == "gelu":
                layers.append(nn.GELU())
            elif self.activation == "selu":
                layers.append(nn.SELU())
            layers.append(nn.Dropout())

            in_features = num
        # Out layers
        layers.append(
            nn.Linear(in_features=in_features, out_features=self.out_features)
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.head(x)
        return output
    

class ClassiferHead_MoE(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: list,
        num_encoder: int = 3,
        activation: str = "relu",
        num_heads: int = 4,
        seq_len: int = 4,
        dropout: float = 0.3,
    ):
        super(ClassiferHead_MoE, self).__init__()
        # Define parameters
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.num_encoder = num_encoder
        self.activation = activation
        self.dropout = dropout
        self.num_heads = num_heads
        self.seq_len = seq_len

        assert (
            in_features % seq_len == 0
        ), "Embedding size must be divisible by number of heads"
        self.embed_size = in_features // seq_len

        # Create the model
        self.head = self.__create_architecture()

    def __create_architecture(self):
        layers = []
        # Flatten
        layers.append(nn.Flatten())
        # Reshape the feature
        layers.append(View((self.seq_len, self.embed_size)))
        # The number of encoder block
        for num in range(self.num_encoder):
            layers.append(nn.LayerNorm(self.embed_size))
            layers.append(Encoder_MoE_Attention_Block(self.embed_size, self.num_heads))
        layers.append(View(self.in_features))
        layers.append(
            ClassifierHead(
                in_features=self.in_features,
                out_features=self.out_features,
                hidden_layers=self.hidden_layers,
                activation=self.activation,
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.head(x)
        return output
    