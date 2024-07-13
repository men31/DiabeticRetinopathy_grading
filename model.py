from typing import Union, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
# import timm
import torchvision

torch.manual_seed(2000)


def GlobalMaxPool2d(x:torch.Tensor) -> torch.Tensor:
    return torch.amax(torch.amax(x, dim=2, keepdim=True), dim=3, keepdim=True)

def GlobalAvgPool2d(x:torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=[2, 3], keepdim=True)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x


class ClassifierHead(nn.Module):
    def __init__(self, in_features:int, out_features:int, hidden_layers:list, activation:str='relu'):
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
            layers.append(nn.BatchNorm1d(in_features))
            layers.append(nn.Linear(in_features=in_features, out_features=num))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'gelu':
                layers.append(nn.GELU())
            elif self.activation == 'selu':
                layers.append(nn.SELU())
            layers.append(nn.Dropout())

            in_features = num
        # Out layers
        layers.append(nn.Linear(in_features=in_features, out_features=self.out_features))
        return nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = self.head(x)
        return output


class ResNet50(nn.Module):
    def __init__(self, num_classes:int, transfer:bool=True, activation:str='relu'):
        super(ResNet50, self).__init__()
        self.activation = activation
        self.weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=self.weights)
        # Transfer learning
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = Identity()
        self.head = ClassifierHead(2048, num_classes, [32], activation=activation)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        output = self.head(output)
        return output
    

class VGG19(nn.Module):
    def __init__(self, num_classes:int, transfer:bool=True, activation:str='relu'):
        super().__init__()
        self.activation = activation
        self.weight = torchvision.models.VGG19_Weights.IMAGENET1K_V1
        self.model = torchvision.models.vgg19(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = Identity()
        self.head = ClassifierHead(25088, num_classes, [49, 512], activation=activation)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        # output = GlobalAvgPool2d(output)
        output = self.head(output)
        return output


class Inception_V3(nn.Module):
    def __init__(self, num_classes:int, transfer:bool=True, activation:str='relu'):
        super(Inception_V3, self).__init__()
        self.activation = activation
        self.weight = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = torchvision.models.inception_v3(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = Identity()
        self.head = ClassifierHead(2048, num_classes, [64], activation=activation)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if self.training:
            output = output[0]
        output = self.head(output)
        return output
    

class ViT16(nn.Module):
    def __init__(self, num_classes:int, transfer:bool=True, activation:str='relu'):
        super(ViT16, self).__init__()
        self.patch_size = 16
        self.activation = activation
        self.weight = 'IMAGENET1K_V1'
        self.model = torchvision.models.vit_b_16(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.heads.head = Identity()
        self.head = ClassifierHead(768, num_classes, [64], activation=activation)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Patch Embbedding
        output = self.model(x)
        output = self.head(output)
        return output
    

class DenseNet161(nn.Module):
    def __init__(self, num_classes:int, transfer:bool=True, activation:str='relu'):
        super(DenseNet161, self).__init__()
        self.activation = activation
        self.weight = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
        self.model = torchvision.models.densenet161(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = Identity()
        # self.head = ClassifierHead(2208, num_classes, [1028, 512, 512, 64], activation=self.activation)
        self.head = ClassifierHead(2208, num_classes, [64], activation=self.activation)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        output = self.head(output)
        return output
    

class Swin_S(nn.Module):
    def __init__(self, num_classes:int, transfer:bool=True, activation:str='relu'):
        super(Swin_S, self).__init__()
        self.activation = activation
        self.weight = torchvision.models.Swin_S_Weights.IMAGENET1K_V1
        self.model = torchvision.models.swin_s(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.head = Identity()
        # self.head = ClassifierHead(768, num_classes, [64], activation=self.activation)
        self.head = ClassifierHead(768, num_classes, [128, 128, 64], activation=self.activation)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        output = self.head(output)
        return output
    
