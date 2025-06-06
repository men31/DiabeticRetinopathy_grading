from typing import Any, Optional, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchmetrics import MetricCollection
from torchmetrics.classification import F1Score, CohenKappa, MatthewsCorrCoef
import lightning as L

from .__utils import Identity
from .Classifier_module import ClassifierHead, ClassiferHead_MoE
from .Encoder_module import ConvEncoder
from .Unet_module import UNet

StageType = Literal["train", "val", "test"]


class DenseNet161Lightning(L.LightningModule):
    def __init__(self, num_classes: int = 10, lr: float = 1e-3, finetune: bool = True):
        super().__init__()
        self.save_hyperparameters("lr")

        # Load pre-trained DenseNet161
        self.model = models.densenet161(
            weights=models.DenseNet161_Weights.IMAGENET1K_V1
        )

        # Fine-tune the model
        if not finetune:
            for _, params in self.model.named_parameters():
                params.requires_grad = False

        # Replace classifier to match desired number of classes
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Centralized metrics for all stages
        self.train_metrics = MetricCollection(
            {
                "f1": F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "qwk": CohenKappa(
                    task="multiclass",
                    num_classes=num_classes,
                    weights="quadratic",
                ),
                "mcc": MatthewsCorrCoef(task="multiclass", num_classes=num_classes),
            },
            prefix="train_",
        )

        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def shared_step(self, batch: Any, stage: StageType) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        preds_class = preds.argmax(dim=1)

        # for name, metric in self.metrics[stage].items():
        #     metric.update(preds_class, y)
        # print(self.device)
        # print(self.metrics[stage].device)
        # sys.exit()
        # self.metrics[stage].update(preds_class, y)

        if stage == "train":
            self.train_metrics.update(preds_class, y)
        elif stage == "val":
            self.val_metrics.update(preds_class, y)
        else:
            self.test_metrics.update(preds_class, y)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self.shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self.shared_step(batch, "test")

    def _log_metrics(self, stage: StageType) -> None:
        # for name, metric in self.metrics[stage].items():
        #     self.log(f"{stage}_{name}", metric.compute(), prog_bar=True)
        #     metric.reset()
        # self.log_dict(self.metrics[stage].compute(), prog_bar=True)
        # self.metrics[stage].reset()

        if stage == "train":
            self.log_dict(self.train_metrics.compute(), prog_bar=True)
            self.train_metrics.reset()
        elif stage == "val":
            self.log_dict(self.val_metrics.compute(), prog_bar=True)
            self.val_metrics.reset()
        else:
            self.log_dict(self.test_metrics.compute(), prog_bar=True)
            self.test_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class DenseNet161(nn.Module):
    def __init__(
        self, num_classes: int, transfer: bool = True, activation: str = "relu", 
        in_channels:int=3, encoder_model:Literal['simple', 'unet']="unet",
    ):
        super(DenseNet161, self).__init__()
        self.transfer = transfer
        self.num_classes = num_classes
        self.activation = activation
        self.in_channels = in_channels
        self.encoder_model = encoder_model
        self.hidden_layers = [1028, 512, 512, 256, 64]
        self.weight = models.DenseNet161_Weights.IMAGENET1K_V1
        self.model = self.__create_architecture()
        # self.model = models.densenet161(weights=self.weight)
        # if transfer:
        #     for param in self.model.parameters():
        #         param.requires_grad = False

        # self.model.classifier = Identity()
        # self.head = ClassifierHead(
        #     2208, num_classes, [1028, 512, 512, 256, 64], activation=self.activation
        # )
        # self.head = ClassifierHead(2208, num_classes, self.hidden_layers, activation=self.activation)

    def __create_architecture(self):
        model_lst = []
        if self.in_channels != 3:
            if self.encoder_model == "simple":
                enc_rgb = ConvEncoder(in_channels=self.in_channels, out_channels=3)
            elif self.encoder_model == "unet":
                enc_rgb = UNet(in_channels=self.in_channels, out_channels=3, features=(64, 128), dropout=0.3)
            else:
                raise ValueError(f"[-] Not found encoder model: {self.encoder_model}")
            model_lst.append(enc_rgb)

        model = models.densenet161(weights=self.weight)
        if self.transfer:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = Identity()
        model_lst.append(model)

        head = ClassifierHead(2208, self.num_classes, self.hidden_layers, activation=self.activation)
        model_lst.append(head)

        return nn.Sequential(*model_lst)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output


class DenseNet161_MoE(nn.Module):
    def __init__(
        self, num_classes: int, transfer: bool = True, activation: str = "relu"
    ):
        super(DenseNet161_MoE, self).__init__()
        self.activation = activation
        self.weight = models.DenseNet161_Weights.IMAGENET1K_V1
        self.model = models.densenet161(weights=self.weight)
        if transfer:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = Identity()
        # self.head = ClassifierHead(
        #     2208, num_classes, [1028, 512, 512, 256, 64], activation=self.activation
        # )
        self.head = ClassiferHead_MoE(
            2208,
            num_classes,
            hidden_layers=[1104, 552, 276],
            activation=self.activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        output = self.head(output)
        return output
