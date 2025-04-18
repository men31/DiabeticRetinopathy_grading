from typing import Any, Optional, Literal, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import F1Score, CohenKappa, MatthewsCorrCoef
import lightning as L

StageType = Literal["train", "val", "test"]


class LightningWrapper_v1(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters("lr")

        self.model = model

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
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, fused=True)
    
    def configure_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)


class LightningModelWrapper(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        metric: MetricCollection,
        criterion: Optional[nn.Module]=None,
        optimizer: Optional[torch.optim.Optimizer]=None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None,
    ):
        super().__init__()
        # self.save_hyperparameters(ignore=["model", "metric", "optimizer", "scheduler"])

        # Load Model
        self.model = model

        # Loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.parameters(), fused=True)
        else:
            self.optimizer = optimizer

        # Scheduler
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
            # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=5, step_size_down=5, mode="exp_range", gamma=0.1)
        else:
            self.scheduler = scheduler

        # Centralized metrics for all stages
        self.train_metrics = metric.clone(prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def shared_step(self, batch: Any, stage: StageType) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        preds_class = preds.argmax(dim=1)

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
        return self.optimizer
    
    def configure_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self.scheduler
