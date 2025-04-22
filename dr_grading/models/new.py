from typing import Union, Iterable, Sequence
from argparse import Namespace
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torchmetrics import Metric

class ToLightning(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        metric: Metric,
        train_loader: Union[DataLoader, GeometricDataLoader],
        val_loader: Union[DataLoader, GeometricDataLoader],
        args: Namespace,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _run_step(self, batch_data: Union[Iterable[torch.Tensor], GeometricData]) -> torch.Tensor:
        if isinstance(batch_data, GeometricData):
            coordinates = batch_data.xyz
            coordinates_pred = self.model(batch_data)
            loss = self.criterion(coordinates_pred, coordinates, batch_data)
        else:
            inputs, targets = batch_data
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        self.metric(outputs, targets)
        return loss

    def training_step(self, batch_data: Union[Iterable[torch.Tensor], GeometricData], batch_idx: int) -> torch.Tensor:
        train_loss = self._run_step(batch_data)
        self.log("train_loss", train_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return train_loss

    def on_train_epoch_end(self) -> None:
        train_metric = self.all_gather(self.metric.compute()).mean()
        self.log("train_metric", train_metric, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # The .reset() method of the metric will automatically be called at the end of an epoch.

    def validation_step(self, batch_data: Union[Iterable[torch.Tensor], GeometricData], batch_idx: int) -> torch.Tensor:
        val_loss = self._run_step(batch_data)
        self.log("val_loss", val_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return val_loss

    def on_validation_epoch_end(self) -> None:
        val_metric = self.all_gather(self.metric.compute()).mean()
        self.log("val_metric", val_metric, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def test_step(self, batch_data: Union[Iterable[torch.Tensor], GeometricData], batch_idx: int) -> torch.Tensor:
        test_loss = self._run_step(batch_data)
        self.log("test_loss", test_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return test_loss

    def on_test_epoch_end(self) -> None:
        test_metric = self.all_gather(self.metric.compute()).mean()
        self.log("test_metric", test_metric, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return self.optimizer

    def configure_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        return self.scheduler

    def configure_callbacks(self) -> Union[Sequence[L.Callback], L.Callback]:
        rich_progress_bar = LitProgressBar()
        return [rich_progress_bar]

# in train script：

def main():
    args = utils.get_args("./config.yaml")
    torch.set_float32_matmul_precision('high')  # Sets the internal precision of float32 matrix multiplications.
    train_set = CIFAR10(
        root="/mnt/device/home/Rich/DataSets",
        train=True,
        transform=Compose([ToTensor(), Resize(224), Normalize(mean=[0.458, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        download=False,
    )
    val_set = CIFAR10(
        root="/mnt/device/home/Rich/DataSets",
        train=False,
        transform=Compose([ToTensor(), Resize(224), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        download=False,
    )
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.train_batch_size, shuffle=False, num_workers=4)
    model = AlexNet(num_classes=10, dropout=0.4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)
    metric = Accuracy(task="multiclass", num_classes=10)
    trainer = Trainer(
        accelerator="cuda",  # auto, cpu, cuda, hpu, ipu, mps, tpu.
        strategy="ddp",
        devices=[2],
        max_epochs=args.epochs,
        # callbacks=[utils.LitProgressBar()],
    )
    model = utils.ToLightning(model, criterion, optimizer, scheduler, metric, train_loader, val_loader, args)
    trainer.fit(model, train_loader, val_loader)