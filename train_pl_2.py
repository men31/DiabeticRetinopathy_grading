from dr_grading.datasets import STL10DataModule
from dr_grading.models import DenseNet161Lightning, DenseNet161, LightningModelWrapper

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection, F1Score, CohenKappa, MatthewsCorrCoef

torch.set_float32_matmul_precision("medium")


def main():
    num_classes = 10

    # Instantiate datamodule
    stl_dm = STL10DataModule(data_dir="./data", batch_size=32)

    # Metrics
    metrics = MetricCollection(
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

    # Load model
    model = DenseNet161(num_classes=num_classes)
    model = LightningModelWrapper(model, metrics)
    # model = DenseNet161Lightning(num_classes=10)

    # Trainer
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="gpu",
        precision="16-mixed",
        callbacks=[ModelCheckpoint(save_top_k=2, monitor="val_f1", mode="max")],
    )

    # Train and test
    trainer.fit(model, datamodule=stl_dm)
    trainer.test(model, datamodule=stl_dm)


if __name__ == "__main__":
    main()
