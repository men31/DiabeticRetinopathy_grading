from typing import Literal

from dr_grading.datasets import STL10DataModule, GenericImageDataModule
from dr_grading.models import DenseNet161Lightning, DenseNet161, LightningModelWrapper
from dr_grading.preprocessing import FourierTransform

import lightning as L
import torch
from torchvision.transforms import v2
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import MetricCollection, F1Score, CohenKappa, MatthewsCorrCoef

torch.set_float32_matmul_precision("medium")
StageType = Literal["train", "val", "test"]


def set_logger_callbacks(logger_name: str = "logs", model_name: str = "my_model"):
    # logger
    tb_logger = TensorBoardLogger(logger_name, name=model_name)

    # Directory for checkpoints (reuse logger path)
    ckpt_dir = tb_logger.log_dir + "/checkpoints"

    # Define callbacks
    # Save best model based on validation F1
    best_ckpt = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=2,
        filename="best-{val_f1:.4f}-{epoch:03d}",
        dirpath=ckpt_dir + "/best_f1/",
    )

    # Save the lowst validation loss
    lowest_ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="lowest-{val_loss:.4f}-{epoch:03d}",
        dirpath=ckpt_dir + "/lowest_loss/",
    )

    # Save checkpoint every 2 epochs
    periodic_ckpt = ModelCheckpoint(
        every_n_epochs=3,
        save_top_k=-1,  # Save all
        filename="{epoch:03d}",
        dirpath=ckpt_dir + "/periodic/",
    )

    return tb_logger, [best_ckpt, lowest_ckpt, periodic_ckpt]


def get_transform(stage: StageType, image_size: int = 224) -> v2.Compose:
    if stage == "train":
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((image_size, image_size)),
                v2.RandomHorizontalFlip(),
                FourierTransform(shift=True, return_abs=True),
                v2.ToDtype(
                    torch.float32,
                    # scale=True
                ),  # Converts and normalizes to [0, 1]
            ]
        )
    else:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((image_size, image_size)),
                FourierTransform(shift=True, return_abs=True),
                v2.ToDtype(
                    torch.float32,
                    # scale=True
                ),  # Converts and normalizes to [0, 1]
            ]
        )


def main():
    num_classes = 10
    # Get transform
    train_transform = get_transform("train")
    test_transform = get_transform("test")

    # Instantiate datamodule
    # data_dir = r"D:\Aj_Aof_Work\OCT_Disease\DATASET\APTOS2019_V4"
    # datamodule = GenericImageDataModule(
    #     data_dir=data_dir,
    #     batch_size=32,
    #     num_workers=8,
    #     train_transform=train_transform,
    #     test_transform=test_transform,
    # )
    datamodule = STL10DataModule(data_dir="./data", batch_size=32, 
                                 train_transform=train_transform, test_transform=test_transform)

    # Metrics
    metrics = MetricCollection(
        {
            "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
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
    model = DenseNet161(num_classes=num_classes, transfer=False)
    model = LightningModelWrapper(model, metrics)

    # Get logger and callbacks
    tb_logger, callbacks = set_logger_callbacks(
        logger_name="aptos2019_logs", model_name="DenseNet161"
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="gpu",
        precision="16-mixed",
        callbacks=callbacks,
        logger=tb_logger,
        enable_model_summary=True,
    )

    # Train and test
    trainer.fit(
        model,
        datamodule=datamodule,
        # ckpt_path=r"checkpoints\periodic\epoch_epoch=011.ckpt",
    )
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
