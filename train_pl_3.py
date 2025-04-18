from typing import Literal

from dr_grading.datasets import STL10DataModule, GenericImageDataModule
from dr_grading.models import (DenseNet161Lightning, 
                               DenseNet161, 
                               Swin_V2_B,
                               Swin_V2_S,
                               LightningModelWrapper, 
                               load_state_from_ckpt)
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
        every_n_epochs=5,
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
                # FourierTransform(shift=True, return_abs=True),
                v2.RandomApply(torch.nn.ModuleList([
                    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                ]), p=0.3),
                v2.RandomAdjustSharpness(2, p=0.4),
                v2.RandomAutocontrast(p=0.4),
                v2.ToDtype(
                    torch.float32, 
                    scale=True
                ),  # Converts and normalizes to [0, 1]
            ]
        )
    else:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((image_size, image_size)),
                # FourierTransform(shift=True, return_abs=True),
                v2.ToDtype(
                    torch.float32, 
                    scale=True
                ),  # Converts and normalizes to [0, 1]
            ]
        )


def main():
    num_classes = 5
    # Get transform
    train_transform = get_transform("train", image_size=512)
    test_transform = get_transform("test", image_size=512)

    # Instantiate datamodule
    data_dir = r"D:\Aj_Aof_Work\OCT_Disease\DATASET\APTOS2019_V4"
    datamodule = GenericImageDataModule(
        data_dir=data_dir, 
        batch_size=8, 
        num_workers=8,
        train_transform=train_transform,
        test_transform=test_transform,
        use_imbalance_sampler=True,
        )

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
    # model = DenseNet161(num_classes=num_classes, transfer=False)
    model = Swin_V2_S(num_classes=num_classes, transfer=True)
    model = LightningModelWrapper(model, metrics)

    # Load checkpoint
    # ckpt_dir = r"aptos2019_logs\SWIN_V2_S\version_1\checkpoints\lowest_loss\lowest-val_loss=0.5442-epoch=396.ckpt"
    # model = load_state_from_ckpt(model, ckpt_dir)


    # Get logger and callbacks
    tb_logger, callbacks = set_logger_callbacks(logger_name="aptos2019_logs", model_name="SWIN_V2_S")

    # Trainer
    trainer = L.Trainer(
        max_epochs=500,
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
        # ckpt_path=r"aptos2019_logs\SWIN_V2_S\version_3\checkpoints\periodic\epoch=229.ckpt",
    )
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
