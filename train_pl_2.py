from dr_grading.datasets import STL10DataModule
from dr_grading.models import DenseNet161Lightning, DenseNet161, LightningModelWrapper

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
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
    model = DenseNet161(num_classes=num_classes, transfer=False)
    model = LightningModelWrapper(model, metrics)
    
    # logger
    tb_logger = TensorBoardLogger("logs", name="my_model")

    # Directory for checkpoints (reuse logger path)
    ckpt_dir = tb_logger.log_dir + "/checkpoints"
    
    # Define callbacks
    # Save best model based on validation F1
    best_ckpt = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=2,
        filename="best-{val_f1:.4f}-{epoch:03d}",
        dirpath= ckpt_dir + "/best/"
    )

    # Save checkpoint every 2 epochs
    periodic_ckpt = ModelCheckpoint(
        every_n_epochs=3,
        save_top_k=-1,  # Save all
        filename="{epoch:03d}",
        dirpath= ckpt_dir + "/periodic/"
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="gpu",
        precision="16-mixed",
        # callbacks=[ModelCheckpoint(save_top_k=2, monitor="val_f1", mode="max")],
        callbacks=[best_ckpt, periodic_ckpt],
        logger=tb_logger,
        enable_model_summary=True,
        )

    # Train and test
    trainer.fit(model, 
                datamodule=stl_dm,
                # ckpt_path=r"checkpoints\periodic\epoch_epoch=011.ckpt",
                )
    trainer.test(model, datamodule=stl_dm)


if __name__ == "__main__":
    main()
