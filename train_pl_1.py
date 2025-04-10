from dr_grading.datasets import STL10DataModule
from dr_grading.models import DenseNet161Lightning
import lightning as L
import torch
from lightning.pytorch.callbacks import RichProgressBar

torch.set_float32_matmul_precision('medium')

def main():
    # Instantiate datamodule
    stl_dm = STL10DataModule(data_dir="./data", batch_size=32)

    # Load model
    model = DenseNet161Lightning(num_classes=10)

    # Trainer
    trainer = L.Trainer(max_epochs=10, accelerator="gpu", precision="16-mixed", callbacks=[RichProgressBar()])

    # Train and test
    trainer.fit(model, datamodule=stl_dm)
    trainer.test(model, datamodule=stl_dm)

if __name__ == "__main__":
    main()


