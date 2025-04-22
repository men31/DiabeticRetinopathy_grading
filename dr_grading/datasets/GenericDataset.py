"""
The dataset format:

your_dataset_path/
├── train/
│   ├── class0/
│   ├── class1/
│   └── ...
├── val/
│   ├── class0/
│   ├── class1/
│   └── ...
└── test/
    ├── class0/
    ├── class1/
    └── ...

"""

from typing import Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as T
import lightning as L
from torchsampler import ImbalancedDatasetSampler


class GenericImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[T.Compose] = None,
        test_transform: Optional[T.Compose] = None,
        use_imbalance_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_imbalance_sampler = use_imbalance_sampler
        self.is_setup = False

        self.train_dataset: Optional[ImageFolder] = None
        self.val_dataset: Optional[ImageFolder] = None
        self.test_dataset: Optional[ImageFolder] = None

        # Use torchvision.transforms.v2
        if train_transform is not None:
            self.train_transform = train_transform
        else:
            self.train_transform = T.Compose(
                [
                    T.ToImage(),
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(),
                    T.ToDtype(
                        torch.float32, scale=True
                    ),  # Converts and normalizes to [0, 1]
                ]
            )
        if test_transform is not None:
            self.test_transform = test_transform
        else:
            self.test_transform = T.Compose(
                [  
                    T.ToImage(),
                    T.Resize((image_size, image_size)),
                    T.ToDtype(torch.float32, scale=True),
                ]
            )

    def prepare_data(self) -> None:
        pass  # No download logic required for ImageFolder datasets

    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage in (None, "fit"):
            self.train_dataset = ImageFolder(
                self.data_dir / "train", transform=self.train_transform
            )
            self.val_dataset = ImageFolder(
                self.data_dir / "val", transform=self.test_transform
            )

        if stage in (None, "test"):
            self.test_dataset = ImageFolder(
                self.data_dir / "test", transform=self.test_transform
            )
        
        if stage == "predict":
            if self.test_dataset is None:
                self.test_dataset = ImageFolder(
                    self.data_dir / "test", transform=self.test_transform
                )
        
        self.is_setup = True

    def train_dataloader(self) -> DataLoader:
        if self.use_imbalance_sampler:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=ImbalancedDatasetSampler(self.train_dataset),
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
