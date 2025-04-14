from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import STL10
import torchvision.transforms as T
import lightning as L

class STL10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        image_size: int = 224,
        train_transform : Optional[T.Compose] = None,
        test_transform : Optional[T.Compose] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.image_size = image_size

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

        # Define transformations
        if train_transform is None:
            self.train_transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
        else:
            print("Use custom transform")
            self.train_transform = train_transform

        if test_transform is None:     
            self.test_transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ])
        else:
            print("Use custom transform")
            self.test_transform = test_transform

    def prepare_data(self) -> None:
        # Download STL10 if necessary
        STL10(self.data_dir, split='train', download=True)
        STL10(self.data_dir, split='test', download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            full_train = STL10(self.data_dir, split='train', transform=self.train_transform)
            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size
            self.train_dataset, self.val_dataset = random_split(full_train, [train_size, val_size])

        if stage in (None, "test"):
            self.test_dataset = STL10(self.data_dir, split='test', transform=self.test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
