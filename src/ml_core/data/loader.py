from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # TODO: Define Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),                  # convert numpy to PIL
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),                    # convert PIL to torch.Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # adjust if RGB
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # TODO: Define Paths for X and Y (train and val)
    x_train_path = base_path / "x_train.h5"
    y_train_path = base_path / "y_train.h5"
    x_val_path   = base_path / "x_val.h5"
    y_val_path   = base_path / "y_val.h5"

    # TODO: Instantiate PCAMDataset for train and val
    train_dataset = PCAMDataset(x_path=str(x_train_path), y_path=str(y_train_path), transform=train_transform)
    val_dataset   = PCAMDataset(x_path=str(x_val_path),   y_path=str(y_val_path),   transform=val_transform)
    # TODO: Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True
    )
    
    
    return train_loader, val_loader
