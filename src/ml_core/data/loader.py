from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
import torch
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
        transforms.ToPILImage(),                 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),                   
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # TODO: Define Paths for X and Y (train and val)
    h5_files = list(base_path.glob("*.h5"))
    x_train_path = next(f for f in h5_files if "train" in f.name and "_x" in f.name)
    y_train_path = next(f for f in h5_files if "train" in f.name and "_y" in f.name)
    x_val_path   = next(f for f in h5_files if "valid" in f.name and "_x" in f.name)
    y_val_path   = next(f for f in h5_files if "valid" in f.name and "_y" in f.name)




    # TODO: Instantiate PCAMDataset for train and val
    train_dataset = PCAMDataset(x_train_path, y_train_path , transform=train_transform)
    val_dataset   = PCAMDataset(x_val_path,   y_val_path,   transform=val_transform)

    labels = []
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        labels.append(label)
    labels_tensor = torch.tensor(labels)

    positives = list(labels_tensor).count(1)
    negatives = list(labels_tensor).count(0)
    num_samples = positives+negatives

    class_counts = [negatives, positives]
    class_weights = []
    for count in class_counts:
        class_weights.append(num_samples / count)

    weights = []
    for i in range(num_samples):
        label = labels_tensor[i].item()
        weights.append(class_weights[label])

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=num_samples,
        replacement=True
    )

    # TODO: Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        sampler=sampler,
        num_workers=config["data"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["batch_size"],
    )


    return train_loader, val_loader
