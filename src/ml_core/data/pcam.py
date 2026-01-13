from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        # TODO: Initialize dataset
        # 1. Check if files exist
        if not self.x_path.is_file():
            raise FileNotFoundError(f"Image file not found: {self.x_path}")
        if not self.y_path.is_file():
            raise FileNotFoundError(f"Label file not found: {self.y_path}")
        
        # 2. Open h5 files in read mode
        self.x_h5 = h5py.File(self.x_path, "r")
        self.y_h5 = h5py.File(self.y_path, "r")

        images_np = self.x_h5["x"][:]
        labels_np = self.y_h5["y"][:]

        means = images_np.mean(axis=(1, 2, 3))
        mask = (means > 0) & (means < 255)

        self.images = images_np[mask]
        self.labels = labels_np[mask]


    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        image = self.images[idx]
        label = self.labels[idx]

        image = image.clip(0, 255)
        image = torch.from_numpy(image).float() / 255.0

        image = image.permute(2,0,1)
        if self.transform is not None:
            image = self.transform(image)

        label = np.squeeze(label)
        label_tensor = torch.tensor(label).long()
        return image, label_tensor
