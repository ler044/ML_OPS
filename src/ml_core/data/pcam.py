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

        # Read into temporary NumPy arrays
        images_np = self.x_h5["x"][:]
        labels_np = self.y_h5["y"][:]

        # Filter out all-black or all-white images
        means = images_np.mean(axis=(1, 2, 3))
        mask = (means > 0) & (means < 255)

        # Assign to dataset attributes
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
        img_np = self.images[idx]
        label = int(self.labels[idx])

        # Ensure numerical stability before uint8 cast
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        img = torch.from_numpy(img_np)

        if self.transform is not None:
            img = self.transform(img)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return img, label_tensor
