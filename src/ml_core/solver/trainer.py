import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import ExperimentTracker, setup_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # 1. Loss function (binary classification → CrossEntropy)
        self.criterion = nn.CrossEntropyLoss()

        # 2. Simple tracker (lists stored in memory)
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    def train_epoch(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> Tuple[float, float, float]:
        self.model.train()

        running_loss = 0.0
        total_samples = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = running_loss / total_samples
        return avg_loss, 0.0, 0.0  # placeholders for accuracy/F1

    def validate(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> Tuple[float, float, float]:
        self.model.eval()

        running_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = running_loss / total_samples
        return avg_loss, 0.0, 0.0  # placeholders

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "val_loss": val_loss,
        }

        path = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss, _, _ = self.train_epoch(train_loader, epoch)
            val_loss, _, _ = self.validate(val_loader, epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            self.save_checkpoint(epoch, val_loss)
