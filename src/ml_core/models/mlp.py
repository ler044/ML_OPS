from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_units: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim
        
        layers = []
        prev_units = input_dim
        
        for unit in hidden_units:
            layers.append(nn.Linear(prev_units, unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_units = unit
        
        layers.append(nn.Linear(prev_units, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        x = torch.flatten(x, start_dim=1)
        return self.model(x)
