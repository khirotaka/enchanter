import torch
import torch.nn as nn


class SELayer1d(nn.Module):
    def __init__(self, in_features: int, reduction: int = 16) -> None:
        super().__init__()
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class SELayer2d(nn.Module):
    def __init__(self, in_features: int, reduction: int = 16) -> None:
        super().__init__()
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
