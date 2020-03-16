import numpy as np
import torch
import torch.nn as nn


class DenseInterpolation(nn.Module):
    def __init__(self, seq_len: int, factor: int) -> None:
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
