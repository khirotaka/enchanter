import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    dropout: nn.Module = ...
    def __init__(self, d_model: int, dropout: float = ..., max_len: int = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...     # type: ignore
