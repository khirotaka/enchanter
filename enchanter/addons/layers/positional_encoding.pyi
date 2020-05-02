import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    dropout: nn.Dropout = ...
    def __init__(self, d_model: int, seq_len: int = ..., dropout: float = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...     # type: ignore
