import torch
import torch.nn as nn
from typing import List, Callable, Union

class MLP(nn.Module):
    layers: Union[List, nn.ModuleList] = ...
    activation: Callable = ...
    def __init__(self, shapes: List[int], activation: Callable = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
