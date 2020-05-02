from typing import Callable, Optional

import torch
import torch.nn as nn



class CausalConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.relu,
            **kwargs
    ) -> None:
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
