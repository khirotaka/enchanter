from typing import Callable, Optional

from torch import relu
from torch.tensor import Tensor
from torch.nn.modules import Module



class CausalConv1d(Module):
    dilation: int = ...
    kernel_size: int = ...
    padding: int = ...
    activation: Optional[Callable[[Tensor], Tensor]] = ...

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            activation: Optional[Callable[[Tensor], Tensor]] = relu,
            **kwargs
    ) -> None: ...

    def forward(self, x: Tensor) -> Tensor: ...
