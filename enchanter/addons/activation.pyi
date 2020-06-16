from typing import Union
from torch.tensor import Tensor
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class Swish(Module):
    weight: Union[Parameter, float] = ...
    def __init__(self, beta: bool = ...) -> None: ...
    def forward(self, inputs: Tensor) -> Tensor: ...    # type: ignore

def mish(x: Tensor) -> Tensor: ...

class Mish(Module):
    tanh: Module = ...
    softplus: Module = ...
    def __init__(self) -> None: ...
    def forward(self, inputs: Tensor) -> Tensor: ...    # type: ignore
