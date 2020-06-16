from torch.tensor import Tensor
from torch.nn.modules import Module


class DenseInterpolation(Module):
    def __init__(self, seq_len: int, factor: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
