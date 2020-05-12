from typing import List, Callable, Union, Optional

from torch import Tensor
from numpy import ndarray


class Compose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms
        ...
    def __call__(self, data: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]: ...

class FixedWindow:
    def __init__(self, window_size: int, start_position: Optional[int] = None) -> None:
        self.window_size = window_size
        self.start_position = start_position
        ...
    def __call__(self, data: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]: ...
