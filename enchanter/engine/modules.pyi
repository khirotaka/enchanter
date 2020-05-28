from typing import Union, Tuple, Any

from numpy import ndarray
from torch.tensor import Tensor
from torch import device as torch_device
from torch.utils.data import Dataset

def is_jupyter() -> bool: ...
def get_dataset(x: Union[ndarray, Tensor], y: Union[ndarray, Tensor]=...) -> Dataset: ...
def send(batch: Tuple[Any], device: torch_device) -> Tuple[Any]: ...
def fix_seed(seed: int): ...
