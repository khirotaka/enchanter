# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Union, Tuple, Any
from os import environ as os_environ
from random import seed as std_seed
from numpy import ndarray
from numpy.random import seed as np_seed
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch import device as torch_device
from torch import manual_seed, Tensor, as_tensor
from torch.cuda import is_available as cuda_is_available


from torch.utils.data import TensorDataset


__all__ = [
    "is_jupyter", "get_dataset", "fix_seed"
]


def is_jupyter() -> bool:
    """
    実行中の環境が Jupyter Notebookかどうかを判定します。

    Returns:
        True if run on jupyrer notebook else False

    """
    if "get_ipython" not in globals():
        return False
    env = get_ipython().__class__.__name__      # NOQA
    if env == "TerminalInteractiveShell":
        return False
    return True


def get_dataset(x: Union[ndarray, Tensor], y: Union[ndarray, Tensor] = None) -> Dataset:
    """
    入力された値をもとに `torch.utils.data.TensorDataset` を生成します。

    Examples:
        >>> import torch
        >>> x = torch.randn(512, 6)
        >>> y = torch.randint(0, 9, size=[512])
        >>> ds = get_dataset(x, y)

    Args:
        x (Union[np.ndarray, torch.Tensor]):
        y (Optional[Union[np.ndarray, torch.Tensor]]):

    Returns:
        `torch.utils.data.TensorDataset`
    """
    x = as_tensor(x)

    if y is not None:
        y = as_tensor(y)
        ds = TensorDataset(x, y)
    else:
        ds = TensorDataset(x)

    return ds


def send(batch: Tuple[Any], device: torch_device) -> Tuple[Any]:
    """
    Send `variable` to `device`

    Args:
        batch: Tuple which contain variable
        device: torch.device

    Returns:
        new tuple

    """

    return tuple(map(lambda x: x.to(device) if isinstance(x, Tensor) else x, batch))


def fix_seed(seed: int, deterministic: bool = False, benchmark: bool = False) -> None:
    """
    PyTorch, NumPy, Pure Python Random のSEED値を一括固定します。

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> fix_seed(0)
        >>> x = torch.randn(...)
        >>> y = np.random.randn(...)

    Args:
        seed (int): SEED値
        deterministic (bool): CuDNN上で可能な限り再現性を担保するかどうか
        benchmark (bool):

    Returns:
        None
    """
    std_seed(seed)
    os_environ["PYTHONHASHSEED"] = str(seed)
    np_seed(seed)
    manual_seed(seed)

    if cuda_is_available():
        if deterministic:
            cudnn.deterministic = True
        if benchmark:
            cudnn.benchmark = False


