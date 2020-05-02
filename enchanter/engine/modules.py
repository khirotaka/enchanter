# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import random
import torch
import numpy as np
from torch.utils.data import TensorDataset


__all__ = [
    "is_jupyter", "numpy2tensor", "get_dataset", "fix_seed"
]


def is_jupyter():
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


def numpy2tensor(inputs):
    """
    入力された `np.ndarray` を `torch.Tensor` に変換します。単に `torch.Tensor` が入力された場合は、そのまま値を返します。

    Examples:
        >>> x = np.random.randn(512, 6)     # np.ndarray (dtype=np.float64)
        >>> x = numpy2tensor(x)             # torch.Tensor (dtype=torch.DoubleTensor)

    Args:
        inputs (Union[np.ndarray, torch.Tensor]):

    Returns:
        `torch.Tensor`
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)

    return inputs


def get_dataset(x, y=None):
    """
    入力された値をもとに `torch.utils.data.TensorDataset` を生成します。

    Examples:
        >>> x = torch.randn(512, 6)
        >>> y = torch.randint(0, 9, size=[512])
        >>> ds = get_dataset(x, y)

    Args:
        x (Union[np.ndarray, torch.Tensor]):
        y (Optional[Union[np.ndarray, torch.Tensor]]):

    Returns:
        `torch.utils.data.TensorDataset`
    """
    x = numpy2tensor(x)

    if y is not None:
        y = numpy2tensor(y)
        ds = TensorDataset(x, y)
    else:
        ds = TensorDataset(x)

    return ds


def send(batch, device):
    """
    Send `variable` to `device`

    Args:
        batch: Tuple which contain variable
        device: torch.device

    Returns:
        new tuple

    """

    return tuple(map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, batch))


def fix_seed(seed):
    """
    PyTorch, NumPy, Pure Python Random のSEED値を一括固定します。

    Examples:
        >>> fix_seed(0)
        >>> x = torch.randn(...)
        >>> y = np.random.randn(...)

    Args:
        seed (int): SEED値

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
