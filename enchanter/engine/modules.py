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
    if "get_ipython" not in globals():
        return False
    env = get_ipython().__class__.__name__      # NOQA
    if env == "TerminalInteractiveShell":
        return False
    return True


def numpy2tensor(inputs):
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)

    return inputs


def get_dataset(x, y=None):
    x = numpy2tensor(x)

    if y is not None:
        y = numpy2tensor(y)
        ds = TensorDataset(x, y)
    else:
        ds = TensorDataset(x)

    return ds


def fix_seed(seed):
    """
    PyTorch, NumPy, Pure Python Random のSEED値を一括固定します。
    Args:
        seed (int): SEED値

    Returns:

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
