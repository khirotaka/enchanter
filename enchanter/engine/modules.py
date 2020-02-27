# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Dict

import torch
import numpy as np
from torch.utils.data import TensorDataset, Dataset


def is_jupyter() -> bool:
    if "get_ipython" not in globals():
        return False
    env = get_ipython().__class__.__name__
    if env == "TerminalInteractiveShell":
        return False
    return True


def numpy2tensor(inputs: np.ndarray) -> torch.Tensor:
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)

    return inputs


def get_dataset(x, y=None) -> Dataset:
    x = numpy2tensor(x)

    if y is not None:
        y = numpy2tensor(y)
        ds = TensorDataset(x, y)
    else:
        ds = TensorDataset(x)

    return ds
