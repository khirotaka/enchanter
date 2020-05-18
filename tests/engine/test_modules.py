import torch
import numpy as np
from torch.utils.data import Dataset
import enchanter.engine.modules as modules


def test_get_dataset_1():
    x = torch.randn(32, 128)
    y = torch.randint(0, 9, (32, ))

    ds = modules.get_dataset(x, y)
    return isinstance(ds, Dataset)


def test_get_dataset_2():
    x = torch.randn(32, 128)

    ds = modules.get_dataset(x)
    return isinstance(ds, Dataset)
