import torch
import numpy as np
from torch.utils.data import Dataset
import enchanter.engine as engine


def test_numpy2tensor_1():
    x = torch.randn(2)
    y = engine.numpy2tensor(x)
    assert y is torch.Tensor


def test_numpy2_tensor_2():
    x = np.random.randn(2)
    y = engine.numpy2tensor(x)
    assert y is torch.Tensor


def test_get_dataset_1():
    x = torch.randn(32, 128)
    y = torch.randint(0, 9, (32, ))

    ds = engine.get_dataset(x, y)
    return ds is Dataset


def test_get_dataset_2():
    x = torch.randn(32, 128)

    ds = engine.get_dataset(x)
    return ds is Dataset
