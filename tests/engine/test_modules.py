from collections import Iterator
import torch
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset
import enchanter.engine.modules as modules


def test_get_dataset_1():
    x = torch.randn(32, 128)
    y = torch.randint(0, 9, (32, ))

    ds = modules.get_dataset(x, y)
    assert isinstance(ds, Dataset)


def test_get_dataset_2():
    x = torch.randn(32, 128)

    ds = modules.get_dataset(x)
    assert isinstance(ds, Dataset)


def test_is_tfds_1():
    x = torch.randn(32, 128)
    y = torch.randint(0, 9, (32,))
    ds = modules.get_dataset(x, y)
    assert modules.is_tfds(ds) is False


def test_is_tfds_2():
    x = np.random.randn(32, 128).astype(np.float32)
    y = np.random.randint(0, 9, (32, )).astype(np.int64)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    assert modules.is_tfds(ds)


def test_tfds_to_numpy():
    x = np.random.randn(32, 128).astype(np.float32)
    y = np.random.randint(0, 9, (32,)).astype(np.int64)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    loader = modules.tfds_to_numpy(ds)
    assert isinstance(loader, Iterator)
