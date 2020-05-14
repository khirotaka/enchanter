import torch
import numpy as np

from enchanter.engine.modules import fix_seed
from enchanter.preprocessing.signal import transforms

fix_seed(0)

seq_len = 512
features = 10

torch_x = torch.randn(seq_len, features)      # [Seq_len, Features]
np_x = np.random.randn(seq_len, features)     # [Seq_len, Features]

torch_x_int = torch.randint(1, 10, (seq_len, features))      # [Seq_len, Features]
np_x_int = np.random.randint(1, 10, (seq_len, features))     # [Seq_len, Features]


def test_fixedwindow():
    size = 128
    is_pass = False

    try:
        _ = transforms.FixedWindow(size, -1)
    except ValueError:
        is_pass = True

    assert is_pass


def test_fixedwindow_torch_1():
    size = 128
    fw = transforms.FixedWindow(size)
    out = fw(torch_x)
    assert out.shape[0] == size
    assert out.shape[1] == features


def test_fixedwindow_torch_2():
    size = 1000
    fw = transforms.FixedWindow(size)
    is_pass = False

    try:
        _ = fw(torch_x)
    except IndexError:
        is_pass = True

    assert is_pass


def test_fixedwindow_torch_3():
    size = 128
    fw = transforms.FixedWindow(size, 1000)
    is_pass = False

    try:
        _ = fw(torch_x)
    except IndexError:
        is_pass = True

    assert is_pass


def test_fixedwindow_np_1():
    size = 128
    fw = transforms.FixedWindow(size)
    out = fw(np_x)
    assert out.shape[0] == size
    assert out.shape[1] == features


def test_fixedwindow_np_2():
    size = 1000
    fw = transforms.FixedWindow(size)
    is_pass = False

    try:
        _ = fw(np_x)
    except IndexError:
        is_pass = True

    assert is_pass


def test_fixedwindow_np_3():
    size = 128
    fw = transforms.FixedWindow(size, 1000)
    is_pass = False

    try:
        _ = fw(np_x)
    except IndexError:
        is_pass = True

    assert is_pass


def test_scaling_torch():
    start = 0.7
    end = 1.1

    scale = transforms.RandomScaling(start, end)

    y = scale(torch_x_int)

    assert (start * torch_x_int < y).sum().item() == (seq_len * features)
    assert (y < end * torch_x_int).sum().item() == (seq_len * features)


def test_scaling_np():
    start = 0.7
    end = 1.1

    scale = transforms.RandomScaling(start, end)

    y = scale(np_x_int)

    assert (start * np_x_int < y).sum().item() == (seq_len * features)
    assert (y < end * np_x_int).sum().item() == (seq_len * features)


def test_pad_1():
    pad = transforms.Pad(1000)
    y = pad(torch_x)

    assert y[512:].sum() == 0


def test_pad_2():
    pad = transforms.Pad(1000, 10.0)
    length = pad.length - seq_len
    y = pad(torch_x)
    assert y[512:].sum().item() == (10.0 * length * features)
