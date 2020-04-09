import torch
import torch.jit

import enchanter.addons.layers as layers


def test_se1d_1():
    x = torch.randn(1, 32, 128).float()
    model = layers.SELayer1d(32)
    try:
        out = model(x)
        is_pass = True
    except Exception:
        is_pass = False

    assert is_pass


def test_se1d_2():
    x = torch.randn(1, 32, 128).float()
    model = torch.jit.trace(layers.SELayer1d(32), (x, ))

    try:
        out = model(x)
        is_pass = True
    except Exception:
        is_pass = False

    assert is_pass


def test_se2d_1():
    x = torch.randn(1, 32, 128, 128).float()
    model = layers.SELayer2d(32)
    try:
        out = model(x)
        is_pass = True
    except Exception:
        is_pass = False

    assert is_pass


def test_se2d_2():
    x = torch.randn(1, 32, 128, 128).float()
    model = torch.jit.trace(layers.SELayer2d(32), (x, ))
    try:
        out = model(x)
        is_pass = True
    except Exception:
        is_pass = False

    assert is_pass
