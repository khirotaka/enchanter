import torch
import torch.jit
import numpy as np
import enchanter.addon as addon


def test_swish1():
    swish = torch.jit.script(addon.Swish())
    x = torch.tensor([1.0], dtype=torch.float32)
    out = swish(x)
    out = np.round(out.detach().numpy(), 4)
    assert out == np.array([0.7311]).astype(np.float32)


def test_swish2():
    swish = torch.jit.script(addon.Swish(beta=True))
    x = torch.tensor([1.0], dtype=torch.float32)
    out = swish(x)
    out = np.round(out.detach().numpy(), 4)
    assert out == np.array([0.7311]).astype(np.float32)


def test_mish1():
    mish = torch.jit.script(addon.Mish())
    x = torch.tensor([1.0], dtype=torch.float32)
    out = mish(x)
    out = np.round(out.detach().numpy(), 4)
    assert out == np.array([0.8651]).astype(np.float32)


def test_mish2():
    x = torch.tensor([1.0], dtype=torch.float32)
    out = addon.mish(x)
    out = np.round(out.detach().numpy(), 4)
    assert out == np.array([0.8651]).astype(np.float32)
