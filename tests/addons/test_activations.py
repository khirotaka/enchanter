import torch
import torch.jit
import numpy as np
import enchanter.addons as addons


def test_swish1():
    swish = torch.jit.script(addons.Swish())
    x = torch.tensor([1.0], dtype=torch.float32)
    out = swish(x)
    out = np.round(out.detach().numpy(), 4)
    assert out == np.array([0.7311]).astype(np.float32)


def test_swish2():
    swish = torch.jit.script(addons.Swish(beta=True))
    x = torch.tensor([1.0], dtype=torch.float32)
    out = swish(x)
    out = np.round(out.detach().numpy(), 4)
    assert out == np.array([0.7311]).astype(np.float32)


def test_mish1():
    mish = torch.jit.script(addons.Mish())
    x = torch.tensor([1.0], dtype=torch.float32)
    out = mish(x)
    out = np.round(out.detach().numpy(), 4)
    assert out == np.array([0.8651]).astype(np.float32)


def test_mish2():
    x = torch.tensor([1.0], dtype=torch.float32)
    out = addons.mish(x)
    out = np.round(out.detach().numpy(), 4)
    assert out == np.array([0.8651]).astype(np.float32)


def test_frelu1d():
    frelu = torch.jit.script(addons.FReLU1d(3))
    x = torch.randn(1, 3, 128)

    try:
        out = frelu(x)
        is_pass = True
    except Exception as e:
        print(e)
        is_pass = False

    assert is_pass


def test_frelu2d():
    frelu = torch.jit.script(addons.FReLU2d(3))
    x = torch.randn(1, 3, 128, 128)
    try:
        out =  frelu(x)
        is_pass = True

    except Exception as e:
        print(e)
        is_pass = False

    assert is_pass
