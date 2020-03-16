import torch
import torch.jit
import enchanter.addons.layers as layers


def test_1():
    x = torch.randn(1, 128, 512)
    model = layers.DenseInterpolation(128, 30)
    try:
        out = model(x)
        is_pass = True

    except Exception as e:
        print(e)
        is_pass = False

    assert is_pass


def test_2():
    x = torch.randn(1, 128, 512)
    model = torch.jit.trace(layers.DenseInterpolation(128, 30), (x,))
    try:
        out = model(x)
        is_pass = True

    except Exception as e:
        print(e)
        is_pass = False

    assert is_pass
