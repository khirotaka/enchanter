import torch
import torch.jit
import enchanter.addons as addons
from enchanter import models


def test_mlp_1():
    x = torch.randn(32, 10)
    model = models.MLP([10, 128, 1], addons.mish)
    out = model(x)
    assert isinstance(out, torch.Tensor)


def test_mlp_2():
    x = torch.randn(32, 10)
    model = models.MLP([10, 128, 1], addons.Mish())
    jit = torch.jit.trace(model, (x,))
    out = jit(x)
    assert isinstance(out, torch.Tensor)
