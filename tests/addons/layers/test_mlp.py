import torch
import torch.jit
import torch.nn as nn
import enchanter.addons as addons


def test_mlp_1():
    x = torch.randn(32, 10)
    model = addons.layers.MLP([10, 128, 1], addons.mish)
    out = model(x)
    assert isinstance(out, torch.Tensor)


def test_mlp_2():
    x = torch.randn(32, 10)
    model = addons.layers.MLP([10, 128, 1], addons.Mish())
    jit = torch.jit.trace(model, (x,))
    out = jit(x)
    assert isinstance(out, torch.Tensor)


def test_ff_1():
    x = torch.randn(1, 128, 64)     # [N, seq_len, features]
    model = addons.layers.PositionWiseFeedForward(64)
    out = model(x)
    assert isinstance(out, torch.Tensor)


def test_ff_2():
    x = torch.randn(1, 128, 64)
    model = addons.layers.PositionWiseFeedForward(64)
    jit = torch.jit.trace(model, (x, ))
    out = jit(x)
    assert isinstance(out, torch.Tensor)


def test_res_seq_1():
    x = torch.randn(1, 64)
    model = addons.layers.ResidualSequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64)
    )
    out = model(x)
    assert isinstance(out, torch.Tensor)


def test_res_seq_2():
    x = torch.randn(1, 64)
    model = addons.layers.ResidualSequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64)
    )
    jit = torch.jit.trace(model, (x, ))
    out = jit(x)
    assert isinstance(out, torch.Tensor)


def test_ae_1():
    x = torch.randn(1, 64)
    model = addons.layers.AutoEncoder([64, 32, 16, 8])
    jit = torch.jit.trace(model, (x, ))
    out = jit(x)
    assert isinstance(out, torch.Tensor)

