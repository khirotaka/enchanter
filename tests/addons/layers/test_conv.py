import torch
import torch.jit
import enchanter.addons.layers as layers


def test_causalconv():
    x = torch.randn(1, 6, 128, requires_grad=True)     # [N, C, L]
    model = layers.CausalConv1d(6, 32, 3)

    print(model.padding)
    out = model(x)
    out = out.sum()
    out.backward()
    assert x.grad != None
