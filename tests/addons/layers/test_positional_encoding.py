import torch
import enchanter.addons as addons


def test_pe():
    x = torch.randn(1, 128, 32).float()
    pe = addons.layers.PositionalEncoding(128, seq_len=32)
    out = pe(x)
    assert isinstance(out, torch.Tensor)
