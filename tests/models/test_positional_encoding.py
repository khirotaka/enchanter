import torch
from enchanter import models


def test_pe():
    x = torch.randn(1, 128, 32).float()
    pe = models.PositionalEncoding(128, max_len=32)
    out = pe(x)
    assert isinstance(out, torch.Tensor)
