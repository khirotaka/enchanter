# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import numpy as np
import torch
import torch.nn as nn


__all__ = [
    "DenseInterpolation"
]


class DenseInterpolation(nn.Module):
    """

    Args:
        seq_len: length of input sequence.
        factor:

    """
    def __init__(self, seq_len, factor):
        super().__init__()
        W = np.zeros((factor, seq_len), dtype=np.float32)

        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1 + m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x):
        """
        Dense Interpolation を入力に適用する。

        Args:
            x (torch.Tensor): 入力する配列の形状は、 `[N, features, seq_len]` を想定

        Returns:
            適用した結果 (torch.Tensor)
        """
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)
        u = torch.bmm(w, x)
        return u.transpose_(1, 2)
