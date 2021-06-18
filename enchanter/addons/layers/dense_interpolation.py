# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import numpy as np
from torch import tensor, bmm
from torch import Tensor
from torch.nn import Module


__all__ = ["DenseInterpolation"]


class DenseInterpolation(Module):
    """

    Args:
        seq_len: length of input sequence.
        factor:

    """

    def __init__(self, seq_len: int, factor: int) -> None:
        super(DenseInterpolation, self).__init__()
        W = np.zeros((factor, seq_len), dtype=np.float32)

        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1 + m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ``Dense Interpolation`` to the input.

        Args:
            x (torch.Tensor): The shape of the input array is assumed to be ``[N, seq_len, features]``.

        Returns:
            (torch.Tensor)
        """
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)  # type: ignore
        u = bmm(w, x)
        return u.transpose_(1, 2)
