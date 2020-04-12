# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import torch
import torch.jit
import torch.nn as nn


__all__ = [
    "Swish", "mish", "Mish"
]


class Swish(nn.Module):
    """
    Swish活性化関数を適用します。
    """
    def __init__(self, beta=False):
        nn.Module.__init__(self)
        if beta:
            self.weight = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        else:
            self.weight = 1.0

    def forward(self, inputs):
        """
        入力値に対してSwishを適用します。

        Examples:
            >>> act = Swish()
            >>> x = torch.randn(2)
            >>> y = act(x)

        Args:
            inputs (torch.Tensor):

        Returns:
            Swishを適用した結果

        """
        out = inputs * torch.sigmoid(self.weight * inputs)
        return out


def mish(x):
    """
    mish活性化関数を適用します。

    Examples:
        >>> x = torch.randn(2)
        >>> y = mish(y)

    Args:
        x (torch.Tensor):

    Returns:
        mishを適用した結果 (torch.Tensor)

    """
    return x * torch.tanh(nn.functional.softplus(x))


class Mish(nn.Module):
    """
    Mish活性化関数を適用します。

    """
    def __init__(self):
        nn.Module.__init__(self)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, inputs):
        """
        入力に対して Mish を適用します。

        Examples:
            >>> act = Mish()
            >>> x = torch.randn(2)
            >>> y = act(y)

        Args:
            inputs(torch.Tensor):

        Returns:
            mishを適用した結果 (torch.Tensor)

        """
        return inputs * self.tanh(self.softplus(inputs))
