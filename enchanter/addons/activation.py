# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from torch.tensor import Tensor
from torch import sigmoid, tanh, tensor
from torch.nn.functional import softplus
from torch.nn import Module, Tanh, Softplus, Parameter


__all__ = [
    "Swish", "mish", "Mish"
]


class Swish(Module):
    """
    Swish活性化関数を適用します。
    """
    def __init__(self, beta: bool = False):
        super(Swish, self).__init__()
        if beta:
            self.weight = Parameter(tensor([1.]), requires_grad=True)
        else:
            self.weight = 1.0

    def forward(self, inputs: Tensor) -> Tensor:
        """
        入力値に対してSwishを適用します。

        Examples:
            >>> import torch
            >>> act = Swish()
            >>> x = torch.randn(2)
            >>> y = act(x)

        Args:
            inputs (torch.Tensor):

        Returns:
            Swishを適用した結果

        """
        out = inputs * sigmoid(self.weight * inputs)
        return out


def mish(x: Tensor) -> Tensor:
    """
    mish活性化関数を適用します。

    Examples:
        >>> import torch
        >>> x = torch.randn(2)
        >>> y = mish(y)

    Args:
        x (torch.Tensor):

    Returns:
        mishを適用した結果 (torch.Tensor)

    """
    return x * tanh(softplus(x))


class Mish(Module):
    """
    Mish活性化関数を適用します。

    """
    def __init__(self):
        super(Mish, self).__init__()
        self.tanh: Module = Tanh()
        self.softplus: Module = Softplus()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        入力に対して Mish を適用します。

        Examples:
            >>> import torch
            >>> act = Mish()
            >>> x = torch.randn(2)
            >>> y = act(y)

        Args:
            inputs(torch.Tensor):

        Returns:
            mishを適用した結果 (torch.Tensor)

        """
        return inputs * self.tanh(self.softplus(inputs))
