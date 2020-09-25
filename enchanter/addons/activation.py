# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************
from typing import Union

import torch
import torch.nn as nn


__all__ = ["Swish", "mish", "Mish", "FReLU1d", "FReLU2d"]


class Swish(nn.Module):
    """

    Apply Swish activate function.

    """

    def __init__(self, beta: bool = False):
        super(Swish, self).__init__()
        if beta:
            self.weight: Union[nn.Parameter, float] = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.weight = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Swish function.

        Examples:
            >>> import torch
            >>> act = Swish()
            >>> x = torch.randn(2)
            >>> y = act(x)

        Args:
            x (torch.Tensor):

        Returns:
            Result of applying Swish (torch.Tensor)

        """
        out = x * torch.sigmoid(self.weight * x)
        return out


@torch.jit.script
def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Apply mish function.

    Examples:
        >>> import torch
        >>> inputs = torch.randn(2)
        >>> outputs = mish(inputs)

    Args:
        x (torch.Tensor): Input Data

    Returns:
        Result of applying mish (torch.Tensor)

    """
    return x * torch.tanh(nn.functional.softplus(x))


class Mish(nn.Module):
    """
    Apply mish activate function.

    """

    def __init__(self):
        super(Mish, self).__init__()
        self.tanh: nn.Module = nn.Tanh()
        self.softplus: nn.Module = nn.Softplus()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply Mish to the input.

        Examples:
            >>> import torch
            >>> act = Mish()
            >>> x = torch.randn(2)
            >>> y = act(y)

        Args:
            inputs(torch.Tensor):

        Returns:
            Result of applying mish (torch.Tensor)

        """
        return inputs * self.tanh(self.softplus(inputs))


class FReLU1d(nn.Module):
    """
    Applies the Funnel Activation (FReLU) for 1d inputs such as sensor signals.

    Examples:
        >>> inputs = torch.randn(1, 3, 128)     # [N, features, seq_len]
        >>> frelu = FReLU1d(3)
        >>> outputs = frelu(inputs)

    """

    def __init__(self, in_features: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(FReLU1d, self).__init__()
        self.conv = nn.Conv1d(
            in_features,
            in_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_features,
        )
        self.norm = nn.BatchNorm1d(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Funnel Activation (FReLU) for 1d inputs such as sensor signals.

        Args:
            x: torch.Tensor

        Returns:

        """
        out = self.conv(x)
        out = self.norm(out)
        out = torch.max(x, out)
        return out


class FReLU2d(nn.Module):
    """
    Applies the Funnel Activation (FReLU) for 2d inputs such as images.

    Examples:
        >>> inputs = torch.randn(1, 3, 128, 128)     # [N, channels, heights, widths]
        >>> frelu = FReLU2d(3)
        >>> outputs = frelu(inputs)

    """

    def __init__(self, in_features: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(FReLU2d, self).__init__()
        self.conv = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_features,
        )
        self.norm = nn.BatchNorm2d(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Funnel Activation (FReLU) for 2d inputs such as images.

        Args:
            x: torch.Tensor

        Returns:

        """
        out = self.conv(x)
        out = self.norm(out)
        out = torch.max(x, out)
        return out
