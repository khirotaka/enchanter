import math
import torch
import torch.nn as nn


__all__ = ["CausalConv1d", "Conv1dSame"]


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """
        Causal Conv1d

        Args:
            in_channels: the number of input channels
            out_channels: the number of output channels
            kernel_size: kernel size
            stride: stride
            dilation: rate of dilation
            groups: the number of groups
            bias: if true use bias (default: True)

        """
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias
        )
        self.left_padding: int = (kernel_size - 1) * dilation

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: [N, in_channels, L]

        Returns:
            outputs: [N, out_channels, L]

        """
        inputs = nn.functional.pad(inputs, [self.left_padding, 0])
        return super(CausalConv1d, self).forward(inputs)


class Conv1dSame(nn.Module):
    """

    References: https://github.com/pytorch/pytorch/issues/3867#issuecomment-598264120

    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1
    ) -> None:
        super(Conv1dSame, self).__init__()
        self.cut_last_element: bool = kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1
        self.padding: int = math.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding, stride=stride, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)
