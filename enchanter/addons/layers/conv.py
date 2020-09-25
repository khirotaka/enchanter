import torch
import torch.nn as nn


__all__ = ["CausalConv1d"]


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

        Paper: `WaveNet: A Generative Model for Raw Audio <https://arxiv.org/abs/1609.03499>`_

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation

        Args:
            x: [N, in_channels, L]

        Returns:
            [N, out_channels, L]

        """
        x = nn.functional.pad(x, [self.left_padding, 0])
        return super(CausalConv1d, self).forward(x)
