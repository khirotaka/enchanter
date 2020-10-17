import torch
import torch.nn as nn


__all__ = ["CausalConv1d", "TemporalConvBlock"]


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


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.5,
        activation: nn.Module = nn.ReLU(),
        final_activation: bool = False,
    ):
        r"""
        Temporal Convolutional Block

        Paper: `An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling \
                <https://arxiv.org/abs/1803.01271>`_

        Args:
            in_features: the number of input channels
            out_features: the number of output channels
            kernel_size: kernel size
            stride: stride
            dilation: dilation rate
            dropout: dropout rate
            activation: activation function (default: ReLU)
            final_activation: If true, apply the activation function after the residual connection
        """
        super(TemporalConvBlock, self).__init__()
        self.final_activation = final_activation
        self.conv = nn.Sequential(
            nn.utils.weight_norm(
                CausalConv1d(in_features, out_features, kernel_size, stride=stride, dilation=dilation)
            ),
            activation,
            nn.Dropout(dropout),
            nn.utils.weight_norm(
                CausalConv1d(out_features, out_features, kernel_size, stride=stride, dilation=dilation)
            ),
            activation,
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_features, out_features, 1) if in_features != out_features else None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity: torch.Tensor = x
        out: torch.Tensor = self.conv(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        if self.final_activation:
            out = self.activation(out)
        return out
