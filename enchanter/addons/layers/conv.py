from torch import relu
from torch.nn import Module, Conv1d

from enchanter.utils.backend import slice_axis


__all__ = [
    "CausalConv1d"
]


class CausalConv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation=relu, **kwargs):
        """
        Causal Conv 1d

        Warnings:
            `torch.jit` を用いたJust-In-Time Compile には現在対応していません。

        Args:
            in_channels: 入力チャンネル数
            out_channels: 出力チャンネル数
            kernel_size: カーネルサイズ
            dilation: dilation
            activation: 活性化関数
            **kwargs:

        """
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.padding = dilation * (kernel_size - 1)
        self.activation = activation

        self.conv1d = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
            **kwargs
        )

    def forward(self, x):
        """
        順伝搬処理


        Args:
            x: [N, C, L] の３次元配列

        Returns:
            CausalConv1dを適用した結果 [N, C, L]

        """
        out = self.conv1d(x)
        if self.activation is not None:
            out = self.activation(out)

        if self.kernel_size > 0:
            out = slice_axis(out, axis=2, begin=0, end=-self.padding)

        return out
