from typing import Callable

import torch
import torch.nn as nn
from torch.tensor import Tensor

from enchanter.utils.backend import slice_axis


__all__ = [
    "CausalConv1d"
]


class CausalConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            activation: Callable[[Tensor], Tensor] = torch.relu,
            **kwargs
    ) -> None:
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
        super(CausalConv1d, self).__init__()
        self.dilation: int = dilation
        self.kernel_size: int = kernel_size
        self.padding: int = dilation * (kernel_size - 1)
        self.activation: Callable[[Tensor], Tensor] = activation

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
            **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
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
