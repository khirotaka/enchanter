import torch
import torch.nn as nn
from enchanter.addons.layers import Conv1dSame


class ConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, stride: int) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv1dSame(in_features, out_features, kernel_size, stride=stride), nn.BatchNorm1d(out_features), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
