from torch.tensor import Tensor
from torch import nn


__all__ = ["SELayer1d", "SELayer2d"]


class SELayer1d(nn.Module):
    def __init__(self, in_features: int, reduction: int = 16) -> None:
        super(SELayer1d, self).__init__()
        reduction_size = max(1, in_features // reduction)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, reduction_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_size, in_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        scaling = x * y.expand_as(x)

        return scaling


class SELayer2d(nn.Module):
    def __init__(self, in_features: int, reduction: int = 16) -> None:
        super(SELayer2d, self).__init__()
        reduction_size = max(1, in_features // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, reduction_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_size, in_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        scaling = x * y.expand_as(x)

        return scaling
