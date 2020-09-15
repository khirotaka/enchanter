# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import List, Callable, Union
from torch import relu
from torch.tensor import Tensor
from torch.nn import Module, ModuleList, Linear, Sequential, Conv1d, ReLU


__all__ = ["MLP", "PositionWiseFeedForward", "ResidualSequential", "AutoEncoder"]


class MLP(Module):
    """
    MLPを作成するクラス

    Args:
        shapes (List[int]): MLPの各層におけるニューロン数。int型の要素で構成される配列を想定。
                与えられる配列の第0番目の要素の値はモデルへの入力次元の数として扱われます。

        activation (Union[Callable[[torch.Tensor], torch.Tensor], nn.Module]): 活性化関数。
                    torch.relu や enchanter.addons.Mish() 等の微分可能な Callableなオブジェクト

    Examples:
        >>> import enchanter.addons as addons
        >>> model = addons.layers.MLP([10, 512, 128, 5], addons.Mish())
        >>> print(model)
        >>> # ModuleList(
        >>> #    (0): Linear(in_features=10, out_features=512, bias=True)
        >>> #    (1): Linear(in_features=512, out_features=128, bias=True)
        >>> #    (2): Linear(in_features=128, out_features=5, bias=True)
        >>> #)

    """

    def __init__(self, shapes: List[int], activation: Union[Callable[[Tensor], Tensor], Module] = relu):
        super(MLP, self).__init__()
        layers: List[Module] = []
        self.activation: Callable[[Tensor], Tensor] = activation

        for i in range(len(shapes) - 1):
            layers.append(Linear(shapes[i], shapes[i + 1]))

        self.layers: Module = ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:  # type: ignore
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)  # type: ignore
        return x


class PositionWiseFeedForward(Module):
    """
    Attention is all you need. で提案された PositionWiseFeedForward の 1×1 Conv1d を使ったバージョン


    Args:
        d_model: the number of expected features in the Position Wise Feed Forward inputs.
        expansion: Magnification rate of the number of hidden dimensions. Default: 2

    Examples:
        >>> import torch
        >>> import enchanter.addons as addons
        >>> x = torch.randn(1, 128, 512)    # [N, seq_len, features]
        >>> ff = addons.layers.PositionWiseFeedForward(512)
        >>> out = ff(x)

    """

    def __init__(self, d_model: int, expansion: int = 2) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.conv: Module = Sequential(
            Conv1d(d_model, d_model * expansion, 1),
            ReLU(),
            Conv1d(d_model * expansion, d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        入力に対して PositionWiseFeedForward を適用します。

        Args:
            x (torch.Tensor):

        Returns:

        """
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class ResidualSequential(Sequential):
    """

    Examples:
        >>> model = ResidualSequential(
        >>>     Linear(10, 10),
        >>>     ReLU(),
        >>>     Linear(10, 10)
        >>> )

    """

    def forward(self, x: Tensor) -> Tensor:
        return x + super(ResidualSequential, self).forward(x)


class AutoEncoder(Module):
    def __init__(self, shapes: List[int], activation: Union[Callable[[Tensor], Tensor], Module] = relu):
        super(AutoEncoder, self).__init__()
        self.encoder = MLP(shapes, activation)
        self.decoder = MLP(list(reversed(shapes)), activation)

    def forward(self, x):
        mid = self.encoder(x)
        out = self.decoder(mid)
        return out
