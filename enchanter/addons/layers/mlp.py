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
from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Sequential, Conv1d, ReLU


__all__ = ["MLP", "PositionWiseFeedForward", "ResidualSequential", "AutoEncoder"]


class MLP(Module):
    """
    Class to create MLP.

    Args:
        shapes (List[int]): The number of neurons in each layer of MLP. Assuming an array consisting of int type elements.
                The value of the 0th element of the given array is treated as the number of input dimensions to the model.

        activation (Union[Callable[[torch.Tensor], torch.Tensor], nn.Module]): Activation Function.
                    Differentiable Callable objects such as ``torch.relu`` and ``enchanter.addons.Mish()``

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
        """
        forward propagation processing.

        Args:
            x: input data

        Returns:

        """
        for layer in self.layers[:-1]:  # type: ignore
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)  # type: ignore
        return x


class PositionWiseFeedForward(Module):
    """
    ``PositionWise FeedForward`` proposed in `Attention is all you need`. This class uses ``1x1 Conv1d`` internally.


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
        Apply ``PositionWiseFeedForward`` to the input.

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
        """
        Forward propagation

        Args:
            x: input data

        Returns:

        """
        return x + super(ResidualSequential, self).forward(x)


class AutoEncoder(Module):
    """
    A class that creates an ``AutoEncoder``.

    Examples:
        >>> ae = AutoEncoder([32, 16, 8])

    """

    def __init__(self, shapes: List[int], activation: Union[Callable[[Tensor], Tensor], Module] = relu):
        super(AutoEncoder, self).__init__()
        self.encoder = MLP(shapes, activation)
        self.decoder = MLP(list(reversed(shapes)), activation)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward propagation processing.

        Args:
            x: input data

        Returns:

        """
        mid = self.encoder(x)
        out = self.decoder(mid)
        return out
