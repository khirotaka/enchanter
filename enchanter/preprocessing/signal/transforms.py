# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from pprint import pformat
from random import choice, gauss, uniform
from typing import List, Callable, Union, Optional

from numpy import ndarray
from torch import from_numpy, Tensor
from torch.nn.functional import pad


__all__ = ["Compose", "FixedWindow", "GaussianNoise", "RandomScaling", "Pad"]


class Compose:
    """
    Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Examples:
        >>> import torch
        >>> x = torch.randn(512, 10)
        >>> transform = Compose([
        >>>     FixedWindow(128),
        >>>     GaussianNoise(),
        >>>     RandomScaling()
        >>> ])
        >>> y = transform(x)

    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms: List[Callable] = transforms

    def __call__(self, data: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        for t in self.transforms:
            data = t(data)

        return data

    def insert(self, index: int, modules: Callable) -> None:
        self.transforms.insert(index, modules)

    def append(self, module: Callable) -> None:
        self.transforms.append(module)

    def extend(self, modules: List[Callable]) -> None:
        self.transforms.extend(modules)

    def __repr__(self):
        return pformat(
            ["({}): {}".format(i, j.__class__.__name__) for i, j in enumerate(self.transforms)],
            width=40,
        )


class FixedWindow:
    """
    Cropping the input data with a fixed length window.

    Args:
        window_size (int): Length of the series to be trimmed from the input data.
        start_position (Optional[int]): Position to start clipping.

    Examples:
        >>> import numpy as np
        >>> x = np.random.randn(512, 18)    # [seq_len, features]
        >>> fw = FixedWindow(128)
        >>> out = fw(x)
        >>> out.shape       # [128, 18]

    """

    def __init__(self, window_size: int, start_position: Optional[int] = None) -> None:
        if isinstance(window_size, int):
            self.window_size: int = window_size
        else:
            raise TypeError("`window_size` must be integer.")

        if start_position:
            if start_position >= 0:
                self.start_position: Optional[int] = start_position
            else:
                raise ValueError("`start_position` must be 0 and over.")
        else:
            self.start_position: Optional[int] = start_position  # type: ignore

    def __call__(self, data: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        """

        Args:
            data: input shape must be `[the length of sequence, features]`

        Returns:
            Cropped the data

        """
        seq_len = data.shape[0]

        if not seq_len > self.window_size:
            raise IndexError("`window size` must be smaller then input sequence length.")

        if not self.start_position:
            start = choice([i for i in range(seq_len - self.window_size)])  # nosec
        else:
            if (seq_len - self.window_size) >= self.start_position:
                start = self.start_position
            else:
                raise IndexError("The start position must be in the range 0 ~ (seq_len - window_size).")

        return data[start : start + self.window_size]


class GaussianNoise:
    r"""
    Apply gaussian noise to input data.

    Examples:
        >>> import torch
        >>> x = torch.randn(512, 10)
        >>> noise = GaussianNoise()
        >>> y = noise(x)

    Args:
        sigma: normal distribution paramter :math:`\sigma`
        mu: normal distribution paramter :math:`\mu`

    """

    def __init__(self, sigma: float = 0.01, mu: float = 0.0) -> None:
        self.noise: float = gauss(mu=mu, sigma=sigma)

    def __call__(self, data: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        return data + self.noise


class RandomScaling:
    r"""
    Scaling the original sequence by a random value in the range `start` and `end` following Eq.

    .. math::

        \mathcal{L_r}(\mathbf{ S }) = \mathbf{ S } \cdot ((\mathrm{end} - \mathrm{start}) * rand() + \mathrm{start})

    References:
        An End-to-End Multi-Task and Fusion CNN for Inertial-Based Gait Recognition

    Examples:
        >>> import torch
        >>> x = torch.randn(512, 10)
        >>> scale = RandomScaling()
        >>> y = scale(x)

    Args:
        start(float): Starting point of scaling range.
        end(float):　End point of scaling range.

    """

    def __init__(self, start: float = 0.7, end: float = 1.1) -> None:
        self.scale: float = ((end - start) * uniform(0.0, 1.0)) + start  # nosec

    def __call__(self, data: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        return data * self.scale


class Pad:
    """
    Fills the end of the given series with the specified method.

    Examples:
        >>> import torch
        >>> x = torch.randn(10, 3)  # [seq_len, features]
        >>> pad = Pad(20)
        >>> y = pad(x)              # [20, 3]
        >>> # OR
        >>> pad = Pad(20, 1.0)
        >>> y = pad(x)

    Args:
        length(int): Length of output series.
        value(Optional[float]): Value to fill.

    """

    def __init__(self, length: int, value: Optional[Union[int, float]] = None) -> None:
        self.length: int = length
        if value:
            self.value: Union[int, float] = value
        else:
            self.value: Union[int, float] = 0.0  # type: ignore

    def __call__(self, data: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        from_np = False

        if isinstance(data, ndarray):
            from_np = True
            data = from_numpy(data)

        seq_len, features = data.shape
        pad_size = self.length - data.shape[0]
        if pad_size < 0:
            raise ValueError("The length of the input series is too short for the padding size.")

        data = data.reshape(1, seq_len, features)
        result = pad(data, [0, 0, 0, pad_size], value=self.value)[0]

        if from_np:
            result = result.numpy()

        return result
