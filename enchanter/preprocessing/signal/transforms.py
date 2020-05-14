# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import random

import numpy as np
import torch
import torch.nn.functional as F


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
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)

        return data


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
    def __init__(self, window_size, start_position=None):
        if isinstance(window_size, int):
            self.window_size = window_size
        else:
            raise TypeError("`window_size` must be integer.")

        if start_position:
            if start_position >= 0:
                self.start_position = start_position
            else:
                raise ValueError("`start_position` must be 0 and over.")
        else:
            self.start_position = start_position

    def __call__(self, data):
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
            start = random.choice([i for i in range(seq_len - self.window_size)])       #nosec
        else:
            if (seq_len - self.window_size) >= self.start_position:
                start = self.start_position
            else:
                raise IndexError("The start position must be in the range 0 ~ (seq_len - window_size).")

        return data[start:start+self.window_size]


class GaussianNoise:
    """
    Apply gaussian noise to input data.

    Examples:
        >>> import torch
        >>> x = torch.randn(512, 10)
        >>> noise = GaussianNoise()
        >>> y = noise(x)

    Args:
        sigma: 正規分布の :math:`\sigma`
        mu: 正規分布の :math:`\mu`

    """
    def __init__(self, sigma=0.01, mu=0.0):
        self.noise = random.gauss(mu=mu, sigma=sigma)

    def __call__(self, data):
        return data + self.noise


class RandomScaling:
    """
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
        start(float): スケーリング範囲の開始点
        end(float):　スケーリング範囲の終了点

    """
    def __init__(self, start=0.7, end=1.1):
        self.scale = ((end - start) * random.uniform(0.0, 1.0)) + start     #nosec

    def __call__(self, data):
        return data * self.scale


class Pad:
    """
    Fills the end of the given series with the specified method.

    Examples:
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
    def __init__(self, length, value=None):
        self.length = length
        if value:
            self.value = value
        else:
            self.value = 0.0

    def __call__(self, data):
        from_np = False

        if isinstance(data, np.ndarray):
            from_np = True
            data = torch.from_numpy(data)

        seq_len, features = data.shape
        pad_size = self.length - data.shape[0]
        if pad_size < 0:
            raise ValueError("The length of the input series is too short for the padding size.")

        data = data.reshape(1, seq_len, features)
        result = F.pad(data, [0, 0, 0, pad_size], value=self.value)[0]

        if from_np:
            result = result.numpy()

        return result
