import random


class Compose:
    """
    Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Examples:
        >>> Compose([
        >>>     FixedWindow(128)
        >>> ])

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
        >>> x = np.random.randn(512, 18)    # [N, features]
        >>> fw = FixedWindow(128)
        >>> out = fw(x)
        >>> out.shape       # [128, 18]

    """
    def __init__(self, window_size, start_position=None):
        self.window_size = window_size
        self.start_position = start_position

    def __call__(self, data):
        seq_len = data.shape[0]

        if not seq_len > self.window_size:
            raise Exception("window size must be smaller then input sequence length.")

        if not self.start_position:
            start = random.choice([i for i in range(seq_len - self.window_size)])
        else:
            start = self.start_position

        return data[start:start+self.window_size]
