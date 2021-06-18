# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from collections import Counter
from typing import Union, List, Type, Optional, Callable

from numpy import array, stack, ndarray, max as np_max, zeros, nan, float32, dstack
from pandas import DataFrame
from tqdm.auto import tqdm


__all__ = ["FixedSlidingWindow", "adjust_sequences"]


_Numerical = Union[int, float]


class FixedSlidingWindow:
    """Fixed sliding window.

    Examples::
        >>> import numpy as np
        >>> from enchanter.preprocessing import signal
        >>> x = np.random.randn(1024, 23)
        >>> y = np.random.randint(0, 9, 1024)
        >>> sw = signal.FixedSlidingWindow(256, overlap_rate=0.5)
        >>> x, y = sw(x, y)
        >>> x.shape     # [6, 256, 23]
        >>> y.shape     # [6, ]

    Args:
        window_size (int): Window size
        overlap_rate (float): overrap rate
        step_size (Optional[int]): step size

    Raises:
        AssertionError: an error occur when
            argument overlap_rate under 0.0 or over 1.0.n error occurred.

    """

    def __init__(self, window_size: int, overlap_rate: Union[float, None], step_size: Optional[int] = None) -> None:
        self.window_size: int = window_size

        if overlap_rate is None and step_size is not None:
            if step_size > 0:
                self.overlap = int(step_size)
        elif isinstance(overlap_rate, float):
            if not 0.0 < overlap_rate <= 1.0:
                raise AssertionError("overlap_rate ranges from 0.0 to 1.0")

            self.overlap = int(window_size * overlap_rate)
        else:
            raise ValueError

    def transform(self, inputs: ndarray, verbose: bool = False) -> ndarray:
        """
        Apply Fixed Sliding Window

        Args:
            inputs: 2 or 3 dim of np.ndarray
            verbose: if True, show progress bar

        Returns:
            np.ndarray

        """
        seq_len = inputs.shape[0]
        if not seq_len > self.window_size:
            raise IndexError(
                "window size ({}) must be smaller then input sequence length ({}).".format(self.window_size, seq_len)
            )

        if verbose:
            data = []
            for i in tqdm(range(0, seq_len - self.window_size, self.overlap)):
                data.append(inputs[i : i + self.window_size])
        else:
            data = [inputs[i : i + self.window_size] for i in range(0, seq_len - self.window_size, self.overlap)]

        data = stack(data, 0)
        return data

    @staticmethod
    def clean(labels: ndarray) -> ndarray:
        """
        Clean up

        Args:
            labels:

        Returns:

        """
        tmp = []
        for lbl in labels:
            window_size = len(lbl)
            counter: Counter = Counter(lbl)
            common = counter.most_common()
            values = list(counter.values())
            if common[0][0] == 0 and values[0] == window_size // 2:
                label = common[1][0]
            else:
                label = common[0][0]

            tmp.append(label)

        return array(tmp)

    def __call__(self, data, target):
        data = self.transform(data)
        label = self.transform(target)
        label = self.clean(label)
        return data, label


def adjust_sequences(
    sequences: List[ndarray],
    max_len: Optional[Union[int, Callable[[List[int]], int]]] = None,
    fill: Union[str, _Numerical] = "ffill",
    dtype: Type = float32,
) -> ndarray:
    """
    The function to adjust the length of the series data to a certain value.
    For each sample, if the series is longer than ``max_len``, the length is up to ``max_len``, and the rest is ignored.
    If it is shorter than ``max_len``, the missing part is filled in with the last value.

    Args:
        sequences: A Python list whose elements have ``np.ndarray`` objects that are not constant in length.
                   Each element is a 2D array, the 0th dimension is the length of the series, the 1st dimension is
                   the number of features in the time series, and all samples must have the same number of features.

        max_len: Processes all input elements to the specified length.
                    If not specified, the largest sequence length in a given sample will be max_len.
                    Also, given functions such as np.max, np.min, and np.mean,
                    you can use them to generate new length series.

        fill: If it is shorter than ``max_len``, specify how to fill in the missing parts.
                If ``fill ='ffill'``, it will be filled with the last value.
                If a number (Python int or Python float) is given, use that value to fill the value.
                ``fill=["ffill" or int or float]``

        dtype:  Specify the data type of NumPy. The data type of the output series is determined based on this value.
                ``dtype`` must be a float.

    Examples:
        >>> import numpy as np
        >>> x = [
        >>>         np.array([[i] for i in [1, 2, 3, 4, 5]]),
        >>>         np.array([[i] for i in [1, 2, 3, 4, 5, 6, 7, 8]]),
        >>>         np.array([[i] for i in [1, 2, 3]]),
        >>> ]
        >>> out = adjust_sequences(x)
        >>> out[-1]
        >>> # array([[1.],
        >>> #        [2.],
        >>> #        [3.],
        >>> #        [3.],
        >>> #        [3.],
        >>> #        [3.],
        >>> #        [3.],
        >>> #        [3.]])
        >>> out = adjust_sequences(x, np.min)
        >>> out
        >>> # array([[[1],
        >>> #         [2],
        >>> #         [3]],
        >>> #
        >>> #        [[1],
        >>> #         [2],
        >>> #         [3]],
        >>> #
        >>> #        [[1],
        >>> #         [2],
        >>> #         [3]]])

    Returns:
        Returns a 3D array of ``[Sample, Seq_len, Features]``.

    """
    features = sequences[0].shape[1]

    lengths = []
    for item in sequences:
        if isinstance(item, ndarray):
            lengths.append(item.shape[0])

    if max_len is None:
        maximum_len = np_max(lengths)
    elif callable(max_len):
        maximum_len = int(max_len(lengths))
    else:
        maximum_len = max_len

    new_seqs = []
    for seq in sequences:
        new_seq = zeros((maximum_len, features), dtype=dtype)
        new_seq[:, :] = nan
        new_seq = DataFrame(new_seq)

        if seq.dtype != dtype:
            seq = seq.astype(dtype)

        if maximum_len > seq.shape[0]:
            new_seq[: seq.shape[0]] = seq
            if fill == "ffill":
                new_seq = new_seq.ffill()
            elif isinstance(fill, int) or isinstance(fill, float):
                new_seq = new_seq.fillna(fill)
            else:
                raise TypeError
        else:
            new_seq[:maximum_len] = seq[:maximum_len]

        new_seqs.append(new_seq.values)

    return dstack(new_seqs).transpose((2, 0, 1))
