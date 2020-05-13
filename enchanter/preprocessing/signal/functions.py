# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from collections import Counter

import numpy as np
import pandas as pd
from enchanter.engine.modules import is_jupyter

if is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


__all__ = [
    "FixedSlidingWindow", "adjust_sequences"
]


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
    def __init__(self, window_size, overlap_rate, step_size=None):
        self.window_size = window_size

        if overlap_rate is None and step_size is not None:
            if step_size > 0:
                self.overlap = int(step_size)
        else:
            if not 0.0 < overlap_rate <= 1.0:
                raise AssertionError("overlap_rate ranges from 0.0 to 1.0")

            self.overlap = int(window_size * overlap_rate)

    def transform(self, inputs, verbose=False):
        """

        Args:
            inputs: 2 or 3 dim of np.ndarray
            verbose: if True, show progress bar

        Returns:
            np.ndarray
        """
        seq_len = inputs.shape[0]
        if not seq_len > self.window_size:
            raise Exception("window size must be smaller then input sequence length.")

        if verbose:
            data = []
            for i in tqdm(range(0, seq_len - self.window_size, self.overlap)):
                data.append(inputs[i:i + self.window_size])
        else:
            data = [inputs[i:i + self.window_size] for i in range(0, seq_len-self.window_size, self.overlap)]

        data = np.stack(data, 0)
        return data

    @staticmethod
    def clean(labels):
        """
        Clean up
        Args:
            labels:
        Returns:
        """
        tmp = []
        for lbl in labels:
            window_size = len(lbl)
            counter = Counter(lbl)
            common = counter.most_common()
            values = list(counter.values())
            if common[0][0] == 0 and values[0] == window_size // 2:
                label = common[1][0]
            else:
                label = common[0][0]

            tmp.append(label)

        return np.array(tmp)

    def __call__(self, data, target):
        data = self.transform(data)
        label = self.transform(target)
        label = self.clean(label)
        return data, label


def adjust_sequences(sequences, max_len=None, fill="ffill", dtype=np.float32):
    """
    長さが一定でない系列データを一定の長さに整える関数。
    各サンプルに対して、 `max_len` よりもサンプルの系列が長い場合は、`max_len` まででそれ以降は無視され、
    `max_len` より短い場合は、足りない部分は最後の値を用いて埋められます。

    Args:
        sequences: 要素に、長さが一定でない np.ndarray オブジェクトを持つ Python配列。
                   各要素は2次元配列、第0次元目は系列の長さ、第1次元目は時系列の特徴の数で、全てのサンプルにおいて同じ特徴数である必要があります。
        max_len: 入力された全ての要素を指定した長さに加工します。
                   もし、指定されなかった場合は、与えられたサンプルの中で最も大きい系列長がmax_lenになります。
                   また、np.max や np.min, np.mean と言った関数が与えられた場合、それを用いて新しい長さの系列を生成できます。
        fill: `max_len` より短い場合は、足りない部分を埋める方法を指定します。 fill='ffill' とされた場合、最後の値を用いて埋められます。
               数値(Python int or Python float) が与えられた場合は、その値を用いて値を埋めます。
               fill=["ffill" or int or float]
        dtype:  NumPy のデータ型を指定してください。この値を元に出力系列のデータ型が決定されます。

    Examples:
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
        長さを調整した系列を np.dstack し、[Samples, Seq_len, Features] の3次元配列にし返します。
    """
    features = sequences[0].shape[1]

    lengths = []
    for item in sequences:
        if isinstance(item, np.ndarray):
            lengths.append(item.shape[0])

    if max_len is None:
        max_len = np.max(lengths)
    elif hasattr(max_len, "__call__"):
        max_len = int(max_len(lengths))

    new_seqs = []
    for seq in sequences:
        new_seq = np.zeros((max_len, features), dtype=dtype)
        new_seq[:, :] = np.nan
        new_seq = pd.DataFrame(new_seq)

        if seq.dtype != dtype:
            seq = seq.astype(dtype)

        if max_len > seq.shape[0]:
            new_seq[:seq.shape[0]] = seq
            if fill == "ffill":
                new_seq = new_seq.ffill()
            elif isinstance(fill, int) or isinstance(fill, float):
                new_seq = new_seq.fillna(fill)
            else:
                raise TypeError
        else:
            new_seq[:max_len] = seq[:max_len]

        new_seqs.append(new_seq.values)

    return np.dstack(new_seqs).transpose((2, 0, 1))
