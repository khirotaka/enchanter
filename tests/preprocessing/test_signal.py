import numpy as np
from enchanter.preprocessing import FixedSlidingWindow, adjust_sequences


def test_fsw_1():
    x = np.random.randn(100, 23)
    y = np.random.randint(0, 9, 200)
    try:
        error_occur = False
        sw = FixedSlidingWindow(256, overlap_rate=0.5)
        x, y = sw(x, y)
    except Exception:
        error_occur = True

    assert error_occur


def test_fsw_2():
    x = np.random.randn(1024, 23)
    y = np.random.randint(0, 9, 1024)
    sw = FixedSlidingWindow(256, overlap_rate=0.5)
    x, y = sw(x, y)
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)


def test_adjust_sequences_1():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]
    max_len = max([i.shape[0] for i in x])
    new = adjust_sequences(x)
    assert max_len == new.shape[1]


def test_adjust_sequences_2():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]
    min_len = min([i.shape[0] for i in x])
    new = adjust_sequences(x, np.min)
    assert min_len == new.shape[1]


def test_adjust_sequences_3():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]

    new = adjust_sequences(x, 64)
    assert new.shape[1] == 64


def test_adjust_sequences_4():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]

    new = adjust_sequences(x, dtype=np.int64)
    assert new.dtype == np.int64

