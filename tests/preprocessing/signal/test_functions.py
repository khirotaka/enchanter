import numpy as np
from enchanter.preprocessing import signal


def test_fsw_1():
    x = np.random.randn(100, 23)
    y = np.random.randint(0, 9, 200)
    try:
        error_occur = False
        sw = signal.FixedSlidingWindow(256, overlap_rate=0.5)
        x, y = sw(x, y)
    except Exception:
        error_occur = True

    assert error_occur


def test_fsw_2():
    x = np.random.randn(1024, 23)
    y = np.random.randint(0, 9, 1024)
    sw = signal.FixedSlidingWindow(256, overlap_rate=0.5)
    x, y = sw(x, y)
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)


def test_adjust_sequences_1():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]
    max_len = max([i.shape[0] for i in x])
    new = signal.adjust_sequences(x)
    assert max_len == new.shape[1]


def test_adjust_sequences_2():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]
    min_len = min([i.shape[0] for i in x])
    new = signal.adjust_sequences(x, np.min)
    assert min_len == new.shape[1]


def test_adjust_sequences_3():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]

    new = signal.adjust_sequences(x, 64)
    assert new.shape[1] == 64


def test_adjust_sequences_4():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]

    new = signal.adjust_sequences(x, dtype=np.int64)
    assert new.dtype == np.int64


def test_adjust_sequences_5():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]

    new = signal.adjust_sequences(x, dtype=np.float64)
    assert new.dtype == np.float64


def test_adjust_sequences_6():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]
    try:
        new = signal.adjust_sequences(x, fill=0)
        passed = True
    except Exception:
        passed = False

    assert passed


def test_adjust_sequences_7():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]
    try:
        new = signal.adjust_sequences(x, fill=0.0)
        passed = True
    except Exception:
        passed = False

    assert passed


def test_adjust_sequences_8():
    x = [
        np.random.randn(128, 5),
        np.random.randn(50, 5),
        np.random.randn(100, 5),
    ]
    passed = False
    try:
        new = signal.adjust_sequences(x, fill="Something")
    except TypeError:
        passed = True

    assert passed
