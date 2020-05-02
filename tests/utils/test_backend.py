import numpy as np
import torch
import mxnet.ndarray as nd

from enchanter.utils import backend as bf


def test_slice_axis_0d():
    x = nd.array([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.]
    ]).astype("float32")

    ans = nd.array([
        [5., 6., 7., 8.],
        [9., 10., 11., 12.]
    ])

    mx = nd.slice_axis(x, axis=0, begin=1, end=3)
    enc = bf.slice_axis(torch.from_numpy(x.asnumpy()), axis=0, begin=1, end=3)

    assert np.sum(ans.asnumpy() == enc.numpy()) == 8
    assert np.sum(mx.asnumpy() == enc.numpy()) == 8


def test_slice_axis_1d():
    x = nd.array(
        [[1., 2., 3., 4.],
         [5., 6., 7., 8.],
         [9., 10., 11., 12.]]
    ).astype("float32")
    ans = nd.array([
        [1., 2.],
        [5., 6.],
        [9., 10.]
    ])
    mx = nd.slice_axis(x, axis=1, begin=0, end=2)
    enc = bf.slice_axis(torch.from_numpy(x.asnumpy()), axis=1, begin=0, end=2)
    assert np.sum(ans.asnumpy() == enc.numpy()) == 6
    assert np.sum(mx.asnumpy() == enc.numpy()) == 6


def test_slice_axis_2d():
    x = nd.random.randn(1, 64).astype("float32")
    mx = nd.slice_axis(x, axis=1, begin=0, end=-4).asnumpy().astype("float32")
    enc = bf.slice_axis(torch.from_numpy(x.asnumpy()), axis=1, begin=0, end=-4).numpy().astype("float32")

    assert np.sum(mx == enc) == 60


def test_slice_axis_3d():
    x = nd.random.randn(1, 6, 64).astype("float32")
    mx = nd.slice_axis(x, axis=2, begin=0, end=-4).asnumpy().astype("float32")
    enc = bf.slice_axis(torch.from_numpy(x.asnumpy()), axis=2, begin=0, end=-4).numpy().astype("float32")

    assert np.sum(mx == enc) == 360
