from torch import narrow, Tensor


__all__ = ["slice_axis"]


def slice_axis(data: Tensor, axis: int, begin: int, end: int) -> Tensor:
    """

    Examples:
        >>> import torch
        >>> x = torch.tensor([
        >>>     [  1.,   2.,   3.,   4.],
        >>>     [  5.,   6.,   7.,   8.],
        >>>     [  9.,  10.,  11.,  12.]
        >>> ])
        >>>
        >>> slice_axis(x, axis=0, begin=1, end=3)
        >>> # [[  5.,   6.,   7.,   8.],
        >>> # [  9.,  10.,  11.,  12.]]
        >>>
        >>> slice_axis(x, axis=1, begin=0, end=2)
        >>> # [[  1.,   2.],
        >>> # [  5.,   6.],
        >>> # [  9.,  10.]]
        >>>
        >>> slice_axis(x, axis=1, begin=-3, end=-1)
        >>> # [[  2.,   3.],
        >>> # [  6.,   7.],
        >>> # [ 10.,  11.]]


    References:
        - `Deep Graph Library \
        <https://github.com/dmlc/dgl/blob/f25bc176d0365234ebb051d5069edff24ad2de4d/python/dgl/backend/pytorch/tensor.py#L159-L160>`_

        - `mxnet.ndarray.slice_axis \
        <https://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.slice_axis.html#mxnet-ndarray-slice-axis>`_

    Args:
        data: Source input
        axis: Axis along which to be sliced
        begin: The beginning index along the axis to be sliced
        end: The ending index along the axis to be sliced

    Returns:
        output - the output of this function.

    """
    if begin < 0:
        begin = data.shape[axis] + begin

    if end < 0:
        end = data.shape[axis] + end
    return narrow(data, axis, begin, end - begin)
