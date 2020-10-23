from typing import Union
from numbers import Number

import torch
import numpy as np

__all__ = ["slice_axis", "is_scalar"]


@torch.jit.script
def slice_axis(data: torch.Tensor, axis: int, begin: int, end: int) -> torch.Tensor:
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
    return torch.narrow(data, axis, begin, end - begin)


def is_scalar(data: Union[Number, Union[np.ndarray, torch.Tensor]]) -> bool:
    """
    Returns True if the type of ``data`` is a scalar type.

    Args:
        data (Union[Number, Union[np.ndarray, torch.Tensor]]): Numerical value

    Returns:
        True if ``data`` is a scalar type, False if it is not.

    Examples:
        >>> a = torch.tensor([1.0])
        >>> is_scalar(a)    # True
        >>> a = torch.tensor(1.0)
        >>> is_scalar(a)    # True
        >>> a = torch.tensor([1, 2, 3])
        >>> is_scalar(a)    # False
        >>> a = 1.0
        >>> is_scalar(a)    # True


    """
    if isinstance(data, Number):
        return True
    else:
        if len(data.shape) == 0:
            return True
        else:
            try:
                _ = data.item()
            except ValueError:
                return False
            else:
                return True
