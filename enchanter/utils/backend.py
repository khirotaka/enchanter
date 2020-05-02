import torch


def slice_axis(data, axis, begin, end):
    """

    References: <Deep Graph Library https://github.com/dmlc/dgl/blob/f25bc176d0365234ebb051d5069edff24ad2de4d/python/dgl/backend/pytorch/tensor.py#L159-L160>_

    Args:
        data: Source input
        axis: Axis along which to be sliced
        begin: The beginning index along the axis to be sliced
        end: The ending index along the axis to be sliced

    Returns:

    """
    if end < 0:
        end = data.shape[axis] + end
    return torch.narrow(data, axis, begin, end - begin)
