from typing import Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


__all__ = ["TimeSeriesLabeledDataset", "TimeSeriesUnlabeledDataset"]


class TimeSeriesUnlabeledDataset(Dataset):
    """
    Examples:
        >>> from torch.utils.data import DataLoader
        >>> ds = TimeSeriesUnlabeledDataset(data=...)
        >>> loader = DataLoader(ds)
        >>> data = next(iter(loader))
    """

    def __init__(
        self, data: Union[torch.Tensor, np.ndarray], transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        self.data: torch.Tensor = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        data = self.data[item]
        if self.transform:
            data = self.transform(data)

        return data


class TimeSeriesLabeledDataset(TimeSeriesUnlabeledDataset):
    """
    Examples:
        >>> from torch.utils.data import DataLoader
        >>> ds = TimeSeriesLabeledDataset(data=..., targets=...)
        >>> loader = DataLoader(ds)
        >>> data, targets = next(iter(loader))
    """

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super(TimeSeriesLabeledDataset, self).__init__(data, transform)
        self.targets = targets

    def __getitem__(self, item):
        data = super(TimeSeriesLabeledDataset, self).__getitem__(item)
        target = self.targets[item]
        return data, target
