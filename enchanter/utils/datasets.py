from typing import Callable, Optional

import torch
from torch.utils.data import Dataset


class TimeSeriesUnlabeledDataset(Dataset):
    def __init__(self, data: torch.Tensor, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
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
    def __init__(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super(TimeSeriesLabeledDataset, self).__init__(data, transform)
        self.target = target

    def __getitem__(self, item):
        data = super(TimeSeriesLabeledDataset, self).__getitem__(item)
        target = self.target[item]
        return data, target
