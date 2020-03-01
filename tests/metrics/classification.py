import torch
from enchanter.metrics import accuracy


def test_accuracy():
    x = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
    y = torch.tensor([0, 1, 1, 2]).long()

    score = accuracy(x, y)
    assert score == 1.0
