import torch
from enchanter.metrics import calculate_accuracy


def test_accuracy():
    x = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
    y = torch.tensor([0, 1, 1, 2]).long()

    score = calculate_accuracy(x, y)
    assert score == 0.75
