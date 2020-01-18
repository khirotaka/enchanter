import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, beta: bool = False):
        super(Swish, self).__init__()
        if beta:
            self.weight = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        else:
            self.weight = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs * torch.sigmoid(self.weight * inputs)
        return out
