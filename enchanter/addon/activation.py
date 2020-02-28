# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import torch
import torch.jit
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, beta: bool = False):
        super().__init__()
        if beta:
            self.weight = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        else:
            self.weight = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs * torch.sigmoid(self.weight * inputs)
        return out


@torch.jit.script
def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


class Mish(nn.Module):
    def forward(self, x):
        return mish(x)
