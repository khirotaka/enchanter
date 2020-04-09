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


__all__ = [
    "Swish", "mish", "Mish"
]


class Swish(nn.Module):
    def __init__(self, beta=False):
        nn.Module.__init__(self)
        if beta:
            self.weight = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        else:
            self.weight = 1.0

    def forward(self, inputs):
        out = inputs * torch.sigmoid(self.weight * inputs)
        return out


@torch.jit.script
def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


class Mish(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, inputs):
        return inputs * self.tanh(self.softplus(inputs))
