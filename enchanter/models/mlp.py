import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, shapes, activation=torch.relu):
        super().__init__()
        self.layers = []
        self.activation = activation

        for i in range(len(shapes) - 1):
            self.layers.append(nn.Linear(shapes[i], shapes[i+1]))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return x
