import torch
import torch.jit
import torch.nn as nn
import torch.optim as optim
from enchanter.addon.optim_wrapper import TransformerOptimizer


def main():
    x = torch.randn(64, 32)
    y = torch.randint(0, 10, [64, ])

    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    )

    optimizer = TransformerOptimizer(optim.Adam(model.parameters()), 16, 10)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    main()
