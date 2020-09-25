# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import torch


__all__ = ["calculate_accuracy"]


def calculate_accuracy(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    A function that calculates accuracy for batch processing.
    Returns accuracy as a Python float.

    Args:
        inputs (torch.Tensor): shape == [N, n_class]
        targets (torch.Tensor): shape == [N]

    Returns: accracy (float)
    """
    with torch.no_grad():
        total = targets.shape[0]
        _, predicted = torch.max(inputs, 1)
        correct = (predicted == targets).cpu().sum().float().item()

    return correct / total
