# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import torch


__all__ = [
    "calculate_accuracy"
]


def calculate_accuracy(inputs, targets):
    """
    バッチ処理向けの精度計算関数。
    分類度を Python float で返します。

    Args:
        inputs (torch.Tensor): shape == [N, n_class]
        targets (torch.Tensor): shape == [N]

    Returns:
        分類精度 (float)
    """
    with torch.no_grad():
        total = targets.shape[0]
        _, predicted = torch.max(inputs, 1)
        correct = (predicted == targets).cpu().sum().float().item()

    return correct / total
