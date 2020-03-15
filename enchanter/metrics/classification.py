import torch


__all__ = [
    "accuracy"
]


def accuracy(inputs, targets):
    """
    バッチ処理向けの精度計算関数。
    分類度を Python float で返します。

    Args:
        inputs (torch.Tensor): shape == [N, n_class]
        targets (torch.Tensor): shape == [N]

    Returns:
        分類制度 (torch.Tensor)
    """
    with torch.no_grad():
        total = targets.shape[0]
        _, predicted = torch.max(inputs, 1)
        correct = (predicted == targets).cpu().sum().float().item()

    return correct / total
