# mypy: ignore-errors
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


__all__ = [
    "generate_sample_indices",
    "generate_anchor_positive_input",
    "generate_negative_input",
    "positive_criterion_for_triplet_loss",
    "negative_criterion_for_triplet_loss",
    "calculate_triplet_loss",
]


def generate_sample_indices(
    n_rand_samples: int, batch_size: int, length: int
) -> Tuple[np.ndarray, int, np.ndarray, int, torch.Tensor]:
    """
    generate sample indices.

    Args:
        n_rand_samples: the number of negative samples
        batch_size: batch size
        length: length of time series.

    Returns:


    """
    if n_rand_samples > 0 and batch_size > 0 and length > 0:
        len_pos_neg: int = np.random.randint(1, length + 1)

        # anchor
        len_anchor: int = np.random.randint(len_pos_neg, length + 1)  # len of anchors
        begin_batches: np.ndarray = np.random.randint(0, length - len_anchor + 1, size=batch_size)

        begin_pos_samples: np.ndarray = np.random.randint(0, len_anchor - len_pos_neg + 1, size=batch_size)
        begin_pos: np.ndarray = begin_batches + begin_pos_samples

        end_pos: np.ndarray = begin_pos + len_pos_neg

        begin_neg_samples: torch.Tensor = torch.randint(
            0, high=length - len_pos_neg + 1, size=(n_rand_samples, batch_size)
        )

        return begin_batches, len_anchor, end_pos, len_pos_neg, begin_neg_samples

    else:
        raise ValueError("The argument must be greater than or equal to 1.")


def generate_anchor_positive_input(
    n_rand_samples: int, batch_size: int, length: int, original_data: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        n_rand_samples: the number of negative samples
        batch_size: batch size
        length: length of time series
        original_data: training data

    Returns:
        - anchor data (torch.Tensor)
        - positive data (torch.Tensor)

    """
    begin_batches, len_anchor, end_pos, len_pos_neg, _ = generate_sample_indices(n_rand_samples, batch_size, length)

    anchor_data = torch.cat(
        [original_data[j : j + 1, :, begin_batches[j] : begin_batches[j] + len_anchor] for j in range(batch_size)]
    )

    positive_data = torch.cat(
        [original_data[j : j + 1, :, end_pos[j] - len_pos_neg : end_pos[j]] for j in range(batch_size)]
    )

    return anchor_data, positive_data


@torch.jit.script
def generate_negative_input(
    begin_neg_samples: torch.Tensor,
    len_pos_neg: int,
    batch_size: int,
    idx: int,
    train: torch.Tensor,
    samples: torch.Tensor,
) -> torch.Tensor:
    """

    Args:
        begin_neg_samples: Starting points of negative samples
        len_pos_neg: length of negative and postive samples
        batch_size: batch size
        idx: Parameter `k` on the paper.
        train: training dataset
        samples:

    Returns:
        negative_input (torch.Tensor) - [batch_size, features, len_pos_neg]

    Notes:
        When running mypy, ``error: Slice index must be an integer or None`` is detected on lines 97 and 98.
        Ignore it for now.
        Also, I didn't add ``type: ignore`` as a comment because it has a negative impact on ``TorchScript`` execution.

    """
    negative_data = torch.cat(
        [
            train[samples[idx, j] : samples[idx, j] + 1][
                :, :, begin_neg_samples[idx, j] : begin_neg_samples[idx, j] + len_pos_neg
            ]
            for j in range(batch_size)
        ]
    )

    return negative_data


@torch.jit.script
def positive_criterion_for_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    r"""

    .. math::

        positive\ loss =
            -\log\Bigl(\sigma(f(x^{ref}) ^\mathrm{T} f(x^{pos}))\Bigr)

    Args:
        anchor: :math:`f(x^{ref})` ... anchor representation
        positive: :math:`f(x^{pos})` ... positive representation

    Returns:
        positive loss (torch.Tensor)

    """
    positive_loss: torch.Tensor = -torch.mean(F.logsigmoid(torch.bmm(anchor, positive)))
    return positive_loss


@torch.jit.script
def negative_criterion_for_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    r"""

    .. math::

        negative\ loss = - \log\Bigl(\sigma(-f(x^{ref}) ^\mathrm{T}f(x_k^{neg}))\Bigl)

    Args:
        anchor: :math:`f(x^{ref})` ... anchor representation
        positive: :math:`f(x^{neg})` ... negative representation

    Returns:
        negative loss (torch.Tensor)

    """
    negative_loss = -torch.mean(F.logsigmoid(-torch.bmm(anchor, positive)))
    return negative_loss


@torch.jit.script
def calculate_triplet_loss(
    positive_loss: torch.Tensor, negative_loss: torch.Tensor, multiplicative_ration: float
) -> torch.Tensor:
    r"""

    .. math::

        Loss = positive\ loss + α × negative\ loss

    Args:
        positive_loss: output of ``positive_criterion_for_triplet_loss``
        negative_loss: output of ``negative_criterion_for_triplet_loss``
        multiplicative_ration: :math:`α`

    Returns:
        loss (torch.Tensor)

    """
    loss: torch.Tensor = positive_loss + multiplicative_ration * negative_loss

    return loss
