from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def generate_sample_indices(
    n_rand_samples: int, batch_size: int, length: int
) -> Tuple[np.ndarray, int, np.ndarray, int, np.ndarray]:
    """

    Args:
        n_rand_samples:
        batch_size:
        length:

    Returns:

    """
    len_pos_neg: int = np.random.randint(1, length + 1)

    # anchor
    random_len: int = np.random.randint(len_pos_neg, length + 1)  # len of anchors
    begin_batches: np.ndarray = np.random.randint(0, length - random_len + 1, size=batch_size)

    begin_pos_samples: np.ndarray = np.random.randint(0, random_len - len_pos_neg + 1, size=batch_size)
    begin_pos: np.ndarray = begin_batches + begin_pos_samples

    end_pos: np.ndarray = begin_pos + len_pos_neg

    begin_neg_samples: np.ndarray = np.random.randint(0, length - len_pos_neg + 1, size=(n_rand_samples, batch_size))

    return begin_batches, random_len, end_pos, len_pos_neg, begin_neg_samples


def generate_anchor_positive_input(
    n_rand_samples: int, batch_size: int, length: int, original_data: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        n_rand_samples:
        batch_size:
        length:
        original_data:

    Returns:

    """
    begin_batches, random_len, end_pos, len_pos_neg, begin_neg_samples = generate_sample_indices(
        n_rand_samples, batch_size, length
    )

    anchor_data = torch.cat(
        [original_data[j : j + 1, :, begin_batches[j] : begin_batches[j] + random_len] for j in range(batch_size)]
    )

    positive_data = torch.cat(
        [original_data[j : j + 1, :, end_pos[j] - len_pos_neg : end_pos[j]] for j in range(batch_size)]
    )

    return anchor_data, positive_data


def generate_negative_input(
    begin_neg_samples: np.ndarray,
    len_pos_neg: int,
    batch_size: int,
    idx: int,
    train: torch.Tensor,
    samples: torch.Tensor,
) -> torch.Tensor:
    """

    Args:
        begin_neg_samples:
        len_pos_neg:
        batch_size:
        idx:
        train:
        samples:

    Returns:
        negative_input: (torch.Tensor) - [batch_size, features, len_pos_neg]

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


def positive_criterion_for_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """

    .. math::

        positive loss =
            -\log\Bigl(\sigma\bigl(f(\bm{x}^{ref}) ^\mathrm{T} f(\bm{x}^{pos})\bigr)\Bigr)

    Args:
        anchor:
        positive:

    Returns:

    """
    positive_loss: torch.Tensor = -torch.mean(F.logsigmoid(torch.bmm(anchor, positive)))
    return positive_loss


def negative_criterion_for_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """

    .. math::

        negative loss =
            - \sum^K_{k=1}\log\Bigl(\sigma\bigl(-f(\bm{x}^{ref}) ^\mathrm{T}f(\bm{x}_k^{neg})\bigr)\Bigl)

    Args:
        anchor:
        positive:

    Returns:

    """
    negative_loss = -torch.mean(F.logsigmoid(-torch.bmm(anchor, positive)))
    return negative_loss
