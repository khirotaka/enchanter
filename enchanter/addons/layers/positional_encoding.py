# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import math
import torch
import torch.nn as nn


__all__ = ["PositionalEncoding"]


class PositionalEncoding(nn.Module):
    """
    ``Positional Encoding`` used in the ``Transformer`` model proposed in `Attention is all you need`.

    References:
        `Sequence-to-Sequence Modeling with nn.Transformer and TorchText \
        <https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model>`_

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1) -> None:
        """


        Args:
            d_model: the number of expected features in the encoder/decoder inputs.
            seq_len: length of input sequence.
            dropout: dropout rate.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation

        Args:
            x: input data ``[N, F, L]``

        Returns:

            (N, E, L)

        """
        x = x.permute(2, 0, 1)  # [L, N, E]
        x = x + self.pe[: x.size(0), :]  # type: ignore
        return self.dropout(x).permute(1, 2, 0)
