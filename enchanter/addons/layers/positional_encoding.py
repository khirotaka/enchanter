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


__all__ = [
    "PositionalEncoding"
]


class PositionalEncoding(nn.Module):
    """
    Attention is all you need. で提案された　Transformer モデルで用いられる Positional Encoding

    References:
        `Sequence-to-Sequence Modeling with nn.Transformer and TorchText \
        <https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model>`_

    """
    def __init__(self, d_model, seq_len, dropout=0.1):
        """


        Args:
            d_model: the number of expected features in the encoder/decoder inputs.
            seq_len: length of input sequence.
            dropout: dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """

        Args:
            x: (N, E, L)
            N ... batch size
            E ... features
            L ... seq len

        Returns:

        """
        x = x.permute(2, 0, 1)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
