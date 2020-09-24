from typing import List, Dict, Optional
import numpy as np
import torch
from enchanter.engine import BaseRunner
from enchanter.addons.ts_triplet_loss import (
    generate_anchor_positive_input,
    generate_negative_input,
    generate_sample_indices,
    positive_criterion_for_triplet_loss,
    negative_criterion_for_triplet_loss,
)
from enchanter.callbacks import Callback


class TimeSeriesUnsupervisedRunner(BaseRunner):
    def __init__(
        self,
        model,
        optimizer,
        experiment,
        n_rand_samples: int,
        negative_penalty: int,
        compared_len=None,
        save_memory: bool = False,
        scheduler: List = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        super(TimeSeriesUnsupervisedRunner, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.experiment = experiment

        if compared_len is None:
            self.compared_len = np.inf
        else:
            self.compared_len = compared_len

        self.n_rand_samples = n_rand_samples
        self.negative_penalty = negative_penalty
        self.save_memory = save_memory
        self.train_ds = None
        if scheduler is None:
            self.scheduler = list()
        else:
            self.scheduler = scheduler

        self.callbacks = callbacks

    def initialize(self) -> None:
        super(TimeSeriesUnsupervisedRunner, self).initialize()
        self.train_ds = self.loaders["train"].dataset

    def calculate_negative_loss(self, positive_loss, anchor_representation):
        train_size: int = len(self.train_ds)
        representation_size: int = anchor_representation.shape[2]
        multiplicative_ration: float = self.negative_penalty / self.n_rand_samples
        batch_size: int = anchor_representation.shape[0]
        length: int = min(self.compared_len, self.train_ds.tensors[0].shape[2])
        samples: torch.Tensor = torch.tensor(
            np.random.choice(train_size, size=(self.n_rand_samples, batch_size)), dtype=torch.long
        )

        _, _, end_pos, len_pos_neg, begin_neg_samples = generate_sample_indices(self.n_rand_samples, batch_size, length)

        for i in range(self.n_rand_samples):
            negative_data: torch.Tensor = generate_negative_input(
                begin_neg_samples, len_pos_neg, batch_size, i, self.train_ds.tensors[0], samples
            ).to(
                self.device
            )  # [batch_size, features, seq_len]

            negative_representation: torch.Tensor = self.model(negative_data).view(batch_size, representation_size, 1)
            negative_loss: torch.Tensor = negative_criterion_for_triplet_loss(
                anchor_representation, negative_representation
            )

            positive_loss: torch.Tensor = positive_loss + multiplicative_ration * negative_loss

            if self.save_memory and i != self.n_rand_samples - 1:
                positive_loss.backward(retain_graph=True)
                positive_loss = torch.tensor(0.0, device=self.device)
                del negative_representation
                torch.cuda.empty_cache()

        return positive_loss

    def train_step(self, batch) -> Dict[str, torch.Tensor]:
        x_train = batch[0]
        batch_size: int = x_train.shape[0]

        length: int = min(self.compared_len, self.train_ds.tensors[0].shape[2])

        anchor_data, positive_data = generate_anchor_positive_input(self.n_rand_samples, batch_size, length, x_train)

        anchor_representation: torch.Tensor = self.model(anchor_data)
        positive_representation: torch.Tensor = self.model(positive_data)

        representation_size: int = anchor_representation.shape[1]

        anchor_representation = anchor_representation.view(batch_size, 1, representation_size)
        positive_representation = positive_representation.view(batch_size, representation_size, 1)

        loss = positive_criterion_for_triplet_loss(anchor_representation, positive_representation)

        if self.save_memory:
            loss.backward(retain_graph=True)
            loss = torch.tensor(0.0, device=self.device)
            del positive_representation
            torch.cuda.empty_cache()

        loss = self.calculate_negative_loss(loss, anchor_representation)

        return {"loss": loss}

    def train_end(self, outputs: List) -> Dict[str, torch.Tensor]:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {"avg_loss": avg_loss}

    def encode(self, data: np.ndarray) -> np.ndarray:
        data = torch.tensor(data, device=self.device)
        out = self.model(data).cpu().numpy()
        return out
