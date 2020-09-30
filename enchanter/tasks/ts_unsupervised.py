from typing import List, Dict, Optional, Union, Tuple

import numpy as np
from comet_ml.experiment import BaseExperiment
import torch
from torch.cuda import amp
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_validate

from enchanter.engine import BaseRunner
from enchanter.addons.criterions.ts_triplet_loss import (
    generate_anchor_positive_input,
    generate_negative_input,
    generate_sample_indices,
    positive_criterion_for_triplet_loss,
    negative_criterion_for_triplet_loss,
    calculate_triplet_loss,
)
from enchanter.callbacks import Callback, BaseLogger
from enchanter.utils.datasets import TimeSeriesUnlabeledDataset


class TimeSeriesUnsupervisedRunner(BaseRunner):
    """
    Runner for unsupervised time series representation learning.

    Unsupervised representation learning for time series uses
    the the Unsupervised Triplet Loss proposed in NeurIPS 2019.

    Paper: `Unsupervised Scalable Representation Learning for Multivariate Time Series \
        <https://papers.nips.cc/paper/8713-unsupervised-scalable-representation-learning-for-multivariate-time-series>`_

    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        experiment: Union[BaseExperiment, BaseLogger],
        n_negative_samples: int = 1,
        negative_penalty: int = 1,
        compared_len: Optional[int] = None,
        evaluator: Union[ClassifierMixin, RegressorMixin] = SVC(),
        save_memory: bool = False,
        scheduler: List = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Initializer

        Args:
            model: PyTorch model which outputting a fixed-length vector regardless of the length of the input series.
            optimizer: PyTorch Optimizer
            experiment: ``comet_ml.BaseExperiment`` or ``enchanter.callbacks.BaseLogger``
            n_negative_samples: Parameter K in the paper. The number of negative samples to be sampled during training.
            negative_penalty: Coefficients that control how much negative values are valued.
            compared_len: Maximum length of randomly chosen time series. (default None).
            save_memory: If True, enables to save GPU memory.
            scheduler: lr scheduler. use ``torch.optim.lr_scheduler``
            callbacks: List of callback.
        """
        super(TimeSeriesUnsupervisedRunner, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.experiment = experiment

        if compared_len is None:
            self.compared_len = np.inf
        else:
            self.compared_len = compared_len

        self.n_negative_samples = n_negative_samples
        self.negative_penalty = negative_penalty
        self.save_memory = save_memory
        self.train_ds = None
        if scheduler is None:
            self.scheduler = list()
        else:
            self.scheduler = scheduler

        self.callbacks = callbacks
        self.evaluator = evaluator

    def initialize(self) -> None:
        super(TimeSeriesUnsupervisedRunner, self).initialize()
        if isinstance(self.loaders["train"].dataset, TimeSeriesUnlabeledDataset):
            self.train_ds = self.loaders["train"].dataset  # type: ignore
        else:
            raise ValueError(
                "You should use the `enchanter.utils.datasets.TimeSeriesUnlabeledDataset` or"
                " `TimeSeriesLabeledDataset` for the dataset class."
            )

    def calculate_negative_loss_per_negative_sample(
        self,
        begin_neg: torch.Tensor,
        len_pos_neg: int,
        batch_size: int,
        negative_sample_step: int,
        anchor_representation: torch.Tensor,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        calculate negative loss per negative sample

        Args:
            begin_neg:
            len_pos_neg:
            batch_size:
            negative_sample_step:
            anchor_representation:
            data:

        Returns:
            negative loss (torch.Tensor)

        """
        representation_size: int = anchor_representation.shape[2]
        negative_data: torch.Tensor = generate_negative_input(
            begin_neg, len_pos_neg, batch_size, negative_sample_step, self.train_ds.data, data  # type: ignore
        ).to(
            self.device
        )  # [batch_size, features, seq_len]

        negative_representation: torch.Tensor = self.model(negative_data).view(batch_size, representation_size, 1)
        negative_loss: torch.Tensor = negative_criterion_for_triplet_loss(
            anchor_representation, negative_representation
        )
        return negative_loss

    def calculate_negative_loss(self, positive_loss, anchor_representation):
        """
        calculate negative loss using all negative samples.

        Args:
            positive_loss:
            anchor_representation:

        Returns:
            loss (torch.Tensor)

        """
        train_size: int = len(self.train_ds)
        multiplicative_ration: float = self.negative_penalty / self.n_negative_samples
        batch_size: int = anchor_representation.shape[0]
        length: int = min(self.compared_len, self.train_ds.data.shape[2])
        samples: torch.Tensor = torch.tensor(
            np.random.choice(train_size, size=(self.n_negative_samples, batch_size)), dtype=torch.long
        )

        _, _, end_pos, len_pos_neg, begin_neg_samples = generate_sample_indices(
            self.n_negative_samples, batch_size, length
        )

        for i in range(self.n_negative_samples):
            negative_loss = self.calculate_negative_loss_per_negative_sample(
                begin_neg_samples, len_pos_neg, batch_size, i, anchor_representation, samples
            )
            positive_loss = calculate_triplet_loss(positive_loss, negative_loss, multiplicative_ration)

            if self.save_memory and i != self.n_negative_samples - 1:
                positive_loss.backward(retain_graph=True)
                positive_loss = torch.tensor(0.0, device=self.device)
                torch.cuda.empty_cache()

        return positive_loss

    def train_step(self, batch) -> Dict[str, torch.Tensor]:
        x_train = batch[0]
        batch_size: int = x_train.shape[0]

        length: int = min(self.compared_len, self.train_ds.data.shape[2])  # type: ignore

        anchor_data, positive_data = generate_anchor_positive_input(
            self.n_negative_samples, batch_size, length, x_train
        )

        with amp.autocast(enabled=isinstance(self.scaler, amp.GradScaler)):
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

    def val_step(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        try:
            x, y = batch
        except ValueError:
            return {}
        else:
            with amp.autocast(enabled=isinstance(self.scaler, amp.GradScaler)):
                encoded = self.model(x)
            return {"encoded": encoded, "targets": y}

    def val_end(self, outputs: List) -> Dict[str, torch.Tensor]:
        """
        Perform cross-validation using the encoded validation data and ``evaluator``.

        Args:
            outputs: outputs of ``self.val_step()``.

        Returns:
            ``fit_time``, ``score_time``, ``test_score``, ``train_score``

        """
        try:
            data = torch.cat([output["encoded"] for output in outputs]).cpu().numpy()
            targets = torch.cat([output["targets"] for output in outputs]).cpu().numpy()
        except KeyError:
            return {}
        else:
            logs = {
                k: v.mean() for k, v in cross_validate(self.evaluator, data, targets, return_train_score=True).items()
            }

        return logs

    def encode(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Output encoded data. The output data has the same data type as the input.

        Args:
            data: data

        Returns:
            encoded data

        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=self.device)
            ndarray = True

        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)
            ndarray = False

        else:
            raise ValueError("Unexpected data type.")

        self.model.eval()
        with torch.no_grad(), amp.autocast(enabled=isinstance(self.scaler, amp.GradScaler)):
            out: torch.Tensor = self.model(data)

        if ndarray:
            out: np.ndarray = out.cpu().numpy()  # type: ignore
        else:
            out: torch.Tensor = out.cpu()  # type: ignore

        return out

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        See Also ``self.encode``

        """
        out = self.encode(x)
        if isinstance(out, torch.Tensor):
            out = out.numpy()

        return out
