from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import torch
from torch.cuda import amp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from enchanter.engine import modules
from enchanter.engine import BaseRunner
from enchanter.engine.typehint import ScikitModel
from enchanter.addons.criterions.ts_triplet_loss import (
    generate_anchor_positive_input,
    generate_negative_input,
    generate_sample_indices,
    positive_criterion_for_triplet_loss,
    negative_criterion_for_triplet_loss,
    calculate_triplet_loss,
)
from enchanter.callbacks import Callback
from enchanter.utils.datasets import TimeSeriesUnlabeledDataset


class TimeSeriesUnsupervisedRunner(BaseRunner):
    """
    Runner for unsupervised time series representation learning.

    Unsupervised representation learning for time series uses
    the the Unsupervised Triplet Loss proposed in NeurIPS 2019.

    Paper: `Unsupervised Scalable Representation Learning for Multivariate Time Series \
        <https://papers.nips.cc/paper/8713-unsupervised-scalable-representation-learning-for-multivariate-time-series>`_


    Examples:
        >>> experiment = ...
        >>> model: torch.nn.Module = ...
        >>> optimizer: torch.optim.Optimizer = ...
        >>> runner = TimeSeriesUnsupervisedRunner(model, optimizer, experiment)
        >>> runner.add_loader("train", ...)
        >>> runner.train_config(...)
        >>> runner.run()

    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        experiment,
        n_negative_samples: int = 1,
        negative_penalty: int = 1,
        compared_len: Optional[int] = None,
        evaluator: ScikitModel = SVC(),
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
        self.evaluator_params = None

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

        _, _, _, len_pos_neg, begin_neg_samples = generate_sample_indices(self.n_negative_samples, batch_size, length)

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
        if len(batch) == 2:
            x_train, _ = batch
        else:
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
        return self.train_step(batch)

    def val_end(self, outputs: List) -> Dict[str, torch.Tensor]:
        return self.train_end(outputs)

    def test_step(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        x, y = batch
        with amp.autocast(enabled=isinstance(self.scaler, amp.GradScaler)):
            encoded = self.model(x)
        return {"encoded": encoded, "targets": y}

    def _generate_train_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        features = []
        targets = []

        loader = self.loaders["train"]
        loader = modules.tfds_to_numpy(loader) if modules.is_tfds(loader) else loader

        self.model.eval()
        with torch.no_grad(), amp.autocast(enabled=isinstance(self.scaler, amp.GradScaler)):
            for batch in loader:
                x, y = batch
                features.append(self.model(x.to(self.device)))
                targets.append(y)

        return torch.cat(features).cpu().numpy(), torch.cat(targets).cpu().numpy()

    def test_end(self, outputs: List) -> Dict[str, torch.Tensor]:
        x_train, y_train = self._generate_train_features()

        if "grid_search" in self.manager.params.keys():
            self.evaluator.set_params(**self.manager.params["grid_search"])
            if len(y_train) <= 10000:
                self.evaluator.fit(x_train, y_train)  # type: ignore
            else:
                split = train_test_split(x_train, y_train, train_size=10000, random_state=0, stratify=y_train)
                self.evaluator.fit(split[0], split[2])  # type: ignore

            # self.experiment.log_parameters(search.best_params_, prefix="grid_search_best_params")
        else:
            self.evaluator.fit(x_train, y_train)  # type: ignore

        x_test = torch.cat([output["encoded"] for output in outputs]).cpu().numpy()
        y_test = torch.cat([output["targets"] for output in outputs]).cpu().numpy()

        return {"evaluator_score": torch.tensor(self.evaluator.score(x_test, y_test))}

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
            out = out.cpu().numpy()  # type: ignore

        return out

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        See Also ``self.encode``

        """
        out = self.encode(x)
        if isinstance(out, torch.Tensor):
            out = out.numpy()

        return out
