from pytorch_lightning.metrics import functional
from ._general import _GeneralModule


__all__ = ["ClassificationTask"]


class ClassificationTask(_GeneralModule):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        accuracy = functional.accuracy(y_hat, y)
        precision, recall = functional.precision_recall(y_hat, y)

        metrics = {
            "val_acc": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_loss": loss,
        }

        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {
            "test_acc": metrics["val_add"],
            "test_precision": metrics["val_precision"],
            "test_recall": metrics["val_recall"],
            "test_loss": metrics["val_loss"],
        }
        self.log_dict(metrics)
