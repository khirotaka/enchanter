from pytorch_lightning.metrics import functional
from ._general import _GeneralModule


__all__ = ["RegressionTask"]


class RegressionTask(_GeneralModule):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        r2 = functional.r2score(y_hat, y)
        mae = functional.mean_absolute_error(y_hat, y)
        mse = functional.mean_squared_error(y_hat, y)

        metrics = {
            "val_r2": r2,
            "val_mea": mae,
            "val_mse": mse,
            "val_loss": loss,
        }

        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {
            "test_r2": metrics["val_r2"],
            "test_mea": metrics["val_mea"],
            "test_mse": metrics["val_mse"],
            "test_loss": metrics["val_loss"],
        }
        self.log_dict(metrics)
