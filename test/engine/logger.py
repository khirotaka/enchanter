import comet_ml
import torch
from enchanter.engine.modules import CometLogger


def main():
    logger = CometLogger(comet_ml.Experiment(project_name="testflight"))

    for epoch in range(10):
        for step in range(20):
            loss = (epoch + 1) / (step + 1)
            accuracy = loss * 100
            logger.log_train(epoch, step, {"loss": torch.tensor(loss), "accuracy": torch.tensor(accuracy)})

        for val in range(20):
            loss_v = (epoch + 1) / (val + 2)
            accuracy_v = loss_v * 90
            logger.log_val(epoch, val, {"loss": torch.tensor(loss_v), "accuracy": torch.tensor(accuracy_v)})

    for test in reversed(range(10)):
        loss = (test + 1) / 10
        accuracy = loss * 80
        logger.log_test({"loss": torch.tensor(loss), "accuracy": torch.tensor(accuracy)})


if __name__ == '__main__':
    main()
