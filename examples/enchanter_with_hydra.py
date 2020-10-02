import hydra
from comet_ml import Experiment

import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris

import enchanter.tasks as tasks
import enchanter.addons.layers as layers


x, y = load_iris(return_X_y=True)
x = x.astype("float32")
y = y.astype("int64")


@hydra.main("config", "config.yaml")
def main(cfg):
    shapes = cfg.model.shapes
    opt_params = cfg.optimizer.params

    experiment = Experiment()
    experiment.add_tag("with_hydra")
    model = layers.MLP(shapes)
    optimizer = optim.Adam(model.parameters(), **opt_params)
    runner = tasks.ClassificationRunner(
        model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        experiment=experiment
    )
    runner.fit(x, y, epochs=10, checkpoint_path="./checkpoints")
    runner.save()


if __name__ == '__main__':
    main()
