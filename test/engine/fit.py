import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import enchanter
import enchanter.addon as addon


def main():
    iris = load_iris()
    X, y = iris.data.astype("float32"), iris.target.astype("int64")
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    model = nn.Sequential(
            nn.Linear(4, 512),
            addon.Swish(),
            nn.Linear(512, 256),
            addon.Swish(),
            nn.Linear(256, 3)
        )

    runner = enchanter.ClassificationRunner(model, nn.CrossEntropyLoss(), optim.Adam, {"lr": 0.01})
    runner.fit(
        x_train, y_train, epochs=2, batch_size=16, verbose=True, checkpoint="../data/checkpoint",
        loader_config={"num_workers": 2}
    )

    metrics = runner.evaluate(x_test, y_test, metrics=[r2_score])
    print(metrics)


if __name__ == '__main__':
    main()
