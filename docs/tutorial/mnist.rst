Training neural network using MNIST
====================================

機械学習分野のHello, World、MNISTを使ってEnchanterの使い方を学びましょう。
まず最初に今回必要なモジュールをインポートします。
ここで注意必要なのが、一番最初に ``comet.ml`` をインポートする必要がある点です。
これはcometの仕様のためです。

::

    from comet_ml import Experiment

    import torch.nn as nn
    import torch.optim as optim
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader

    import enchanter.tasks as tasks
    import enchanter.addons as addons
    from enchanter.callbacks import EarlyStopping


では次に、今回使う畳込みニューラルネット(CNN)を定義していきます。

::

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                addons.Mish(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3),
                addons.Mish(),
                nn.MaxPool2d(2)
            )
            self.fc = nn.Sequential(
                nn.Linear(64*5*5, 512),
                addons.Mish(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            out = self.conv(x)
            out = out.view(-1, 64*5*5)
            out = self.fc(out)
            return out

これでCNNの定義が終わりました。
次にデータセットの読み込みとログ取りの準備をします。

::

    experiment = Experiment()

    train_ds = MNIST("../data/", download=True, train=True, transform=transforms.ToTensor())
    test_ds = MNIST("../data/", download=True, train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


最後に、モデルの定義と学習を行います。

::

    model = CNN()
    optimizer = optim.Adam(model.parameters())
    runner = tasks.ClassificationRunner(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        experiment,
        scheduler=[optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)],
        early_stop=EarlyStopping("train_avg_loss", min_delta=0.1, patience=1)
    )

Enchanterでは、``runner`` に使用目的に合わせたDataLoaderを追加していくという方式をとっています。
また、Epochの回数は ``.train_config()`` メドッドで指定します。
最後に ``.run()`` メソッドでRunnerを実行します。

::

    runner.add_loader("train", train_loader).add_loader("test", test_loader)
    runner.train_config(epochs=5)
    runner.run(verbose=True)
