Define Custom Runner
=====================

Enchanterでは、 ``enchanter.engine.BaseRunner`` を継承することでカスタムRunnerを定義する事が出来ます。
このチュートリアルでは、分類タスク用のRunnerを定義します。

::

    from comet_ml import Experiment

    import torch
    import torch.nn as nn
    import torch.optim as optim

    import enchanter

実装に最低限必要なライブラリは以上の通りです。
なお、実験的に ``TensorBoard`` をサポートしており、``comet_ml.Experiment`` の代わりに、
``enchanter.callbacks.TensorBoardLogger`` を使うことも可能です。

はじめに、簡単なニューラルネットを定義してみましょう。

::

    model = nn.Sequential(
        nn.Linear(10, 126),
        nn.ReLU(),
        nn.Linear(126, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    optimizer = optim.Adam(model.parameters())


では、さっそくカスタムRunnerを定義していきましょう。

::

    class CustomRunner(enchanter.engine.BaseRunner):
        def __init__(self):
            # REQUIRED
            super().__init__()
            self.model = model
            self.optimizer = optimizer
            self.criterion = nn.CrossEntropyLoss()

            self.experiment = Experiment()

        def train_step(self, batch):
            # REQUIRED
            x, y = batch
            out = self.model(x)
            loss = self.criterion(out, y)

            return {"loss": loss}           # `loss` というキーを持つ辞書を戻り値にする必要があります。


カスタムRunnerを定義する際、``__init__()``, ``train_step()`` を必ず定義する必要があります。
なお、``__init__()`` の中で、``self.model``, ``self.optimizer``, ``self.experiment`` を定義する必要があります。

これで、カスタムRunnerの定義か完了しました。あとは、データセットを用意して実行するだけです。

Enchanterを用いた訓練方法は、scikit-learnスタイルの ``.fit()`` メソッドを使う方法と、
``add_loader()`` メソッドを用いて、``torch.utils.data.DataLoader`` を追加し、 ``.run()`` メソッドで実行する2種類があります。
基本的に後者を用いることをお勧めします。

::

    runner = CustomRunner()
    runner.add_loader("train", train_loader)
    runner.add_loader("test", test_loader)

    runner.train_config(epochs=10)
    runner.run()

これで、ニューラルネットの訓練を行う事が出来ます。
なお、分類タスク用のRunnerは ``enchanter.wrappers.ClassificationRunner`` を利用する事も出来ます。
詳細は :doc:`該当のドキュメント <../source/enchanter.wrappers>` を参照してください。
