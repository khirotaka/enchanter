Hyper parameters tuning using comet.ml
=======================================

Enchanter は comet.ml と PyTorch を高度に統合することを目的に開発されています。
ここでは、comet.mlが提供するハイパーパラメータチューニングをEnchanterと一緒に使う方法を紹介します。

comet.mlが提供するハイパーパラメータチューニング機能は細やかな設定とログ管理ができる一方、
辞書形式で設定を記述する必要があり、可読性が悪いという問題があります。
Enchanterでは、この設定を容易にするために
Optunaスタイルのconfigジェネレータ ``enchanter.utils.comet.TunerConfigGenerator`` を提供しています。

::

    from comet_ml import Optimizer

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_iris

    import enchanter.tasks as tasks
    import enchanter.addons as addons
    import enchanter.addons.layers as layers
    from enchanter.utils import comet

ライブラリの準備ができました、早速 ``TunerConfigGenerator`` を使ってconfigを生成しましょう。

::

    config = comet.TunerConfigGenerator(
        algorithm="bayes",
        metric="train_avg_loss",
        objective="minimize",
        seed=0,
        trials=5
    )

    config.suggest_categorical("activation", ["addons.mish", "torch.relu", "torch.sigmoid"])

ここでは、モデルに最適な活性化関数を探索することにします。``.suggest_categorical()`` は変数の名前と探索対象を与えます。
comet.mlの仕様上、カテゴリカル変数の場合、探索対象は要素に文字列を持つリストを与える必要があります。

::

    opt = Optimizer(config.generate())

作成したconfigは ``.generate()`` メソッドを使うことで簡単に comet.ml が求める辞書形式にする事が出来ます。
これで準備が整いました、早速探索を行ってみましょう。

::

    for experiment in opt.get_experiments():
        model = layers.MLP([4, 512, 128, 3], eval(experiment.get_parameter("activation")))
        optimizer = optim.Adam(model.parameters())
        runner = tasks.ClassificationRunner(
            model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), experiment=experiment
        )
        x, y = load_iris(return_X_y=True)
        x = x.astype("float32")
        y = y.astype("int64")

        runner.fit(x, y, epochs=1, batch_size=32)

ここで注意が必要なのは、先ほどの ``.suggest_categorical()`` メソッドで指定した変数についてです。
comet.mlでは カテゴリカル変数は文字列として与えられ、文字列を返すという仕組みになっています。
Pythonの関数名を与えた場合、文字列として返されてしまうので、再度関数にするためには、 ``eval()`` を使う必要があります。
