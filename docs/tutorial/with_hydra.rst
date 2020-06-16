Enchanter with Hydra
====================

このチュートリアルでは、Hydraを活用したハイパーパラメータ管理について紹介します。

`Hydra <https://hydra.cc>`_ は、Facebook Research が開発を行っている設定管理ツールです。
はじめに、 ``Hydra`` のインストールをしましょう。

.. code-block:: sh

    pip install hydra-core


次に設定ファイルを作成します。以下のような構造のディレクトリ ``config`` を作成してください。

.. code-block:: sh

    examples/
        ├── config/
        │     ├── config.yaml
        │     ├── model/
        │     │     ├── mlp1.yaml
        │     │     └── mlp2.yaml
        │     └── optimizer/
        │           └── adam.yaml
        └── with_hydra.py

それぞれのファイルの中身は以下のようにして下さい。

.. code-block:: yaml

    # config/config.yaml
    defaults:
      - model: mlp
      - optimizer: adam

.. code-block:: yaml

    # config/model/mlp1.yaml
    model:
      shapes:
        - 4
        - 512
        - 128
        - 3

.. code-block:: yaml

    # config/model/mlp2.yaml
    model:
      shapes:
        - 4
        - 16
        - 32
        - 6
        - 3

.. code-block:: yaml

    # config/optimizer/adam.yaml
    optimizer:
      params:
        lr: 0.001


これで設定ファイルの準備は完了です。次に、Irisデータセットを用いた実験を行いましょう。
``with_hydra.py`` という名称で、上のディレクトリ構造になるように保存して下さい。

::

    import hydra
    from comet_ml import Experiment

    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_iris

    import enchanter.wrappers as wrappers
    import enchanter.addons.layers as layers


    experiment = Experiment()
    x, y = load_iris(return_X_y=True)
    x = x.astype("float32")
    y = y.astype("int64")

必要なライブラリとデータセットの準備が完了しました。次に ``main()`` 関数の実装を行います。
``Hydra`` を用いる場合は、引数 ``cfg`` を持つ関数に ``@hydra.main`` デコレータを付けるだけで ``config`` を読み込ませる事が出来ます。

::

    @hydra.main("config/config.yaml")
    def main(cfg):
        shapes = cfg.model.shapes
        opt_params = cfg.optimizer.params

        experiment.add_tag("with_hydra")
        model = layers.MLP(shapes)
        optimizer = optim.Adam(model.parameters(), **opt_params)
        runner = wrappers.ClassificationRunner(
            model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            experiment=experiment
        )
        runner.train_config(epochs=10, checkpoint_path="./checkpoints")
        runner.fit(x, y)
        runner.save()


    if __name__ == '__main__':
        main()


これで準備完了です。
ターミナルで、

.. code-block:: sh

    $ python with_hydra.py

を実行すれば、同じディレクトリに実行結果を格納した ``outputs`` ディレクトリが生成されます。
今回の設定では、同ディレクトリ内に ``checkpoints`` ディレクトリが生成され、各エポック毎の重みが保存されたファイルが生成されているはずです。

では次に、Optimizerの学習率を変更してみましょう。実行時以下のように引数を与えてみて下さい。

.. code-block:: sh

    $ python with_hydra.py optimizer.params.lr=0.1

さらに、別の設定ファイルを使ってモデルを書き換える方法を試してみましょう。
これで、Optimizerの学習率を事前に設定していた ``0.001`` から ``0.1`` に書き換える事が出来ます。

また、読み込む設定ファイルを別の物に変子する事も出来ます。
試しに、事前に作成した、 ``config/model/mlp2.yaml`` を元に新しいモデルを作成、実験を行ってみましょう。


.. code-block:: sh

    $ python with_hydra.py model=mlp2

これで、実行されるモデルの構造は ``config/model/mlp2.yaml`` に記載されている構造が採用されます。
