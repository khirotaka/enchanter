:tocdepth: 1

Contribution
=============
ここでは、Enchanterへの開発へ参加、特にコードの変更を伴うプルリクエストの送り方のヒントを提供します。


プルリクエストの送り方
-------------------------

`コントリビューションガイド <https://github.com/khirotaka/enchanter/blob/master/CONTRIBUTING.md>`_ に書かれている通り、
Enchanterへの貢献には2つのカテゴリに分類されます。

1. 新機能の実装
2. 機能の向上、もしくは、バグ修正

どちらのカテゴリも開発を開始する前に、`リポジトリのissues <https://github.com/khirotaka/enchanter/issues>`_ を確認し、
あなたの考えている新機能やバグ修正に類似する投稿が既にあるなら、そちらへ参加してください。
もし、まだ無いようでしたら、 ``New issues`` から新たにissueを開始してください。
また、既存のissue内の説明が十分でない場合は、遠慮せず詳細を尋ねてください。

なお、issues内では英語が推奨されますが、日本語でも問題ありません。


手元でEnchanterをインストールし開発する
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
まず、自分のGitHub アカウント上にEnchanterをforkしましょう。続けて、forkしたリポジトリを自身のPC上にcloneしてください。
大まかな流れとしては次の通りです。

.. code-block:: shell

    $ git clone git@github.com:${YOUR_GITHUB_NAME}/enchanter.git
    $ cd enchanter/
    $ git remote add origin git@github.com:${YOUR_GITHUB_NAME}/enchanter.git
    $ git remote add upstream git@github.com:khirotaka/enchanter.git


Enchanterでは開発のための仮想環境に `Poetry <https://python-poetry.org>`_ を利用することができます。
既にインストール済みなら、以下のコマンドで開発環境を整備することができます。

.. code-block:: shell

    $ poetry develop
    $ poetry shell

これにより、Enchanterを編集可能モードでインストールすることができ、変更内容がすぐ反映されます。

もし、Poetryをインストールしていなくても、``pip install -e .`` を実行することで上と同じ結果を得ることができます。
なお、Enchanterが確実に動作するPythonのバージョンは3.6.5 ~ 3.8です。


それではインストールが成功したら、作業用ブランチを切り実際にコードに変更を加えてみましょう。

.. code-block:: shell

    $ git checkout -b WORKING_BRANCH_NAME

開発用のIDEはお好きなものをご利用ください。


プルリクエストを送る
~~~~~~~~~~~~~~~~~~~~~~~~~
コードに変更を加え、自身のGitHub上にPushした後に、
`オリジナルのEnchanterプロジェクトページ <https://github.com/khirotaka/enchanter>`_ を開くと
``Compare & pull request`` というボタンが出現します。このボタンを押し、プルリクエストを送ってください。

プルリクエスト作成時は次のことに注意してください。

* タイトルは変更を端的に表す一文にする。文末にピリオドは付けない。
* 対応する issue へのリンクをはる。
* 変更内容を詳細に記述する。
* GitHub Actionsでテストが走ります。エラーが出た場合は確認し、修正してください。
