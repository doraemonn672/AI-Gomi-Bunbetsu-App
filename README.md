# AIゴミ分別判定アプリ

## 1. 概要
ゴミの写真を入力し、PyTorchで実装した畳み込みニューラルネットワーク（CNN）で、`cardboard / glass / metal / paper / plastic / trash` の6クラスに分類するWebアプリです。Gradioを利用してブラウザから操作できます。

## 2. 授業との関係
Python、データ分析、ニューラルネットワーク、CNN、画像分類、パラメータ探索、実験設計という授業内容を組み合わせています。

## 3. データセット
TrashNetを利用します。元データは2,527枚、6クラスです。出典：Gary Thung & Mindy Yang, Stanford CS229 project repository.
https://github.com/garythung/trashnet

本プロジェクトでは、データセットを実行時にダウンロードする方式にしています。ZIPそのものは容量とライセンス・再配布上の扱いを考慮して提出物に含めていません。

## 4. セットアップ
Python 3.10以上を推奨。

```bash
pip install -r requirements.txt
```

初回学習：
```bash
python src/train.py --channels 16 --lr 0.001 --epochs 10
```

パラメータ探索：
```bash
python src/experiment.py
```

Webアプリ：
```bash
python src/main.py
```

初回実行時にデータセットが自動ダウンロードされます。通信環境によって時間がかかります。

## 5. 実験
比較例としてCNNの最初の畳み込み層のチャンネル数と学習率を変更します。
- channels: 8 / 16 / 32
- learning rate: 0.001 / 0.01
- train/validation/test: 70% / 15% / 15%
- seed: 42

`results/parameter_search.csv` に実測結果を保存します。発表では、実際に自分で実行した結果を使用してください。

## 6. 注意
このモデルは研究・授業用の画像分類器です。自治体ごとに分別ルールが異なるため、モデルの出力を正式な分別案内として扱わないでください。
