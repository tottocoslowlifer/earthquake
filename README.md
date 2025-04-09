# GeoSciAI2025リポジトリ
このリポジトリの説明.2025年4月から.

## 概要
地震発生時の地震波形の信号対雑音比（SNR）を向上させる.
具体的には,評価関数
![Eq_CostFunction](https://github.com/user-attachments/assets/4d569abf-9ef1-4e23-8886-f6096b815020)
の最大化を図る.
（各文字の定義は https://sites.google.com/jpgu.org/geosciai2025/%E5%9C%B0%E9%9C%87 を参照.）

## フォルダの説明
### data
使用する学習用・検証用データ,および実験結果データを格納する.

### notebook
分析に使用したipynbファイルを格納する.

### scripts
作成したモデル・評価用の関数が記述されたファイルを格納する.

## 手順
1. 以下のコマンドを入力する.
~~~
pip3 install -r requirements.txt
~~~

2. `data/Learning` 内の各ファイルをダウンロードする.

3. `notebook/experiment.ipynb` 内のセルを順に実行する.