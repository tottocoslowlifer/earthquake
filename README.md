# GeoSciAI2024リポジトリ
このリポジトリの説明.2025年4月から.

## 概要
地震発生時の地震波形の信号対雑音比（SNR）を向上させる.
具体的には,評価関数
![Eq_CostFunction](https://github.com/user-attachments/assets/4d569abf-9ef1-4e23-8886-f6096b815020)
の最大化を図る.
（各文字の定義は https://sites.google.com/jpgu.org/geosciai2025/%E5%9C%B0%E9%9C%87 を参照.）

## フォルダの説明
### data
使用するデータを格納する.

### notebook
分析に使用したnotebookを格納する.

### scripts
データの整形に使用するファイル.

## 手順
1. 以下のコマンドを入力する.
~~~
pip3 install -r requirements.txt
~~~

2. `data/Learning` 内の各ファイルをダウンロードする.

3. `scripts` 上で以下のコマンドを順に入力する.
~~~
python3 evaluate.py
~~~