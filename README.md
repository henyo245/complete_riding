# complete_riding

## 概要

`complete_riding` は日本の鉄道駅データを用いて路線グラフを構築し、
キー駅（接続駅・終端駅）に基づく縮約グラフや、完全被覆（完乗）に関わる最短距離計算を行うためのツール群です。

主な処理は Python スクリプトで構成されており、以下の機能を備えます。

- 駅・路線データの前処理（特定事業者の路線抽出など）
- 路線グラフの可視化（`matplotlib`/`networkx`）
- キー駅（分岐・終端）抽出と縮約隣接行列の作成
- 最小完全マッチング等に基づく経路計算（`cpp.py` の CPP 実装を利用）

このリポジトリは実験的な解析や可視化を目的としており、`rsc/` にある CSV データを読み込んで処理します。

## 必要条件

- Python 3.10 以上（PEP 604 の `|` 型注釈等を使用しているため）
- pip でインストール可能な Python パッケージ:
  - pandas
  - numpy
  - networkx
  - matplotlib
  - japanize_matplotlib

依存パッケージは次のようにインストールできます:

```bash
python -m pip install --upgrade pip
pip install pandas numpy networkx matplotlib japanize_matplotlib
```

（必要に応じて `pyproject.toml` の設定に従って仮想環境で作業してください。）

## フォルダ構成（抜粋）

- `src/` : メインのスクリプト群（`generate_map.py`, `line_cpp.py`, `cpp.py`, `visualize.py` 等）
- `rsc/` : 入力となる元データ CSV
- `output/` : スクリプトが出力する CSV（縮約隣接行列やキー駅リストなど）
- `test/` : ユニットテスト

## 使い方

以下は代表的なワークフローです。

1) 路線データの前処理と縮約行列の作成・可視化

`generate_map.py` はデータを読み込み、指定した事業者（company_cd）に対応する駅・接続情報を抽出し、
キー駅の抽出、縮約隣接行列の保存を行います。

デフォルトは JR 北海道（`company_cd=1`）です。会社コードを指定して他社を処理できます。

例: JR 北海道（デフォルト）を実行

```bash
python src/generate_map.py
```

例: 会社コードを指定して実行（例: `--company-cd 2`）

```bash
python src/generate_map.py --company-cd 2
```

特定の路線コードを除外したい場合はカンマ区切りで指定できます。

```bash
python src/generate_map.py --company-cd 1 --exclude-lines 11118,12345
```

実行後、`output/` に次のようなファイルが作られます（プレフィックスは会社や指定により変わります）:

- `{prefix}_adjmatrix.csv` : 縮約隣接行列（行・列ラベルはキー駅の `station_cd`）
- `{prefix}_stations.csv` : キー駅リスト（`station_cd`, `station_name`, `lon`, `lat` など）

2) 縮約行列を用いた最短経路／完乗推定（CPP）

`line_cpp.py` は `generate_map.py` が出力した CSV を読み込み、
`cpp.py` のアルゴリズム（Complete Postman Problem に関連した処理）を実行します。

`--prefix` 引数で読み込むファイルのプレフィックスを指定してください（デフォルトは `company_1_key_stations`）。

例:

```bash
python src/line_cpp.py --prefix company_1_key_stations
```

このスクリプトは最小完全マッチングのペアを標準出力に表示し、可視化を行います（`Visualizer` を利用）。

## テスト

テストは `pytest` を使って実行できます。リポジトリルートで次のコマンドを実行してください。

```bash
pytest -q
```

（テストは小規模で、内部の `CPP` 実装やユーティリティ関数を検証します。）

## 開発メモ / 注意点

- スクリプトは `rsc/` にある CSV のスキーマに依存しています。別のデータセットを使う場合は列名や ID の対応を確認してください。
- `generate_map.py` の `preprocess_company` 関数を利用すれば、任意の `company_cd` に対して同様の前処理が可能です。
- 型注釈（`list[int] | None` など）を使用しているため、Python 3.10 以上を推奨します。

## 追加の作業候補

- 出力ディレクトリやファイル名のさらなるパラメータ化
- 大きなデータセットに対する処理高速化（並列化や外部 DB の利用）
- 可視化結果の画像保存オプション

---

この README に追加してほしい項目や、実行環境の細かい指定（仮想環境の設定例や `pyproject.toml` を使ったインストール手順）があれば教えてください。
