# uma
### 目的
- 競馬で勝つ
### 方法
- 機械学習でなんとかする
### 議事録
- notion: https://www.notion.so/5f5bc50bac8c4c0f8d03eb1564b5b7f7?pvs=4
### 役に立ちそうなリンク
- 機械学習帳: https://chokkan.github.io/mlnote/index.html
- ネット競馬: https://db.netkeiba.com/?pid=horse_top
- スピード指数: http://www.rightniks.ne.jp/index.php?action=whatspidx_contents&name=sikumi
- git研修: https://mixi-developers.mixi.co.jp/23-technical-training-20a6f610140c
- git4windows: https://prog-8.com/docs/git-env-win
- python早見帳: https://chokkan.github.io/python/index.html
- ガウス過程回帰: https://www.ism.ac.jp/~daichi/lectures/H26-GaussianProcess/gp-lecture2-daichi.pdf
- Pythonによる主成分回帰（PCR）と部分的最小２乗回帰（PLS）: https://www.salesanalytics.co.jp/datascience/datascience118/
- 部分的最小二乗回帰(Partial Least Squares Regression, PLS): https://datachemeng.com/partialleastsquares/
- PLS：https://gochikika.ntt.com/Modeling/multicollinearity_regression.html
- パターン認識と機械学習: https://deeplearning.jp/seminar-2/
### 勉強ノート
- Miro: https://miro.com/welcomeonboard/dXVVeUFrMzFoNW1PQWwyOEVVWjhZaU9pSnlYcFV1RlVpUkNmaWhkOTNxalphcHhKdmxwVTUxSHVHQW9NNm1QbHwzNDU4NzY0NTcwMzgwODY2NTMxfDI=?share_link_id=928920182370
# ディレクトリについて
## src
- コード置き場
## data
- csvなどのデータ置き場
## fig
- 画像置き場

# src について
## 実行方法
- <path>/uma のディレクトリで、`python src/<scriptname>.py`として実行する。
- コマンド引数のオプションは`python src/<scriptname>.py -h`で確認できる。
- オプションを指定するときは`python src/main_v2.py -m 1`とか`python src/plotting.py --parameter 0 1 0`みたいな感じで実行する。
  - オプションの指定がない場合はデフォルト値が入る。
  - 要求される引数の数はオプションによって異なる。
## main_v2.py
- 当選確率や払戻金の期待値を算出するモジュールで、実行すると data/ に三連単の期待値 ordered*.csv や三連複の期待値 unordered*.csv の計算結果が作成される。
- 現在用意されているオプションは下記の通り。
  - -m/--model: 線形回帰やガウス分布など、使用するモデルを選択する。
  - -b/--binomen: 学習に使うデータの種類を選択する。
## makeCSV.py
- netkeiba.com に転がってるテーブルをスクレイピングして data/ に UMA_DB.csv などを出力する。
- 現在用意されているオプションは下記の通り。
  - -t/--type: 作成するデータの種類を選択する。
  - -n/--number: ヒットチャートからスクレイピングする時に使用され、何位までURLを取得するかを指定する。
  - -u/--upperbound: 騎手データをスクレイピングする時に使用され、一人当たりに取得する試合の数の上限を指定する。（1ページ分のテーブルをまとめて取得するので、ここで指定した値は厳密な意味での上限ではない。特に困らないので仕様とする。）
## plotting.py
- スクレイピングしたデータを色んな方法でプロットして、確率分布の形状や変数の相関を調べるための補助モジュール。fig/ に *.png が作成される。
- 現在用意されているオプションは下記の通り。
  - -t/--type: 使用するデータの種類を選択する。
  - -p/--parameter: RBFカーネルに使用するパラメータを指定する。本来は3つのハイパーパラメータを持つカーネルなので引数を3つ指定する仕様としているが、現行のコードでは1Dのノンパラメトリック推定にのみ使っているので第1,第3引数は結果に一切影響しない。
