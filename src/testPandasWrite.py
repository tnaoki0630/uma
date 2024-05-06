import pandas as pd
from pandas import plotting
from makeCSV import scrape_hitchart
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

## matplotlib を日本語フォントに変更
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

## コマンド引数関連
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
## 学習に使用するテーブル数
parser.add_argument("-n", "--number", nargs=1, type=int, help="NUMBER \\in Z (default: 3)", default=[3])
## 引数の取得
args = parser.parse_args()

## ヒットチャートのページから馬ごとのURLを取得
ids = scrape_hitchart("https://db.netkeiba.com/?pid=horse_hitchart")
if not ids:
    print("Error at scrape_hitchart")
    exit()

# 変数リスト
varlist = ['頭 数', '馬 番', 'オ ッ ズ', '着 順', '斤 量', '上り', '馬体重']
varlist_wID = varlist[:]
varlist_wID.insert(0, 'ID')

## 1つめはヘッダー込みでデータフレームを作成
url = f"https://db.netkeiba.com/horse/result/{ids[0]}"
# pandasでスクレイピング
df_all = pd.read_html(url, encoding='euc-jp')[0]
df = df_all[varlist]
# うまIDを追加
df_all['ID'] = ids[0]
df['ID'] = ids[0]
# 要素を入れ替え
df = df[varlist_wID]

N = min(100, max(1, args.number[0])) # 1~100となるよう例外処理
for id in ids[1:N]:
    url = f"https://db.netkeiba.com/horse/result/{id}"
    # pandasでスクレイピング
    try:
        df1_all = pd.read_html(url, encoding='euc-jp')[0]
    except ValueError:
        print("Table not found")
    df1 = df1_all[varlist]
    # うまIDを追加
    df1_all['ID'] = id
    df1['ID'] = id
    # 要素を入れ替え
    df1 = df1[varlist_wID]
    # データフレームを結合
    df_all = pd.concat([df_all, df1_all], ignore_index=True)
    df = pd.concat([df, df1], ignore_index=True)


# 取得したデータフレームの出力
df_all.to_csv('output_file.csv', index=False)
# 型チェック
# print(df.dtypes)
# df['着 順'] = pd.to_numeric(df['着 順'], errors='coerce') # なぜか着順が object なので数値に変換
df = df.apply(pd.to_numeric, errors='coerce') # すべての列を数値に変換（失敗した場合はNaNが入る）
df = df.dropna() # NaNを削除

# 散布図行列のプロット
plotting.scatter_matrix(df.iloc[:,1:], figsize=(10, 10), alpha=0.5)
plt.savefig('ScatterMatrix.png')
plt.clf()
