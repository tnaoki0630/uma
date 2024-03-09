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

varlist = ['ID', '頭 数', '馬 番', 'オ ッ ズ', '着 順', '斤 量', '上り', '馬体重']
csvpath = "UMA_DB.csv"
df_all = pd.read_csv(csvpath)
df = df_all[varlist]
print(df)