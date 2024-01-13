import numpy as np
from makeCSV import scrape_hitchart, extract_dataset
import argparse
import matplotlib.pyplot as plt

## test for cdist
# # 例として行列Xと行列Yを作成する
# X = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])

# Y = np.array([[12, 11, 10],
#               [9, 8, 7],
#               [6, 5, 4],
#               [3, 2, 1]])

# # cdist関数を使用して行列Xと行列Yの各行ベクトル間のユークリッド距離を計算
# distances = cdist(X, Y, 'euclidean')

# # 結果を表示
# print(distances)

## test for NLL
## コマンド引数関連
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
## データ読み込み方式
parser.add_argument("-d", "--database", nargs=1, type=int, choices=range(2), help="DATABASE = 0: html(default)\nDATABASE = 1: csv", default=[0])
## 学習に使用するテーブル数
parser.add_argument("-n", "--number", nargs=1, type=int, help="NUMBER \\in Z (default: 3)", default=[3])
## モデル選択
parser.add_argument("-m", "--model", nargs=1, type=int, choices=range(3), help="MODEL = 0: LR\nMODEL = 1: GPR(defoult)\nMODEL = 2: NN", default=[1])
## ハイパーパラメータ
parser.add_argument("-p", "--parameter", nargs=3, type=float, help="PARAMETER[0] \\in R (default: 1.0)\nPARAMETER[1] \\in R (default: 20.0)\nPARAMETER[2] \\in R (default: 10.0)", default=[1.0, 20.0, 10.0])

## 引数の取得
args = parser.parse_args()

## データ収集
print("start data collection")
N = min(100, max(1, args.number[0])) # 1~100となるよう例外処理
required_data = 0, 6, 8, 13, 14, 17, 23, 4, 3
if args.database[0] == 0:
    ## ヒットチャートのページから馬ごとのURLを取得
    ids = scrape_hitchart("https://db.netkeiba.com/?pid=horse_hitchart")
    if not ids:
        print("Error at scrape_hitchart")
        exit()

    ## htmlを解析してデータ取得
    data = extract_dataset(ids[:N], required_data)
    if data:
        X_data = data[0]
        Y_data = data[1]
        mean_data = data[2]
        var_data = data[3]
    else:
        print("Error at extract_dataset")
        exit()
elif args.database[0] == 1:
    ## 未実装なのでexit
    print("Error:: csv reading is under development...")
    exit()
    ## 事前取得したcsvからデータ取得
    data = extract_dataset(ids[:N], required_data)
    if data:
        X_data = data[0]
        Y_data = data[1]
    else:
        print("Error at extract_dataset")
        exit()
print('done.')
