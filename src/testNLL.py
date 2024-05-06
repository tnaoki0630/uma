import numpy as np
from main import RBFkarnel, NLL
from makeCSV import scrape_hitchart, extract_dataset
import argparse
import matplotlib.pyplot as plt

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

## initial hyper-parameter
theta_init = np.array(args.parameter)
## check karnel
K = RBFkarnel(theta_init, X_data, X_data, True, False)
plt.imshow(K, cmap='jet')
plt.colorbar()
plt.savefig('karnel.png')
plt.clf()

## theta array
width = [3, 3, 3]
tmin = theta_init - width
tmax = theta_init + width
theta1 = np.linspace(tmin[0], tmax[0], 50)
theta2 = np.linspace(tmin[1], tmax[1], 50)
theta3 = np.linspace(tmin[2], tmax[2], 50)

## theta1-theta2 plot
t3 = theta_init[2]
row = np.shape(theta1)
col = np.shape(theta2)
L = np.zeros((row[0], col[0]))
i = 0
j = 0
for t1 in theta1:
    for t2 in theta2:
        theta = np.array([t1,t2,t3])
        L[i,j] = NLL(theta, X_data, Y_data)
        j += 1
    i += 1
    j = 0
## check NLL
plt.title(f't3 = {t3:.2f}')
plt.xlabel('t1')
plt.ylabel('t2')
plt.imshow(L, cmap='jet', extent=[theta1.min(), theta1.max(), theta2.min(), theta2.max()])
plt.colorbar()
plt.savefig('NLL_t1t2.png')
plt.clf()

## theta1-theta3 plot
t2 = theta_init[1]
row = np.shape(theta1)
col = np.shape(theta3)
L = np.zeros((row[0], col[0]))
i = 0
j = 0
for t1 in theta1:
    for t3 in theta3:
        theta = np.array([t1,t2,t3])
        L[i,j] = NLL(theta, X_data, Y_data)
        j += 1
    i += 1
    j = 0
## check NLL
plt.title(f't2 = {t2:.2f}')
plt.xlabel('t1')
plt.ylabel('t3')
plt.imshow(L, cmap='jet', extent=[theta1.min(), theta1.max(), theta2.min(), theta2.max()])
plt.colorbar()
plt.savefig('NLL_t1t3.png')
plt.clf()

## theta2-theta3 plot
t1 = theta_init[0]
row = np.shape(theta2)
col = np.shape(theta3)
L = np.zeros((row[0], col[0]))
i = 0
j = 0
for t2 in theta2:
    for t3 in theta3:
        theta = np.array([t1,t2,t3])
        L[i,j] = NLL(theta, X_data, Y_data)
        j += 1
    i += 1
    j = 0
## check NLL
plt.title(f't1 = {t1:.2f}')
plt.xlabel('t2')
plt.ylabel('t3')
plt.imshow(L, cmap='jet', extent=[theta1.min(), theta1.max(), theta2.min(), theta2.max()])
plt.colorbar()
plt.savefig('NLL_t2t3.png')
plt.clf()
