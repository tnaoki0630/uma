import requests
from bs4 import BeautifulSoup
import csv
import numpy as np
import re
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from makeCSV import scrape_hitchart, scrape_grades, extract_integer, strtime2fltsec, strdate2intdate, extract_dataset
import argparse
###
def constructLR(X, Y):
    ## 単回帰
    # W_P = Polynomial.fit(X[:,1], Y, 1)
    # print(W_P.convert().coef)
    ## 重回帰
    invXtX = np.linalg.inv(np.dot(X.T, X))
    W = np.dot(np.dot(invXtX, X.T), Y)
    return W
###
def RBFkarnel(X1, X2, diag):
    ## ハイパーパラメータの取得
    theta = args.parameter
    ## RBFカーネルの計算
    diff = np.subtract.outer(X1, X2) # 1入力
    Kmtrx = theta[0]*np.exp(-0.5*(diff/theta[1])**2)
    if diag:
        ## 対角成分に観測ノイズの補正を付与
        shape = np.shape(X1)
        Kmtrx = Kmtrx + theta[2]*np.eye(shape[0])
    return Kmtrx
###
def constructGPR(X, Y):
    K = RBFkarnel(X, X, True)
    invK = np.linalg.inv(K)
    # print(np.dot(K, invK)) # check
    ## Yから平均を引いて正規化
    delY = Y - np.mean(Y)
    iKY = np.dot(invK, delY)
    return invK, iKY
###
def getGPR(X_data, X_hat, invK, iKY):
    k_hat = RBFkarnel(X_data, X_hat, False)
    mu = np.dot(k_hat.T, iKY)
    K_hat = RBFkarnel(X_hat, X_hat, True)
    khTiKkh = np.dot(k_hat.T, np.dot(invK, k_hat))
    sgm = K_hat - khTiKkh
    return mu, sgm
###
def getNN(X):
    Y = X[:,0]
    return Y
###
def show_graph(X, Y, x, y, arr):
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(X, Y, marker='.')
    ax.plot(x, y, 'tab:red')
    ax.set_xlabel(arr[0])
    ax.set_ylabel(arr[1])
    # ax.set_title(arr[3])
    # ax.set_xlim(0, 35)
    # ax.set_ylim(-250, 2000)
    ax.grid()
    plt.show()
###
def showGPR_1D(X, Y, x, y_mean, y_var, arr):
    ## ハイパーパラメータの取得
    theta = args.parameter
    ## 描画
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(X, Y, marker='.')
    ax.plot(x, y_mean, 'tab:red')
    print(y_var)
    ax.fill_between(x, y_mean - 2*y_var, y_mean + 2*y_var, color='lightblue', alpha=0.5, label='Variance Range')
    ax.set_xlabel(arr[0])
    ax.set_ylabel(arr[1])
    ax.set_title(f'theta1 = {theta[0]}, theta2 = {theta[1]}, theta3 = {theta[2]}')
    ax.grid()
    plt.savefig(f'GPR_{theta}.png')

if __name__ == "__main__":
    ## コマンド引数関連
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ## モデル選択
    parser.add_argument("-m", "--model", nargs=1, type=int, choices=range(3), help="MODEL = 0: LR(default)\nMODEL = 1: GPR\nMODEL = 2: NN", default=[0])
    ## データ読み込み方式
    parser.add_argument("-d", "--database", nargs=1, type=int, choices=range(2), help="DATABASE = 0: html(default)\nDATABASE = 1: csv", default=[0])
    ## ハイパーパラメータ
    parser.add_argument("-p", "--parameter", nargs=3, type=float, help="hyper-parameter for RBF karnel", default=[10.0, 100.0, 1.0])
    
    ## 引数の取得
    args = parser.parse_args()

    print("start data collection")
    N = 3 # 学習に使用するテーブルの数
    required_data = 6, 7, 14, 17 # 頭数, 枠番, 距離, タイム
    if args.database[0] == 0:
        ## ヒットチャートのページから馬ごとのURLを取得
        ids = scrape_hitchart("https://db.netkeiba.com/?pid=horse_hitchart")
        if not all(ids):
            print("Error at scrape_hitchart")
            exit()

        ## htmlを解析してデータ取得
        data = extract_dataset(ids[:N-1], required_data)
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
        ### (下みたいな感じでhtmlを解析する時と同じように書けたらめっちゃラク。
        ### extract_datasetはhtml解析も含んでしまってるので、コードの設計を変える必要がある。)
        data = extract_dataset(ids[:N-1], required_data)
        if data:
            X_data = data[0]
            Y_data = data[1]
        else:
            print("Error at extract_dataset")
            exit()
    print("done.")

    ## 回帰モデルの構築
    print("start model construction")
    if args.model[0] == 0:
        ## 線形回帰
        W = constructLR(X_data, Y_data)
    elif args.model[0] == 1:
        ## ガウス過程回帰
        invK, iKY = constructGPR(X_data, Y_data)
    elif args.model[0] == 2:
        ## ニューラルネット
        print("Error:: NN is under development...")
        exit()
    print("done.")

    ## データ生成
    print("start data generation")
    x1_hat = np.linspace(1, 3, 100)
    x2_hat = np.linspace(1500, 2500, 30)
    x3_hat = np.linspace(0, 1, 100)
    x0_hat = np.ones_like(x1_hat)
    # X_hat = np.transpose(np.vstack((x0_hat, x2_hat)))
    # X_hat = np.transpose(np.vstack((x0_hat, x1_hat, x2_hat, x3_hat)))
    if args.model[0] == 0:
        Y_hat = np.dot(X_hat, W)
    elif args.model[0] == 1:
        mu, sgm = getGPR(X_data, x2_hat, invK, iKY)
        ## 元のスケールに戻す
        mu = mu + np.mean(Y_data)
    elif args.model[0] == 2:
        Y_hat = getNN(X_hat)
    print("done.")

    ## 描画
    label = "dist[m]", "time[s]"
    # label = "order/horses[-]", "time[s]"
    # show_graph(X_data[:,1], Y_data, x2_hat, Y_hat, label)
    # show_graph(X_data[:,2], Y_data, x2_hat, Y_hat, label)

    showGPR_1D(X_data, Y_data, x2_hat, mu, np.diagonal(sgm), label)