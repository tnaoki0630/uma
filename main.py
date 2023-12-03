import requests
from bs4 import BeautifulSoup
import csv
import numpy as np
import re
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from makeCSV import scrape_hitchart, scrape_grades, extract_integer, strtime2fltsec, strdate2intdate, extract_dataset
import argparse
def constructLR(X, Y):
    ## 単回帰
    # W_P = Polynomial.fit(X_data[:,1], Y_data, 1)
    # print(W_P.convert().coef)
    ## 重回帰
    invXtX = np.linalg.inv(np.dot(X_data.T, X_data))
    W = np.dot(np.dot(invXtX, X_data.T), Y_data)
    return W
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
def getGPR(X):
    Y = X[:,0]
    return Y
###
def getNN(X):
    Y = X[:,0]
    return Y

if __name__ == "__main__":
    ## コマンド引数を取得
    ### モデル選択
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", nargs=1, type=int, choices=range(3), help="MODEL = 0: LR(default)\nMODEL = 1: GPR\nMODEL = 2: NN", default=0)
    args = parser.parse_args()

    print("start data collection")
    ## ヒットチャートのページから馬ごとのURLを取得
    ids = scrape_hitchart("https://db.netkeiba.com/?pid=horse_hitchart")
    if not all(ids):
        print("Error at scrape_hitchart")
        exit()

    ## N 位まで学習に使用
    N = 3
    required_data = 6, 7, 14, 17 # 頭数, 枠番, 距離, タイム
    data = extract_dataset(ids[:N-1], required_data)
    if data:
        X_data = data[0]
        Y_data = data[1]
    else:
        print("Error at extract_dataset")
        exit()
    print("done.")

    ## 回帰モデルの構築
    if args.model[0] == 0:
        ## 線形回帰
        print("start model construction for LR")
        W = constructLR(X_data, Y_data)
    elif args.model[0] == 1:
        ## ガウス過程回帰
        print("start model construction for GPR")
        print("Error:: GPR is under development...")
        exit()
    elif args.model[0] == 2:
        ## ニューラルネット
        print("start model construction for NN")
        print("Error:: NN is under development...")
        exit()
    print("done.")

    ## データ生成
    print("start data generation")
    x1_hat = np.linspace(1, 3, 100)
    x2_hat = np.linspace(1000, 2500, 100)
    x3_hat = np.linspace(0, 1, 100)
    x0_hat = np.ones_like(x1_hat)
    X_hat = np.transpose(np.vstack((x0_hat, x2_hat)))
    # X_hat = np.transpose(np.vstack((x0_hat, x1_hat, x2_hat, x3_hat)))
    if args.model[0] == 0:
        Y_hat = np.dot(X_hat, W)
    elif args.model[0] == 1:
        Y_hat = getGPR(X_hat)
    elif args.model[0] == 2:
        Y_hat = getNN(X_hat)
    print("done.")

    ## 描画
    label = "dist[m]", "time[s]"
    # label = "order/horses[-]", "time[s]"
    show_graph(X_data[:,1], Y_data, x2_hat, Y_hat, label)
    # show_graph(X_data[:,2], Y_data, x2_hat, Y_hat, label)