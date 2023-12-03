import requests
from bs4 import BeautifulSoup
import csv
import numpy as np
import re
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from makeCSV import scrape_hitchart, scrape_grades, extract_integer, strtime2fltsec, strdate2intdate, extract_dataset

def learn_linear(X, Y):
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

if __name__ == "__main__":
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

    ## 回帰モデルの構築
    # hat_Y = Regression(hat_X) として使いたい。ここの書き方は回帰モデル依存？
    ## 線形回帰
    W = learn_linear(X_data, Y_data)
    ## ガウス過程回帰

    ## ニューラルネット

    ## 描画
    x1_hat = np.linspace(1, 3, 100)
    x2_hat = np.linspace(1000, 2500, 100)
    x3_hat = np.linspace(0, 1, 100)
    x0_hat = np.ones_like(x1_hat)
    X_hat = np.transpose(np.vstack((x0_hat, x2_hat)))
    # X_hat = np.transpose(np.vstack((x0_hat, x1_hat, x2_hat, x3_hat)))
    Y_hat = np.dot(X_hat, W)
    label = "dist[m]", "time[s]"
    # label = "order/horses[-]", "time[s]"
    show_graph(X_data[:,1], Y_data, x2_hat, Y_hat, label)
    # show_graph(X_data[:,2], Y_data, x2_hat, Y_hat, label)