import requests
from bs4 import BeautifulSoup
import csv
import numpy as np
import re
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from makeCSV import scrape_hitchart, scrape_grades, extract_integer, strtime2fltsec, strdate2intdate, extract_dataset
import argparse
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
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
def RBFkarnel(theta, X1, X2, BOOL_diag, BOOL_grad):
    ## RBFカーネル
    diff = np.subtract.outer(X1, X2) # 1入力
    # diff = cdist(X1, X2, 'euclidean') # 複数入力
    Karnel = np.exp(theta[0])*np.exp(-0.5*(diff**2/np.exp(2*theta[1])))
    if BOOL_diag:
        ## 対角成分に観測ノイズの補正を付与
        Karnel = Karnel + np.exp(theta[2])*np.eye(X1.shape[0])
    if BOOL_grad:
        ## 勾配を出力
        dKdtheta = np.zeros((X1.shape[0], X2.shape[0], 3))
        dKdtheta[:,:,0] = np.exp(theta[0])*np.exp(-0.5*(diff**2/np.exp(2*theta[1])))
        dKdtheta[:,:,1] = np.dot(dKdtheta[:,:,0], diff**2)/np.exp(2*theta[1])
        dKdtheta[:,:,2] = np.exp(theta[2])*np.eye(X1.shape[0])
        return Karnel, dKdtheta
    else:
        return Karnel
###
def NLL(theta, X, Y):
    ## RBF Karnel
    K = RBFkarnel(theta, X, X, True, False)
    invK = np.linalg.inv(K)
    ## Log Likelifood
    log_likelihood = - np.log(np.linalg.det(K)) - np.dot(Y.T, np.dot(invK, Y))
    return -log_likelihood
###
def NLL_grad(theta, X, Y):
    ## minimize では NLL_grad に引数を渡せないので、いったんNLLを呼び出す
    negative_log_likelihood = NLL(theta, X, Y)
    ## RBF Karnel
    K, dKdth = RBFkarnel(theta, X, X, True, True)
    invK = np.linalg.inv(K)
    # invK = np.linalg.pinv(K) # 擬似逆行列
    iKY = np.dot(invK, Y)
    ## Likelifood gradient
    rows = np.shape(theta)
    L_grad = np.zeros(rows[0])
    for row in range(rows[0]):
        L_grad[row] = - np.trace(np.dot(invK, dKdth[:,:,row])) + np.dot(iKY.T, np.dot(dKdth[:,:,row], iKY))
        print(- np.trace(np.dot(invK, dKdth[:,:,row])))
        print(np.dot(iKY.T, np.dot(dKdth[:,:,row], iKY)))
    return -L_grad
###
def constructGPR(X_data, Y_data):
    ## initial hyper-parameter
    theta = np.array(args.parameter)
    ## normalize
    X_normal = (X_data-np.mean(X_data))/np.sqrt(np.var(X_data)) # 1入力
    Y_normal = (Y_data-np.mean(Y_data))/np.sqrt(np.var(Y_data))
    ## optimize parameter
    # result = minimize(NLL, theta, method='BFGS', args=(X_normal, Y_normal)) # numerical gradient
    result = minimize(NLL, theta, method='L-BFGS-B', jac=NLL_grad, args=(X_normal, Y_normal)) # analytical gradient
    theta = result.x
    print("Minimized value:", result.fun)
    print("Optimal parameters:", result.x)
    print("Success:", result.success)
    print("Message:", result.message)
    print("Number of iterations:", result.nit)
    print("Number of function evaluations:", result.nfev)
    return theta
###
def getGPR(theta, X_data, Y_data, X_hat):
    ## normalize
    X_ndata = (X_data-np.mean(X_data))/np.sqrt(np.var(X_data)) # 1入力
    Y_ndata = (Y_data-np.mean(Y_data))/np.sqrt(np.var(Y_data))
    X_nhat = (X_hat-np.mean(X_data))/np.sqrt(np.var(X_data)) # 1入力
    ## optimal RBF karnel
    K = RBFkarnel(theta, X_ndata, X_ndata, True, False)
    invK = np.linalg.inv(K)
    ## mu の計算
    k_hat = RBFkarnel(theta, X_ndata, X_nhat, False, False)
    mu = np.dot(k_hat.T, np.dot(invK, Y_ndata))
    # 元のスケールに戻す
    mu = mu*np.sqrt(np.var(Y_data)) + np.mean(Y_data)

    ## sgm の計算
    K_hat = RBFkarnel(theta, X_nhat, X_nhat, True, False)
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
def showGPR_1D(theta, X, Y, x, y_mean, y_var, arr):
    ## 描画
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(X, Y, marker='.')
    ax.plot(x, y_mean, 'tab:red')
    ax.fill_between(x, y_mean - 2*y_var, y_mean + 2*y_var, color='lightblue', alpha=0.5, label='Variance Range')
    ax.set_xlabel(arr[0])
    ax.set_ylabel(arr[1])
    ax.set_title(f'theta1 = {theta[0]:.2f}, theta2 = {theta[1]:.2f}, theta3 = {theta[2]:.2f}')
    ax.grid()
    plt.savefig(f'GPR_{arr[2]}.png')

if __name__ == "__main__":
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
    start_time = time.time()
    print("start data collection")
    N = min(100, max(1, args.number[0])) # 1~100となるよう例外処理
    required_data = 6, 7, 14, 17, 23 # 頭数, 枠番, 距離, タイム, 体重
    if args.database[0] == 0:
        ## ヒットチャートのページから馬ごとのURLを取得
        ids = scrape_hitchart("https://db.netkeiba.com/?pid=horse_hitchart")
        if not all(ids):
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
        ### (下みたいな感じでhtmlを解析する時と同じように書けたらめっちゃラク。
        ### extract_datasetはhtml解析も含んでしまってるので、コードの設計を変える必要がある。)
        data = extract_dataset(ids[:N], required_data)
        if data:
            X_data = data[0]
            Y_data = data[1]
        else:
            print("Error at extract_dataset")
            exit()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done. elapsed time: {elapsed_time}")

    ## 回帰モデルの構築
    start_time = time.time()
    print("start model construction")
    if args.model[0] == 0:
        ## 線形回帰
        W = constructLR(X_data, Y_data)
    elif args.model[0] == 1:
        ## ガウス過程回帰
        theta = constructGPR(X_data, Y_data)
    elif args.model[0] == 2:
        ## ニューラルネット
        print("Error:: NN is under development...")
        exit()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done. elapsed time: {elapsed_time}")

    ## データ生成
    start_time = time.time()
    print("start data generation")
    x1_hat = np.linspace(1, 3, 100)
    x6_hat = np.linspace(1500, 2500, 100)
    x3_hat = np.linspace(0, 1, 100)
    x7_hat = np.linspace(440, 540, 100)
    x0_hat = np.ones_like(x1_hat)
    # X_hat = np.transpose(np.vstack((x0_hat, x2_hat)))
    # X_hat = np.transpose(np.vstack((x0_hat, x1_hat, x2_hat, x3_hat)))
    # X_hat = np.transpose(np.vstack((x6_hat, x7_hat)))
    X_hat = x6_hat
    if args.model[0] == 0:
        Y_hat = np.dot(X_hat, W)
    elif args.model[0] == 1:
        mu, sgm = getGPR(theta, X_data, Y_data, X_hat)
    elif args.model[0] == 2:
        Y_hat = getNN(X_hat)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done. elapsed time: {elapsed_time}")

    ## 描画
    # label = "order/horses[-]", "time[s]"
    # show_graph(X_data[:,1], Y_data, x2_hat, Y_hat, label)
    label = "dist[m]", "time[s]", "L-t"
    # showGPR_1D(theta, X_data[:,0], Y_data, X_hat[:,0], mu, np.diagonal(sgm), label)
    showGPR_1D(theta, X_data[:], Y_data, X_hat[:], mu, np.diagonal(sgm), label)
    # label = "wight[kg]", "time[s]", "W-t"
    # showGPR_1D(theta, X_data[:,1], Y_data, X_hat[:,1], mu, np.diagonal(sgm), label)
    # print(mu)
    # print(np.diagonal(sgm))