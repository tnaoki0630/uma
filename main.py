import requests
import csv
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from makeCSV import scrape_hitchart, extract_dataset
import argparse
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
import pandas as pd

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
    # diff = np.subtract.outer(X1, X2) # 1入力
    diff = cdist(X1, X2, 'euclidean') # 複数入力
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
    # invK = np.linalg.pinv(K) # 擬似逆
    ## Log Likelifood
    # log_likelihood = - np.log(np.linalg.det(K)) - np.dot(Y.T, np.dot(invK, Y))
    sign, logabsdetK = np.linalg.slogdet(K)
    if sign <= 0:
        raise ValueError("Error at NLL: det K <= 0.")
    log_likelihood = - logabsdetK - np.dot(Y.T, np.dot(invK, Y))
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
        # print(- np.trace(np.dot(invK, dKdth[:,:,row])))
        # print(np.dot(iKY.T, np.dot(dKdth[:,:,row], iKY)))
    return -L_grad
###
def constructGPR(X_data, Y_data):
    ## initial hyper-parameter
    theta = np.array(args.parameter)
    ## optimize parameter
    # result = minimize(NLL, theta, method='BFGS', args=(X_data, Y_data)) # numerical gradient
    result = minimize(NLL, theta, method='L-BFGS-B', jac=NLL_grad, args=(X_data, Y_data)) # analytical gradient
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
    ## optimal RBF karnel
    K = RBFkarnel(theta, X_data, X_data, True, False)
    invK = np.linalg.inv(K)
    ## mu の計算
    k_hat = RBFkarnel(theta, X_data, X_hat, False, False)
    mu = np.dot(k_hat.T, np.dot(invK, Y_data))
    ## sgm の計算
    K_hat = RBFkarnel(theta, X_hat, X_hat, True, False)
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

#テラダ追記_始
def strtime2fltsec(arr):
    result = []
    for time_str in arr:
        # ":" で文字列を分割して分と秒に分ける
        minutes, seconds = map(float, time_str.split(':'))
        
        # 分を秒に変換して合算
        total_seconds = minutes * 60 + seconds
        
        result.append(total_seconds)
    return result
#テラダ追記_終

if __name__ == "__main__":
    ## コマンド引数関連
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ## データ読み込み方式
    parser.add_argument("-d", "--database", nargs=1, type=int, choices=range(2), help="READ = 0: html(default)\nREAD = 1: csv", default=[0])
    ## 学習に使用するテーブル数
    parser.add_argument("-n", "--number", nargs=1, type=int, help="NUMBER \\in Z (default: 3)", default=[3])
    ## モデル選択
    parser.add_argument("-m", "--model", nargs=1, type=int, choices=range(3), help="MODEL = 0: LR\nMODEL = 1: GPR w/o optimization (default)\nMODEL = 2: GPR w/ optimization", default=[1])
    ## ハイパーパラメータ
    parser.add_argument("-p", "--parameter", nargs=3, type=float, help="PARAMETER[0] \\in R (default: 0.0)\nPARAMETER[1] \\in R (default: 0.0)\nPARAMETER[2] \\in R (default: 0.0)", default=[-1, -1, -1])

    ## 引数の取得
    args = parser.parse_args()

    ## データ収集
    start_time = time.time()
    print("start data collection")
    N = min(100, max(1, args.number[0])) # 1~100となるよう例外処理
    varlist = ['ID', '頭 数', '馬 番', 'オ ッ ズ', '着 順', '斤 量', '上り', '馬体重', 'タイム']
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
        ## 事前取得したcsvからデータ取得
        csvpath = "UMA_DB.csv"
        df = pd.read_csv(csvpath)
        df = df[varlist]
        print(df)

        #テラダ追記_始
        unique_ids = df['ID'].unique()
        # 各IDごとに処理
        for id_value in unique_ids:
            # 特定のIDに対応するデータを抽出
            subset_data = df[df['ID'] == id_value]
            print(subset_data['タイム'])
            # ここで subset_data を使って必要な処理を実行
            subset_data['タイム'] = strtime2fltsec(subset_data['タイム'])
            average_time = subset_data['タイム'].mean()
            variance_time = np.sqrt(subset_data['タイム'].var())
            print(f"ID：{id_value}, 平均：{average_time}, 分散：{variance_time}")
        #テラダ追記_終
            
        exit()

        data = extract_dataset(ids[:N], required_data)
        if data:
            X_data = data[0]
            Y_data = data[1]
            mean_data = data[2]
            var_data = data[3]
        else:
            print("Error at extract_dataset")
            exit()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done. elapsed time: {elapsed_time}")

    ## 主成分分析

    ## 回帰モデルの構築
    start_time = time.time()
    print("start model construction")
    if args.model[0] == 0:
        ## 線形回帰
        W = constructLR(X_data, Y_data)
    elif args.model[0] == 2:
        ## ガウス過程回帰
        theta = constructGPR(X_data, Y_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done. elapsed time: {elapsed_time}")

    ## データ生成
    start_time = time.time()
    print("start data generation")
    x1_hat = np.linspace(-2, 2, 100) ## normalized
    X_hat = np.transpose(np.vstack((x1_hat, x1_hat, x1_hat, x1_hat)))
    if args.model[0] == 0:
        Y_hat = np.dot(X_hat, W)
    elif args.model[0] == 1:
        ## コマンド引数でパラメータを与える
        theta = np.array(args.parameter)
        mu, sgm = getGPR(theta, X_data, Y_data, X_hat)
    elif args.model[0] == 2:
        ## 最適化でパラメータを求める
        mu, sgm = getGPR(theta, X_data, Y_data, X_hat)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done. elapsed time: {elapsed_time}")

    ## 描画
    # label = "order/horses[-]", "time[s]"
    # show_graph(X_data[:,1], Y_data, x2_hat, Y_hat, label)

    ## 元のスケールに戻す
    Y_rawdata = Y_data*np.sqrt(var_data[0]) + mean_data[0]
    mu_raw = mu*np.sqrt(var_data[0]) + mean_data[0]

    ## L-t diagram
    # label = "dist[m]", "time[s]", "L-t"
    # # showGPR_1D(theta, X_data[:,0], Y_data, X_hat[:,0], mu, np.diagonal(sgm), label)
    # # showGPR_1D(theta, X_data[:], Y_data, X_hat[:], mu, np.diagonal(sgm), label)
    # X_rawdata = X_data[:,0]*np.sqrt(var_data[6]) + mean_data[6]
    # X_rawhat = X_hat[:,0]*np.sqrt(var_data[6]) + mean_data[6]
    # showGPR_1D(theta, X_rawdata, Y_rawdata, X_rawhat, mu_raw, np.diagonal(sgm), label)

    ## W-t diagram
    # label = "wight[kg]", "time[s]", "W-t"
    # # showGPR_1D(theta, X_data[:,1], Y_data, X_hat[:,1], mu, np.diagonal(sgm), label)
    # X_rawdata = X_data[:,3]*np.sqrt(var_data[7]) + mean_data[7]
    # X_rawhat = X_hat[:,3]*np.sqrt(var_data[7]) + mean_data[7]
    # showGPR_1D(theta, X_rawdata, Y_rawdata, X_rawhat, mu_raw, np.diagonal(sgm), label)

    ## o-v diagram
    label = "order [-]", "mean velocity [m/s]", "o-v"
    # showGPR_1D(theta, X_data[:,1], Y_data, X_hat[:,1], mu, np.diagonal(sgm), label)
    X_rawdata = X_data[:,0]*np.sqrt(var_data[3]) + mean_data[3]
    X_rawhat = X_hat[:,0]*np.sqrt(var_data[3]) + mean_data[3]
    showGPR_1D(theta, X_rawdata, Y_rawdata, X_rawhat, mu_raw, np.diagonal(sgm), label)

    ## lw-t diagram
    label = "load wight[kg]", "mean velocity [m/s]", "lw-t"
    # showGPR_1D(theta, X_data[:,1], Y_data, X_hat[:,1], mu, np.diagonal(sgm), label)
    X_rawdata = X_data[:,1]*np.sqrt(var_data[4]) + mean_data[4]
    X_rawhat = X_hat[:,1]*np.sqrt(var_data[4]) + mean_data[4]
    showGPR_1D(theta, X_rawdata, Y_rawdata, X_rawhat, mu_raw, np.diagonal(sgm), label)

    ## hw-v diagram
    label = "horse wight[kg]", "mean velocity [m/s]", "hw-v"
    X_rawdata = X_data[:,2]*np.sqrt(var_data[6]) + mean_data[6]
    X_rawhat = X_hat[:,2]*np.sqrt(var_data[6]) + mean_data[6]
    showGPR_1D(theta, X_rawdata, Y_rawdata, X_rawhat, mu_raw, np.diagonal(sgm), label)

    ## odds-v diagram
    label = "odds [-]", "mean velocity [m/s]", "odds-v"
    X_rawdata = X_data[:,3]*np.sqrt(var_data[7]) + mean_data[7]
    X_rawhat = X_hat[:,3]*np.sqrt(var_data[7]) + mean_data[7]
    # X_rawdata = np.exp(X_data[:,3]*np.sqrt(var_data[7]) + mean_data[7])
    # X_rawhat = np.exp(X_hat[:,3]*np.sqrt(var_data[7]) + mean_data[7])
    showGPR_1D(theta, X_rawdata, Y_rawdata, X_rawhat, mu_raw, np.diagonal(sgm), label)

    while True:
        try:
            print("input uma information: 馬番、頭数、斤量、体重、オッズ")
            user_input = input()
            ## :q と入力があったら終了
            if user_input == ":q": break
            ## 入力をスペース区切りで分割
            data = user_input.split()
            ## 型変換 + 正規化
            order = (int(data[0])/int(data[1]) - mean_data[3])/np.sqrt(var_data[3])
            weight1 = (float(data[2]) - mean_data[4])/np.sqrt(var_data[4])
            weight2 = (int(data[3]) - mean_data[6])/np.sqrt(var_data[6])
            odds = (np.log(float(data[4])) - mean_data[7])/np.sqrt(var_data[7])
            ## 説明変数の準備
            x_hat = np.array([order, weight1, weight2, odds])
            x_hat = x_hat.reshape((1,4))
            ## 予測
            mu, sgm = getGPR(theta, X_data, Y_data, x_hat)
            print(f"mean: {mu*np.sqrt(var_data[0])+mean_data[0]}")
            print(f"var: {np.diagonal(sgm)}")
        except ValueError:
            print("input error")
