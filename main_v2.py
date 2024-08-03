import requests
import csv
import numpy as np
import matplotlib.pyplot as plt
from makeCSV import strtime2fltsec, extract_integer, strtime2fltsec_v2, extract_integer_v2, extract_integer_in_parenthesis_v2
import argparse
import time
import pandas as pd
import math
from scipy.special import erf
from itertools import combinations, permutations
import random as rnd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline

def showPDF(csvpath, v, arr_label):
    # csv から取得
    df = pd.read_csv(csvpath)
    df = df[varlist]
    df = df.dropna() # NaNを削除

    fig, ax = plt.subplots(dpi=200)
    # 各IDごとに処理
    unique_ids = df['ID'].unique()
    i = 0
    for id_value in unique_ids:
        i += 1
        # 特定のIDに対応するデータを抽出
        subset_data = df[df['ID'] == id_value]
        # ここで subset_data を使って必要な処理を実行
        arr_time = np.array(strtime2fltsec(subset_data['タイム']))
        arr_dist = np.array(extract_integer(subset_data['距離']))
        arr_vel = arr_dist/arr_time
        
        # 期待値、標準偏差
        mu = arr_vel.mean()
        if(arr_vel.std() == 0):
            sgm = 1e-8
        else:
            sgm = arr_vel.std()
        # 確率分布とデータのプロット
        p_v = 1/np.sqrt(2*math.pi)/sgm*np.exp(-(v-mu)**2/(2*sgm**2))
        ax.plot(v, p_v, label=f'uma #{i}')
        p_v = [0.01+0.1*(i-1)+0.08*rnd.random() for iter in range(len(arr_vel))]
        ax.scatter(arr_vel, p_v, s=10, marker='.', alpha=0.1)
    ax.set_xlabel(arr_label[0])
    ax.set_ylabel(arr_label[1])
    ax.legend(loc='upper right')
    ax.grid()
    plt.savefig(f'fig/PDF.png')
##
def tanshou_integration(Numa, arr_mu, arr_sgm, arr_odds, vmin, vmax, dv):
    arr_num = []
    arr_prob = []
    arr_earn = []
    for i in range(Numa):
        coef = 1/(math.sqrt(2*math.pi)*arr_sgm[i])
        # 初期化
        sum = 0
        for j in range(Nv):
            vi = vmin[i] + dv[i]*j
            # 初期化
            prod = 1
            for k in range(Numa):
                # i 以外についてerfの積をとる
                if(i != k):
                    intg_top = math.erf((vi-arr_mu[k])/(math.sqrt(2)*arr_sgm[k]))
                    intg_btm = math.erf(-arr_mu[k]/(math.sqrt(2)*arr_sgm[k]))
                    prod *= (intg_top - intg_btm)/2
            sum += math.exp(-(vi-arr_mu[i])**2/(2*arr_sgm[i]**2))*prod*dv[i]
        arr_num.append(i)
        arr_prob.append(coef*sum)
        arr_earn.append(coef*sum*arr_odds[i])
    # リストをDataFrameに変換
    df_num = pd.DataFrame(arr_num)
    df_prob = pd.DataFrame(arr_prob)
    df_earn = pd.DataFrame(arr_earn)
    # 列名を設定
    df_num.columns = ['うま番号']
    df_prob.columns = ['当選確率']
    df_earn.columns = ['払戻金']
    # DataFrameを結合
    DataFrame = pd.concat([df_num, df_prob, df_earn], axis=1)
    print(f"sum of prob: {(df_prob.values).sum()}")
    print(f"tanshou probability: {arr_prob}")
    return DataFrame.dropna()
##
def tanshou_integration_v0(Numa, arr_mu, arr_sgm, arr_odds, vmin, vmax, dv):
    prod = 1
    for i in range(Numa):
        prod *= arr_sgm[i]
    coef = 1/(math.sqrt((2*math.pi)**Numa)*prod)
    # 積分計算
    arr_num = []
    arr_prob = []
    arr_earn = []
    for i in range(Numa):
        sum = 0
        for j in range(Nv):
            vi = vmin[i] + dv[i]*j
            prod = 1
            for k in range(Numa):
                # i 以外について 0 から vi まで数値積分して、その積をとる
                if(i != k):
                    sum2 = 0
                    for l in range(Nv):
                        vk = vmin[k] + dv[k]*l
                        if(vk < vi):
                            sum2 += math.exp(-(vk-arr_mu[k])**2/(2*arr_sgm[k]**2))*dv[k]
                    prod *= sum2
            sum += math.exp(-(vi-arr_mu[i])**2/(2*arr_sgm[i]**2))*prod*dv[i]
        arr_num.append(i)
        arr_prob.append(coef*sum)
        arr_earn.append(coef*sum*arr_odds[i])
    # リストをDataFrameに変換
    df_num = pd.DataFrame(arr_num)
    df_prob = pd.DataFrame(arr_prob)
    df_earn = pd.DataFrame(arr_earn)
    # 列名を設定
    df_num.columns = ['うま番号']
    df_prob.columns = ['当選確率']
    df_earn.columns = ['払戻金']
    # DataFrameを結合
    DataFrame = pd.concat([df_num, df_prob, df_earn], axis=1)
    print(f"sum of prob: {df_prob.sum()}")
    print(f"tanshou probability: {arr_prob}")
    return DataFrame.dropna()
##
def sanrentan_integration(Numa, arr_mu, arr_sgm, arr_odds, vmin, vmax, dv):
    tensor_num = np.zeros((Numa, Numa, Numa), dtype=object)
    tensor_prob = np.zeros((Numa, Numa, Numa))
    tensor_earn = np.zeros((Numa, Numa, Numa))
    ## 愚直に積分
    for i in range(Numa):
        for j in range(Numa):
            if(j != i):
                for k in range(Numa):
                    if(k != i and k != j):
                        print(f"integrating for i = {i}, j = {j}, k = {k}")
                        coef = 1/(math.sqrt((2*math.pi)**3)*arr_sgm[i]*arr_sgm[j]*arr_sgm[k])
                        # i について積分
                        sum_i = 0
                        for ii in range(Nv):
                            vi = vmin[i] + dv[i]*ii
                            # j について積分
                            sum_j = 0
                            for jj in range(Nv):
                                vj = vmin[j] + dv[j]*jj
                                if(vj < vi):
                                    # k について積分
                                    sum_k = 0
                                    for kk in range(Nv):
                                        vk = vmin[k] + dv[k]*kk
                                        if(vk < vj):
                                            prod = 1
                                            for l in range(Numa):
                                                # i,j,k 以外についてerfの積をとる
                                                if(l != i and l != j and l != k):
                                                    intg_top = math.erf((vk-arr_mu[l])/(math.sqrt(2)*arr_sgm[l]))
                                                    intg_btm = math.erf(-arr_mu[l]/(math.sqrt(2)*arr_sgm[l]))
                                                    prod *= (intg_top - intg_btm)/2
                                            sum_k += math.exp(-(vk-arr_mu[k])**2/(2*arr_sgm[k]**2))*prod*dv[k]
                                    sum_j += math.exp(-(vj-arr_mu[j])**2/(2*arr_sgm[j]**2))*sum_k*dv[j]
                            sum_i += math.exp(-(vi-arr_mu[i])**2/(2*arr_sgm[i]**2))*sum_j*dv[i]
                        # うま番号
                        tensor_num[i,j,k] = f"{i+1}-{j+1}-{k+1}"
                        # 当選確率
                        tensor_prob[i,j,k] = coef*sum_i
                        # 払戻金の期待値（いったん掛け算で計算）
                        # tensor_earn[i,j,k] = coef*sum_i*arr_odds[i]*arr_odds[j]*arr_odds[k]
                        tensor_earn[i,j,k] = coef*sum_i*arr_odds[i]**3*arr_odds[j]**2*arr_odds[k]
                    else:
                        # 重複するものはNone
                        tensor_num[i,j,k] = None
                        tensor_prob[i,j,k] = None
                        tensor_earn[i,j,k] = None
            else:
                for k in range(Numa):
                    # 重複するものはNone
                    tensor_num[i,j,k] = None
                    tensor_prob[i,j,k] = None
                    tensor_earn[i,j,k] = None
    # リストをDataFrameに変換
    df_num = pd.DataFrame(tensor_num.flatten())
    df_prob = pd.DataFrame(tensor_prob.flatten())
    df_earn = pd.DataFrame(tensor_earn.flatten())
    # 列名を設定
    df_num.columns = ['うま番号']
    df_prob.columns = ['当選確率']
    df_earn.columns = ['払戻金']
    # DataFrameを結合
    DataFrame = pd.concat([df_num, df_prob, df_earn], axis=1)
    print(f"sum of prob: {df_prob.sum()}")
    print(f"tanshou probability: {np.nansum(tensor_prob, axis=(1,2))}")
    return DataFrame.dropna()
##
def sanrentan_integration_v2(Numa, arr_mu, arr_sgm, arr_odds, vmin, vmax, dv):
    tensor_num = np.zeros((Numa, Numa, Numa), dtype=object)
    tensor_prob = np.zeros((Numa, Numa, Numa))
    tensor_earn = np.zeros((Numa, Numa, Numa))
    ## ベクトル演算で積分
    for i in range(Numa):
        for j in range(Numa):
            if(j != i):
                for k in range(Numa):
                    if(k != i and k != j):
                        coef = 1/(math.sqrt((2*math.pi)**3)*arr_sgm[i]*arr_sgm[j]*arr_sgm[k])
                        
                        # v のnp配列
                        vi = np.linspace(vmin[i], vmax[i], Nv)
                        vj = np.linspace(vmin[j], vmax[j], Nv)
                        vk = np.linspace(vmin[k], vmax[k], Nv)
                        
                        # exp のnp配列
                        vi_exp = np.exp(-(vi-arr_mu[i])**2/(2*arr_sgm[i]**2))
                        vj_exp = np.exp(-(vj-arr_mu[j])**2/(2*arr_sgm[j]**2))
                        vk_exp = np.exp(-(vk-arr_mu[k])**2/(2*arr_sgm[k]**2))
                        
                        # 積分領域を指定するブール行列 (H_ij: R^N_i -> R^N_j)
                        BOOL_vk = vk[np.newaxis, :] < vj[:, np.newaxis]
                        BOOL_vj = vj[np.newaxis, :] < vi[:, np.newaxis]

                        # i,j,k 以外についてerfの積をとる
                        prod = 1
                        for l in range(Numa):
                            if(l != i and l != j and l != k):
                                prod *= (erf((vk-arr_mu[l])/(math.sqrt(2)*arr_sgm[l])) - erf(-arr_mu[l]/(math.sqrt(2)*arr_sgm[l])))/2
                        
                        ## 積分
                        sum_k = np.dot(BOOL_vk, vk_exp*prod)*dv[k]
                        sum_j = np.dot(BOOL_vj, vj_exp*sum_k)*dv[j]
                        sum_i = np.sum(vi_exp*sum_j)*dv[i]

                        # うま番号
                        tensor_num[i,j,k] = f"{i+1}-{j+1}-{k+1}"
                        # 当選確率
                        tensor_prob[i,j,k] = coef*sum_i
                        # 払戻金の期待値
                        # tensor_earn[i,j,k] = coef*sum_i*arr_odds[i]*arr_odds[j]*arr_odds[k] ## ただの掛け算
                        # tensor_earn[i,j,k] = coef*sum_i*(arr_odds[i]**3)*(arr_odds[j]**2)*(arr_odds[k]) ## 冪乗して重みづけ（やりすぎ？）
                        tensor_earn[i,j,k] = coef*sum_i*(3*arr_odds[i])*(2*arr_odds[j])*(arr_odds[k]) ## 係数で重みづけ
                    else:
                        # 重複するものはNone
                        tensor_num[i,j,k] = None
                        tensor_prob[i,j,k] = None
                        tensor_earn[i,j,k] = None
            else:
                for k in range(Numa):
                    # 重複するものはNone
                    tensor_num[i,j,k] = None
                    tensor_prob[i,j,k] = None
                    tensor_earn[i,j,k] = None
    # 各テンソルを1次元のDataFrameに変換
    df_num = pd.DataFrame(tensor_num.flatten())
    df_prob = pd.DataFrame(tensor_prob.flatten())
    df_earn = pd.DataFrame(tensor_earn.flatten())
    # DataFrameを結合
    DataFrame_ordered = pd.concat([df_num, df_prob, df_earn], axis=1)
    # 列名を設定
    DataFrame_ordered.columns = ['うま番号', '当選確率', '払戻金']

    ## 三連複の確率計算
    num_unordered = []
    prob_unordered = []
    earn_unordered = []
    # 順序を区別しない組み合わせのリストを取得
    for combo in combinations(range(Numa), 3):
        sum = 0
        # タプルで馬番を取得
        i, j, k = combo
        num_unordered.append(f"{i+1}-{j+1}-{k+1}")
        # combo の中で、順序を区別する全ての組み合わせを取得
        for perm in permutations(combo):
            # 確率の和をとる
            sum += tensor_prob[perm]
        prob_unordered.append(sum)
        # 払戻金を計算
        earn_unordered.append(sum*arr_odds[i]*arr_odds[j]*arr_odds[k])
        ## check
        # print(combo, sum, tensor_prob[combo])
    # DataFrameに変換
    df_num = pd.DataFrame(num_unordered)
    df_prob = pd.DataFrame(prob_unordered)
    df_earn = pd.DataFrame(earn_unordered)
    # DataFrameを結合
    DataFrame_unordered = pd.concat([df_num, df_prob, df_earn], axis=1)
    # 列名を設定
    DataFrame_unordered.columns = ['うま番号', '当選確率', '払戻金']
    print(f"sum of prob: {np.nansum(tensor_prob)}")
    print(f"tanshou probability: {np.nansum(tensor_prob, axis=(1,2))}")
    return DataFrame_ordered.dropna(), DataFrame_unordered
##
if __name__ == "__main__":
    ## コマンド引数関連
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ## モデル選択
    parser.add_argument("-m", "--model", nargs=1, type=int, choices=range(6), help="\
                        MODEL = 0: LR\n\
                        MODEL = 1: GPR w/o optimization\n\
                        MODEL = 2: GPR w/ optimization\n\
                        MODEL = 3: Gauss distribution (default)\n\
                        MODEL = 4: Gauss distribution w/ pedigree\n\
                        MODEL = 5: Gauss distribution w/ Bayesian\n\
                        ", default=[3])
    ## UMA or Jockey
    parser.add_argument("-b", "--binomen", nargs=1, type=int, choices=range(2), help="\
                        BINOMEN = 0: Equus caballus (default)\n\
                        BINOMEN = 1: Homo sapiens\n\
                        ", default=[0])
    ## 引数の取得
    args = parser.parse_args()

    ## データ収集
    start_time = time.time()
    print("start data collection")
    
    ### 事前取得したcsvから取得
    ## 戦績
    if args.binomen[0] == 0:
        df = pd.read_csv("data/UMA_DB_EL_train.csv")
    elif args.binomen[0] == 1:
        df = pd.read_csv("data/Jockey_DB_EL.csv")
    ## データフレームを使用する変数だけに削減
    varlist = ['ID', '頭 数', '馬 番', 'オ ッ ズ', '着 順', '斤 量', '上り', '馬体重', 'タイム', '距離']
    # varlist = ['ID', 'タイム', '距離']
    df = df[varlist]
    ## NaNを削除
    df = df.dropna()
    ## 文字列変換
    df['タイム'] = df['タイム'].apply(strtime2fltsec_v2)
    df['距離'] = df['距離'].apply(extract_integer_v2)
    ##馬体重と増減の分離
    df['体重増減'] = df['馬体重'].apply(extract_integer_in_parenthesis_v2)
    df['馬体重'] = df['馬体重'].apply(extract_integer_v2) 
    ## 平均速度の追加
    df['速度'] = df['距離'] / df['タイム']
    #着順をint型に変換
    df['着 順'] = df['着 順'].astype(int)
    ## IDと平均速度だけ使用
    #df = df[['ID', '速度']]
    ## たまに文字列処理の例外が出るので、改めてNaNを削除
    #df = df.dropna()

#PLS回帰？
#訓練データから主成分とローディングの算出
    #訓練データの取得と標準化（y_train：目的変数，X_train：説明変数）
    y_train = df['タイム']
    X_train = df[['馬 番', 'オ ッ ズ', '馬体重', '体重増減', '距離']]
    X_mean = np.mean(X_train, axis=0)
    X_std_dev = np.std(X_train, axis=0)
    X_std = (X_train-X_mean)/X_std_dev
    y_mean = np.mean(y_train)
    y_std_dev = np.std(y_train)
    y_std = (y_train-y_mean)/y_std_dev

    N_PLS = 3
    X = X_std
    y = y_std

    w = np.dot(X_std.T, y_std)/np.linalg.norm(np.dot(X_std.T, y_std))
    t = np.dot(X_std, w)
    pn = np.dot(X_std.T, t)/np.dot(t.T, t)
    qn = np.dot(y_std.T, t)/np.dot(t.T, t)
    p = pn
    q = qn

    for i in range(N_PLS-1):
        X = X - np.dot(t.reshape(-1,1), pn.reshape(1, -1))
        y = y - qn * t
        Wn = np.dot(X.T, y)/np.linalg.norm(np.dot(X.T, y))
        t = np.dot(X, Wn)
        pn = np.dot(X.T, t)/np.dot(t.T, t)
        qn = np.dot(y.T, t)/np.dot(t.T, t)
        W = np.vstack((w, Wn))
        p = np.vstack((p, pn))
        if i == 0:
            q = np.array([q, qn])
        else:
            qn = np.array([qn])
            q = np.hstack([q,qn])
    W = W.T
    p = p.T


    """#第1成分の算出
    w1 = np.dot(X_std.T, y_std)/np.linalg.norm(np.dot(X_std.T, y_std))
    t1 = np.dot(X_std, w1)
    p1 = np.dot(X_std.T, t1)/np.dot(t1.T, t1)
    q1 = np.dot(y_std.T, t1)/np.dot(t1.T, t1)
    #第２成分の算出
    X2 = X_std-np.dot(t1.reshape(-1,1), p1.reshape(1, -1))
    y2 = y_std-q1*t1
    w2 = np.dot(X2.T, y2)/np.linalg.norm(np.dot(X2.T, y2))
    t2 = np.dot(X2, w2)
    p2 = np.dot(X2.T, t2)/np.dot(t2.T, t2)
    q2 = np.dot(y2.T, t2)/np.dot(t2.T, t2)
    #第3成分の算出
    X3 = X2-np.dot(t2.reshape(-1,1), p2.reshape(1, -1))
    y3 = y2-q2*t2
    w3 = np.dot(X3.T, y3)/np.linalg.norm(np.dot(X3.T, y3))
    t3 = np.dot(X3, w3)
    p3 = np.dot(X3.T, t3)/np.dot(t3.T, t3)
    q3 = np.dot(y3.T, t3)/np.dot(t3.T, t3)
#w, p, qの整理
    W = np.vstack((w1, w2, w3))
    W = W.T
    p = np.vstack((p1, p2, p3))
    p = p.T
    q = np.array([q1, q2, q3])
"""

#テストデータの取得
    df_test = pd.read_csv("data/UMA_DB_EL_test.csv")
    df_test = df_test[varlist]
    ## NaNを削除
    df_test = df_test.dropna()
    ## 文字列変換
    df_test['タイム'] = df_test['タイム'].apply(strtime2fltsec_v2)
    df_test['距離'] = df_test['距離'].apply(extract_integer_v2)
    ##馬体重と増減の分離
    df_test['体重増減'] = df_test['馬体重'].apply(extract_integer_in_parenthesis_v2)
    df_test['馬体重'] = df_test['馬体重'].apply(extract_integer_v2) 
    ## 平均速度の追加
    df_test['速度'] = df_test['距離'] / df_test['タイム']
    #着順をint型に変換
    df_test['着 順'] = df_test['着 順'].astype(int)
    ## IDと平均速度だけ使用
    #df = df[['ID', '速度']]
    ## たまに文字列処理の例外が出るので、改めてNaNを削除
    #df = df.dropna()
    #IDリストの作成
    ID_test = np.unique(df_test['ID'])
    #print(ID_test)
    UMA_1 = ID_test[2]
    #print(UMA_1)
    df_test = df_test[df_test.iloc[:, 0] == UMA_1]

    y_test = df_test['タイム']
    X_test = df_test[['馬 番', 'オ ッ ズ', '馬体重', '体重増減', '距離']]
    #print(X_test)

#yの推定
    X_test_mean = np.mean(X_test, axis=0)
    X_test_std_dev = np.std(X_test, axis=0)
    X_test_std = (X_test-X_test_mean)/X_test_std_dev
    print(X_test_std)
    y_test_mean = np.mean(y_test, axis=0)
    y_test_std_dev = np.std(y_test, axis=0)
    y_test_std = (y_test-y_test_mean)/y_test_std_dev
    W_pls = np.dot(W, np.linalg.inv(np.dot(p.T, W)))
    print(W_pls)
    y_pre_1 = np.dot(X_test_std, W_pls)
    y_pre = np.dot(y_pre_1, q)
    print(y_pre)


    # パイプラインのインスタンス生成
    #PCR_model = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
    # 学習
    #PCR_model.fit(X, Y)
    #PLS回帰モデルの作成
    #PLS = PLSRegression(n_components=2)
    #PLS.fit(X, Y)
    #主成分の取得
    #X_scores = PLS.x_scores_
    #print(X_scores)
    #print(np.shape(X_scores))

    #print(X_test_std)
    #print(y_test_std)
    result = pd.concat([X_test_std, y_test_std], axis = 1)
    y_pre = pd.DataFrame({'予測タイム': y_pre})
    result.reset_index(drop=True, inplace=True)
    result = pd.concat([result, y_pre], axis = 1)
    print(result)

    """unique_labels = np.unique(result['距離'])
    print(unique_labels)
    # プロット
    result_X = result['馬 番']
    result_y = result['タイム']
    for label in unique_labels:
        indices = unique_labels == label
        plt.scatter(result_X[indices], result[indices], label = f'Label{label}')
"""
    """
    labels = result['距離'].unique()
    # プロット
    plt.figure(figsize=(10, 6))
    for label in labels:
        subset =result[result['距離'] == label]
        plt.plot(subset['体重増減'], subset['タイム'], marker='x', linestyle='', label=label)
        plt.plot(subset['体重増減'], subset['予測タイム'], marker='o', linestyle='', label=label)
"""
    plot_X = result['オ ッ ズ']
    plot_y_true = result['タイム']
    plot_y_pre = result['予測タイム']
    legend_data = result['距離']

    unique_legends = np.unique(legend_data)
    color = plt.cm.jet(np.linspace(0, 1, len(unique_legends)))

    plt.figure(figsize=(10, 5))

    for i, legend_value in enumerate(unique_legends):
        mask = legend_data == legend_value
        plt.plot(plot_X[mask], plot_y_true[mask], 'o', label = f'true (distance:{legend_value}) ', color = color[i], fillstyle = 'none')

    for i, legend_value in enumerate(unique_legends):
        mask = legend_data == legend_value
        plt.plot(plot_X[mask], plot_y_pre[mask], 'x', label = f'predicted (distance:{legend_value}) ', color = color[i])



    #plt.scatter(X_test_std['距離'], y_test_std, c = 'red', label = 'true')
    #plt.scatter(X_test_std['距離'], y_pre, marker = 'x', c = 'blue', label = 'predict')
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0.)
    plt.title('test data')
    plt.xlabel("weight")
    plt.ylabel("time")
    plt.tight_layout()
    plt.show()

    # R2（決定係数）
    #R2 = round(PCR_model.score(X, Y),2)
    """# グラフ
    plt.scatter(Y, PCR_model.predict(X))
    plt.title('Train data  -  R2='+str(R2))
    plt.xlabel("actual")
    plt.ylabel("predicted")
    plt.show()"""

    exit()#確認用，後で消す

    ## DB を使用
    if args.model[0] == 4:
        ## 例外処理
        if args.binomen[0] != 0:
            print("Fatal Error: Jockey with pedigree is not supported.")
            exit()
        ## 戦績
        df_all = pd.read_csv("data/UMA_DB.csv")
        df_all = df_all[varlist]
        df_all = df_all.dropna()
        df_all['タイム'] = df_all['タイム'].apply(strtime2fltsec_v2)
        df_all['距離'] = df_all['距離'].apply(extract_integer_v2)
        df_all['速度'] = df_all['距離'] / df_all['タイム']
        df_all = df_all[['ID', '速度']]
        df_all = df_all.dropna()
        ## 血統データ
        df_ped_all = pd.read_csv("data/UMA_ped.csv")
        df_ped = pd.read_csv("data/UMA_ped_EL.csv")

    ## オッズ
    arr_odds = []
    with open("data/UMA_ODDS.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            arr_odds.append(float(row[0]))  # データをfloat型に変換して追加

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done. elapsed time: {elapsed_time}\n")

    ## 回帰モデルの構築
    start_time = time.time()
    print("start model construction")
    if args.model[0] == 3:
        ## 正規分布
        unique_ids = df['ID'].unique()
        # mu, sgm を格納する配列を準備
        Numa = len(unique_ids)
        arr_mu = []
        arr_sgm = []
        # 各IDごとに処理
        for id_value in unique_ids:
            # 特定のIDに対応するデータを抽出
            subset_data = df[df['ID'] == id_value]
            ## Numpy 配列として格納
            arr_mu = np.append(arr_mu, subset_data['速度'].mean())
            if(subset_data['速度'].std() == 0):
                arr_sgm = np.append(arr_sgm, 1e-8)
            else:
                arr_sgm = np.append(arr_sgm, subset_data['速度'].std())
        print(f"mu: {arr_mu}")
        print(f"sgm: {arr_sgm}")
    elif args.model[0] == 4:
        ## 正規分布 w/ 血統データ
        unique_ids = df['ID'].unique()
        # mu, sgm を格納する配列を準備
        Numa = len(unique_ids)
        arr_mu = []
        arr_sgm = []
        # 正規化定数
        phi = 0.001/(5*2**5) # 血統情報が全て一致する馬のデータをスカラー倍してカウント
        # 血統の重みベクトル（親等と字数が対応しているためlenで判定）
        arr_weight =  np.array([2**(5-len(idx)) for idx in list(df_ped)])
        # 各IDごとに処理
        for id_value in unique_ids:
            # 特定のIDに対応するデータを抽出
            subset_data = df[df['ID'] == id_value]
            vi_sum = subset_data['速度'].sum()
            ## 距離関数の計算
            # df_ped から該当するIDの血統データを抽出
            subset_ped = df_ped[df_ped['ID'] == id_value]
            # df_ped_all からumaIDが一致する行を探索
            matching_rows = df_ped_all[df_ped_all['ID'] == id_value]
            # 一致する行を取り除いたデータフレームを作成
            df_ped_reduced = df_ped_all.drop(matching_rows.index)
            # df_ped_reducedと同じサイズになるようにsubset_pedをコピーし、Numpy配列として保持 
            subset_expanded = np.array([subset_ped.values[0]]*len(df_ped_reduced))
            # ブール行列
            H = subset_expanded == df_ped_reduced.values
            # 血統の一致度を距離関数として採用
            arr_score = np.dot(H,arr_weight)
            ## 距離関数を用いた重み付け和の計算
            # df_all からumaIDが一致する行を探索
            matching_rows = df_all[df_all['ID'] == id_value]
            # 一致する行を取り除いたデータフレームを作成
            df_reduced = df_all.drop(matching_rows.index)
            # ID ごとにグループ化して和をとる
            df_groupedsum = df_reduced.groupby('ID', sort=False)['速度'].sum()
            # 重み付きのデータ数
            Zi = len(subset_data) + phi*(arr_score*len(df_groupedsum)).sum()
            # 速度データを足しこんで平均値を計算
            sum = vi_sum + phi*(arr_score*df_groupedsum.values).sum()
            mu = sum/Zi
            arr_mu = np.append(arr_mu, mu)
            # 自乗誤差ベクトルの計算
            sqerrsum = ((subset_data['速度']-mu)**2).sum()
            df_groupedsum = df_reduced.groupby('ID', sort=False)['速度'].apply(lambda vi: ((vi-mu) ** 2).sum())
            if False: # groupby を使わずに和を取った場合と比較して検証
                check = df_reduced['速度'].apply(lambda vi: ((vi-sum/Zi) ** 2))
                print(f"groupedsum: {df_groupedsum}")
                print(f"check: {check}")
                print(f"error: {df_groupedsum.sum()-check.sum()}")
            sum = sqerrsum + phi*(arr_score*df_groupedsum.values).sum()
            arr_sgm = np.append(arr_sgm, np.sqrt(sum/Zi))
        # テーブルの統計量
        print(f"mu w/o ped: {(df.groupby('ID', sort=False)['速度'].mean()).values}")
        print(f"sgm w/o ped: {(df.groupby('ID', sort=False)['速度'].std()).values}")
        # 取得した統計量
        print(f"mu w/ ped: {arr_mu}")
        print(f"sgm w/ ped: {arr_sgm}")
    elif args.model[0] == 5:
        ## 正規分布 w/ ベイズ推定（分散は既知と仮定）
        # 馬のIDリストを取得
        unique_ids = df['ID'].unique()
        Numa = len(unique_ids)
        # sgm は既知と仮定し、コマンド引数より取得
        sgm = args.parameter[0]
        ## ハイパーパラメータをコマンド引数より取得
        m = args.parameter[1]
        sgm_mu = args.parameter[2]
        ## 事後分布のパラメータを計算
        df_groupedsum = df.groupby('ID', sort=False)['速度'].count()
        print(f"count: {df_groupedsum.values}")
        arr_sgm_mu = 1/(np.sqrt(1/sgm_mu**2 + df_groupedsum.values/sgm**2))
        df_groupedsum = df.groupby('ID', sort=False)['速度'].sum()
        print(f"sum: {df_groupedsum.values}")
        arr_m = (arr_sgm_mu/sgm_mu)**2*m + (arr_sgm_mu/sgm)**2*df_groupedsum.values
        ## 予測分布のパラメータを計算
        arr_mu = arr_m
        arr_sgm = sgm*np.ones(Numa) + arr_sgm_mu
        # テーブルの統計量
        print(f"mu from table: {(df.groupby('ID', sort=False)['速度'].mean()).values}")
        print(f"sgm from table: {(df.groupby('ID', sort=False)['速度'].std()).values}")
        # 取得した統計量
        print(f"mu: {arr_mu}")
        print(f"sgm: {arr_sgm}")
    else:
        print("model construction is not supported.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done. elapsed time: {elapsed_time}\n")

    ## 描画
    start_time = time.time()
    if args.model[0] >= 3:
        print("start p(v) plotting")
        label = "mean velocity [m/s]", "PDF [s/m]"
        # 描画範囲を設定
        vmin = np.min(arr_mu - 4*arr_sgm)
        vmax = np.max(arr_mu + 4*arr_sgm)
        Nv = 500
        arr_v = np.linspace(vmin, vmax, Nv)
        if args.binomen[0] == 0:
            showPDF("data/UMA_DB_EL.csv", arr_v, label)
        elif args.binomen[0] ==1:
            showPDF("data/Jockey_DB_EL.csv", arr_v, label)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done. elapsed time: {elapsed_time}\n")

    ## 期待値計算
    print("start probability integration")
    if args.model[0] >= 3:
        # 積分領域を設定
        Nv = 500
        vmin = []
        vmax = []
        dv = []
        for i in range(Numa):
            vmin.append(arr_mu[i] - 8*arr_sgm[i])
            vmax.append(arr_mu[i] + 8*arr_sgm[i])
            dv.append((vmax[i] - vmin[i])/Nv)

        ## 積分計算
        # 単勝
        start_time = time.time()
        df_result = tanshou_integration(Numa, arr_mu, arr_sgm, arr_odds, vmin, vmax, dv)
        end_time = time.time()
        print(f"elapsed time(tanshou): {end_time - start_time}\n")
        if False: # とても遅いので基本OFF
            # 単勝（数値積分）
            start_time = time.time()
            df_result = tanshou_integration_v0(Numa, arr_mu, arr_sgm, arr_odds, vmin, vmax, dv)
            end_time = time.time()
            print(f"elapsed time(tanshou_v0): {end_time - start_time}\n")
            # 三連単
            start_time = time.time()
            df_result = sanrentan_integration(Numa, arr_mu, arr_sgm, arr_odds, vmin, vmax, dv)
            end_time = time.time()
            print(f"elapsed time(sanrentan): {end_time - start_time}\n")
        # 三連単 w/ ベクトル演算
        start_time = time.time()
        df_ordered, df_unordered = sanrentan_integration_v2(Numa, arr_mu, arr_sgm, arr_odds, vmin, vmax, dv)
        end_time = time.time()
        print(f"elapsed time(sanrentan_v2): {end_time - start_time}\n")
        
        ### CSV出力
        ## 三連単
        # CSVファイルに書き込み
        fname = f"data/ordered.csv"
        df_ordered.to_csv(fname, index=False)
        # prob でソート
        fname = f"data/ordered_prob.csv"
        df_sorted = df_ordered.sort_values('当選確率', ascending=False)
        df_sorted.to_csv(fname, index=False)
        # earn でソート
        fname = f"data/ordered_earn.csv"
        df_sorted = df_ordered.sort_values('払戻金', ascending=False)
        df_sorted.to_csv(fname, index=False)
        ## 三連複
        # CSVファイルに書き込み
        fname = f"data/unordered.csv"
        df_unordered.to_csv(fname, index=False)
        # prob でソート
        fname = f"data/unordered_prob.csv"
        df_sorted = df_unordered.sort_values(by='当選確率', ascending=False)
        df_sorted.to_csv(fname, index=False)
        # earn でソート
        fname = f"data/unordered_earn.csv"
        df_sorted = df_unordered.sort_values(by='払戻金', ascending=False)
        df_sorted.to_csv(fname, index=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"done.")