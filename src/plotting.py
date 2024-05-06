import math
import argparse
import pandas as pd
from makeCSV import strtime2fltsec, extract_integer, strtime2fltsec_v2, extract_integer_v2
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
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
def showPDF_v2(df, nr, Nv, arr_label):
    # 引数からハイパーパラメータを取得
    theta = np.array(args.parameter)
    # カラーマップを取得
    cmap = plt.get_cmap("tab10")
    
    # 各IDごとに描画
    unique_ids = df['ID'].unique()
    i = 0
    arr_mu = (df.groupby('ID', sort=False)['速度'].mean()).values
    arr_sgm = (df.groupby('ID', sort=False)['速度'].std()).values
    print(arr_mu)
    print(arr_sgm)
    for id_value in unique_ids:
        fig, ax = plt.subplots(dpi=200)

        # 描画領域を設定
        vmin = np.min(arr_mu[i] - nr*arr_sgm[i])
        vmax = np.max(arr_mu[i] + nr*arr_sgm[i])
        v_plot = np.linspace(vmin, vmax, Nv)
        dv = (vmax-vmin)/(Nv-1)

        # 正規分布のプロット
        p_v = 1/np.sqrt(2*math.pi)/arr_sgm[i]*np.exp(-(v_plot-arr_mu[i])**2/(2*arr_sgm[i]**2))
        # ax.plot(v, p_v, color=cmap(i), label=f'uma #{i+1}')
        ax.plot(v_plot, p_v, label=f'data #{i+1}: Gaussian')
        # print(f" integrate of gaussian: {np.sum(p_v)*dv}")

        ## 経験分布のプロット(ノンパラメトリック推定 w/ RBFカーネル)
        subset_data = df[df['ID'] == id_value]
        v_data = subset_data['速度'].values
        # print(f'#{i+1}: v = {v_data}')
        
        # RBFカーネルの計算
        Phi = RBFkarnel(theta, v_data, v_plot, False, False)
        # v_data 方向に総和をとって正規化)
        p_v = np.sum(Phi, axis=0)/(np.sum(Phi)*dv)
        # ax.plot(v_plot, prob, '--', color=cmap(i), label=f'uma #{i+1}: empirical') # 正規分布と同じ色
        ax.plot(v_plot, p_v, '-', label=f'data #{i+1}: empirical') # 正規分布と別の色
        # print(f" integrate of empirical: {np.sum(p_v)*dv}")

        ## 元データのプロット
        p_v = [0.5*rnd.random() for iter in range(len(v_data))]
        # ax.scatter(v_data, p_v, s=10, color=cmap(i), marker='.', alpha=0.5)
        plt.scatter(v_data, p_v, s=10, c=np.arange(len(v_data)), cmap='viridis', marker='.') # 時系列でデータを色分け
        
        ## 画像保存
        plt.colorbar(label='Index')
        ax.set_xlabel(arr_label[0])
        ax.set_ylabel(arr_label[1])
        ax.legend(loc='upper right')
        ax.grid()
        ax.set_title(f'parameter: theta[1] = {theta[1]}')
        plt.savefig(f'fig/PDF_{arr_label[2]}_#{i+1}.png')
        plt.clf()

        ## ループ番号を進める
        i+=1

    # まとめて描画
    i = 0
    # カラーマップを取得
    cmap = plt.get_cmap("tab20")
    # fig, ax = plt.subplots(dpi=200) # 1枚のpngに対して1つの図を作成
    fig, ax = plt.subplots(dpi=200, nrows=1, ncols=2, figsize=(11, 4)) # 1枚のpngに対して2つの図を作成
    for id_value in unique_ids:
        # 描画領域を設定
        vmin = np.min(arr_mu[i] - 6*arr_sgm[i])
        vmax = np.max(arr_mu[i] + 6*arr_sgm[i])
        v_plot = np.linspace(vmin, vmax, Nv)
        dv = (vmax-vmin)/(Nv-1)

        # 正規分布のプロット
        p_v = 1/np.sqrt(2*math.pi)/arr_sgm[i]*np.exp(-(v_plot-arr_mu[i])**2/(2*arr_sgm[i]**2))
        # ax.plot(v_plot, p_v, color=cmap(i), label=f'data #{i+1}: Gaussian')
        ax[0].plot(v_plot, p_v, color=cmap(i), label=f'data #{i+1}', linewidth=1)
        # print(f" integrate of gaussian: {np.sum(p_v)*dv}")

        ## 経験分布のプロット(ノンパラメトリック推定 w/ RBFカーネル)
        subset_data = df[df['ID'] == id_value]
        v_data = subset_data['速度'].values
        # RBFカーネルの計算
        Phi = RBFkarnel(theta, v_data, v_plot, False, False)
        # v_data 方向に総和をとって正規化)
        p_v = np.sum(Phi, axis=0)/(np.sum(Phi)*dv)
        # ax.plot(v_plot, p_v, '--', color=cmap(i), label=f'data #{i+1}: empirical') # Gaussian に重ねてプロット
        ax[1].plot(v_plot, p_v, color=cmap(i), label=f'data #{i+1}', linewidth=1) # 別の図にプロット
        # print(f" integrate of empirical: {np.sum(p_v)*dv}")

        ## 元データのプロット
        # p_v = [0.5*rnd.random() for iter in range(len(v_data))]
        # ax.scatter(v_data, p_v, s=10, color=cmap(i), marker='.', alpha=0.5)

        ## ループ番号を進める
        i+=1
    
    ## 画像保存
    ax[0].set_xlabel(arr_label[0])
    ax[0].set_ylabel(arr_label[1])
    ax[0].grid()
    ax[0].set_title(f'Gaussian')
    ax[1].set_xlabel(arr_label[0])
    ax[1].set_ylabel(arr_label[1])
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax[1].grid()
    ax[1].set_title(f'empirical (parameter: theta[1] = {theta[1]})')
    # レイアウトの調整
    plt.tight_layout()
    plt.savefig(f'fig/PDF_{arr_label[2]}.png')
    plt.clf()

###

## main
if __name__ == "__main__":
    ## コマンド引数関連
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ## 使用するデータの種類
    parser.add_argument("-t", "--type", nargs=1, type=int, choices=range(3), help="\
                        TYPE = 0: EntryList (default) \n\
                        TYPE = 1: DataBase\n\
                        TYPE = 2: Jockey\n\
                        ", default=[0])
    ## ハイパーパラメータ
    parser.add_argument("-p", "--parameter", nargs=3, type=float, help="\
                        PARAMETER[0] \\in R (default:  0.0)\n\
                        PARAMETER[1] \\in R (default: -3.0)\n\
                        PARAMETER[2] \\in R (default:  0.0)\n\
                        ", default=[0, -3, 0])
    ## テーブル数の上限
    parser.add_argument("-u", "--upperbound", nargs=1, type=int, help="\
                        UPPERBOUND \\in Z (default: 20)\n\
                        (UPPERBOUND = -1: all)\n\
                        ", default=[20])
    ## 引数の取得
    args = parser.parse_args()

    ## ファイル名
    if(args.type[0] == 0):
        name = "UMA_DB_EL"
    elif(args.type[0] == 1):
        name = "UMA_DB"
    elif(args.type[0] == 2):
        name = "Jockey_DB_EL"

    ## 戦績
    df = pd.read_csv(name+".csv")
    ## データフレームを使用する変数だけに削減
    # varlist = ['ID', '頭 数', '馬 番', 'オ ッ ズ', '着 順', '斤 量', '上り', '馬体重', 'タイム', '距離']
    varlist = ['ID', 'タイム', '距離']
    df = df[varlist]
    ## NaNを削除
    df = df.dropna()
    ## 文字列変換
    df['タイム'] = df['タイム'].apply(strtime2fltsec_v2)
    df['距離'] = df['距離'].apply(extract_integer_v2)
    ## 平均速度の追加
    df['速度'] = df['距離'] / df['タイム']
    ## IDと平均速度だけ使用
    df = df[['ID', '速度']]
    ## たまに文字列処理の例外が出るので、改めてNaNを削除
    df = df.dropna()

    ## 描画(分散で正規化した領域を指定)
    norm_range = 6
    Nv = 500
    label = "mean velocity [m/s]", "PDF [s/m]", name
    showPDF_v2(df, norm_range, Nv, label)
