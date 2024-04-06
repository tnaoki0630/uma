import pandas as pd
import numpy as np
# 出力時に省略しない
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

## 事前取得したcsvからデータ取得
df_all = pd.read_csv("UMA_ped.csv")
df = pd.read_csv("UMA_ped_EL.csv")
# df1をdf2と同じ行数に拡張
for index, row in df.iterrows():
    # umaIDがrowと一致する行を探索
    matching_rows = df_all[df_all['ID'] == row['ID']]
    # 一致する行を取り除いたデータフレームを作成
    df_reduced = df_all.drop(matching_rows.index)
    # df_reducedと同じサイズになるようにrowをコピーし、Numpy配列として保持 
    row_expanded = np.array([row.values] * len(df_reduced))

    # 重みベクトル（親等と字数が対応しているためlenで判定）
    arr_weight =  np.array([2**(5-len(idx)) for idx in row.index])
    # ブール行列
    H = row_expanded == df_reduced.values
    # 血統の一致度
    arr_score = np.dot(H,arr_weight)

    ## 型チェック
    # print(type(row_expanded))
    # print(type(df_reduced.values))
    # print(type(arr_weight))
    ## 内容チェック
    # print(df_all['ID'])
    # print(df_reduced['ID'])
    # print(H)
    # print(arr_weight)
    print(np.shape(H))
    print(arr_score)