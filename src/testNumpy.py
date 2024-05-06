import numpy as np

# # test for cdist
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

# test for broadcast
Ni = 6
Nj = 11
vi = np.linspace(0, 10, Ni)
vj = np.linspace(-20, 20, Nj)
vk = np.linspace(0, 20, Ni)
condition = vi[np.newaxis, :] < vj[:, np.newaxis]
print(vi)
print(vj)
print(vk)
print(condition)
print(np.dot(condition, vk))