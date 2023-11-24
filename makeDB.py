import requests
from bs4 import BeautifulSoup
import csv
import numpy as np
import re
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
###
def scrape_hitchart(url_hitchart):
    # HTTP GETリクエストを送信してHTMLを取得
    response = requests.get(url_hitchart)

    # エラーハンドリング
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    
    # エンコーディングを取得
    # detected_encoding = chardet.detect(response.content)['encoding']
    detected_encoding = 'EUC-JP'

    # BeautifulSoupを使ってHTMLを解析
    soup = BeautifulSoup(response.content, 'html.parser', from_encoding=detected_encoding)

    # 成績表のデータを取得（この例では表のデータがtableタグ内にあると仮定）
    table = soup.find('table')

    # テーブルが見つからない場合
    if table is None:
        print("Table not found")
        return None

    # テーブルの各行を取得
    rows = table.find_all('tr')

    # 行ごとにデータを配列に追加
    arr = []
    for row in rows:
        ## 各セルのデータを取得
        cells = row.find_all('td')
        # row_data = [cell.get_text(strip=True) for cell in cells]
        ## <a> タグの href 属性の値を取得
        for cell in cells:
            ## 空の行は無視する
            if cell:
                anchor_tag = cell.find('a')
                if anchor_tag:
                    href_value = anchor_tag.get('href')
                    # IDを抽出
                    match = re.search(r'\d+', href_value)
                    if match:
                        arr.append(int(match.group()))
    return arr
###
def scrape_grades(url, cols):
    # HTTP GETリクエストを送信してHTMLを取得
    response = requests.get(url)

    # エラーハンドリング
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    # エンコーディングを取得
    # detected_encoding = chardet.detect(response.content)['encoding']
    detected_encoding = 'EUC-JP'

    # BeautifulSoupを使ってHTMLを解析
    soup = BeautifulSoup(response.content, 'html.parser', from_encoding=detected_encoding)

    # 成績表のデータを取得（この例では表のデータがtableタグ内にあると仮定）
    table = soup.find('table')

    # テーブルが見つからない場合
    if table is None:
        print("Table not found")
        return None

    # テーブルの各行を取得
    rows = table.find_all('tr')

    # 行ごとにデータを配列に追加
    data = []
    for row in rows:
        # 各セルのデータを取得
        # cells = row.find_all(['th', 'td'])    ## テーブルの先頭を読み込みたいとき
        cells = row.find_all('td')              ## テーブルの先頭が不要なとき
        row_data = [cell.get_text(strip=True) for cell in cells]
        # 空の行は無視する
        if row_data:
            # 使うデータが欠けていれば無視する
            if all(row_data[col] for col in cols):
                data.append(row_data)
    return data
###
def extract_integer(arr):
    result = []
    for string in arr:
        # 正規表現を使って%dの部分を抜き出す
        match = re.search(r'\d+', string)
        if match:
            result.append(int(match.group()))
        else:
            # マッチしなかった場合の処理（例: エラーハンドリング）
            result.append(None)  # または何か適切な処理を行う
    return result
###
def strtime2fltsec(arr):
    result = []
    for time_str in arr:
        # ":" で文字列を分割して分と秒に分ける
        minutes, seconds = map(float, time_str.split(':'))
        
        # 分を秒に変換して合算
        total_seconds = minutes * 60 + seconds
        
        result.append(total_seconds)
    return result
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

## ヒットチャートのページから馬ごとのURLを取得
ids = scrape_hitchart("https://db.netkeiba.com/?pid=horse_hitchart")
if not all(ids):
    print("Error at scrape_hitchart")
    exit()

## N 位まで学習に使用
N = 3
required_data = 6, 7, 14, 17 # 頭数, 枠番, 距離, タイム
X = []
Y = []
for n in range(N):
    url = f"https://db.netkeiba.com/horse/result/{ids[n]}"
    ## URL先の成績表をNumPy配列として取得
    grades_data = scrape_grades(url, required_data)
    if grades_data:
        data = np.array(grades_data)
    else:
        print("Error at scrape_grades")
        exit()

    ## データベクトルの準備
    row, col = np.shape(data)
    vec_id = ids[n]*np.ones(row, np.uint8) # うまID
    vec_dist = extract_integer(data[:, 14]) # 距離
    vec_order = np.int32(data[:, 7]) / np.int32(data[:, 6]) # 枠番/馬数
    vec_time = strtime2fltsec(data[:, 17]) # タイム
    # print(ids[n],vec_time[0:3])
    X = np.append(X, vec_dist)
    Y = np.append(Y, vec_time)

## 回帰モデルの構築
# hat_Y = Regression(hat_X) として使いたい。ここの書き方は回帰モデル依存？
# 線形回帰の場合に限定すると、hat_Y = np.dot(hat_X,W)と書けばいいのでWを求めればいい。
W = Polynomial.fit(X, Y, 1)
W.convert().coef
x = np.linspace(1000, 2500, 100)
# x = np.linspace(0, 1, 100)
y_hat = W(x)
label = "dist[m]", "time[s]"
# label = "order/horses[-]", "time[s]"
show_graph(X, Y, x, y_hat, label)