import requests
from bs4 import BeautifulSoup
import csv
import numpy as np
import re
from numpy.polynomial import Polynomial

###
def scrape_hitchart(url):
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
    tables = soup.find_all('table')

    # テーブルが見つからない場合
    if not tables:
        print("Table not found")
        return None

    # initialize
    arr = []
    # loop for table
    for table in tables:
        # テーブルの各行を取得
        rows = table.find_all('tr')

        # 行ごとにデータを配列に追加
        for row in rows:
            ## 各セルのデータを取得
            cells = row.find_all('td')
            #row_data = [cell.get_text(strip=True) for cell in cells]
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
def strdate2intdate(arr):
    date = []
    for date_str in arr:
        #"/"で文字列を分割して年と月と日に分ける
        year, month, day = map(int, date_str.split('/'))

        #年月日の順に並べる
        yyyymmdd = year * 10000 + month * 100 + day

        date.append(yyyymmdd)
    return date
###
def extract_dataset(arr_id, arr_req):
    ## initialize
    X = []
    Y = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []
    x8 = []
    x9 = []
    n = 0
    ## loop for id
    for id in arr_id:
        url = f"https://db.netkeiba.com/horse/result/{id}"
        ## URL先の成績表をNumPy配列として取得
        grades_data = scrape_grades(url, arr_req)
        if grades_data:
            data = np.array(grades_data)
        else:
            print("Error at scrape_grades")
            exit()

        ## データベクトルの準備
        row, col = np.shape(data)
        vec_id = (n+1)*np.ones(row) # ヒットチャート順位
        vec_date = strdate2intdate(data[:, 0]) # 日付
        vec_num = extract_integer(data[:, 6]) # 頭数
        vec_order = extract_integer(data[:, 8]) # 馬番
        vec_weight_1 = extract_integer(data[:, 13]) # 斤量
        vec_dist = extract_integer(data[:, 14]) # 距離
        vec_time = strtime2fltsec(data[:, 17]) # タイム
        vec_weight_2 = extract_integer(data[:, 23]) # 馬体重
        vec_race = extract_integer(data[:, 4]) # レース名（数字のみを抽出:G1→1）
        vec_race_num = extract_integer(data[:, 3]) # レース番号
        ## データの追加
        Y = np.append(Y, vec_time)
        x1 = np.append(x1, vec_id)
        x2 = np.append(x2, vec_date)
        x3 = np.append(x3, vec_num)
        x4 = np.append(x4, vec_order)
        x5 = np.append(x5, vec_weight_1)
        x6 = np.append(x6, vec_dist)
        x7 = np.append(x7, vec_weight_2)
        x8 = np.append(x8, vec_race)
        x9 = np.append(x9, vec_race_num)
        ## インクリメント
        n += 1
    ## バイアス項
    x0 = np.ones_like(x1)
    X = x6 # 単回帰
    # X = np.transpose(np.vstack((x0, x1, x2, x3))) # 重回帰
    # X = np.transpose(np.vstack((x0, x6))) # 重回帰を単回帰と比較
    # X = np.transpose(np.vstack((x6, x7))) # 2入力
    Z = np.transpose(np.vstack((Y, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)))
    return X, Y, Z
###

## 直接実行された場合のみ処理する（関数として呼び出された場合は実行しない）
if __name__ == "__main__":
    ## ヒットチャートのページから馬ごとのURLを取得
    ids = scrape_hitchart("https://db.netkeiba.com/?pid=horse_hitchart")
    if not all(ids):
        print("Error at scrape_hitchart")
        exit()

    ## N 位まで学習に使用
    N = 100
    required_data = 6, 7, 14, 17 # 頭数, 枠番, 距離, タイム
    data = extract_dataset(ids[:N-1], required_data)
    if data:
        X_data = data[0]
        Y_data = data[1]
    else:
        print("Error at extract_dataset")
        exit()

    Z_data = data[2]
    #訓練用データと検証用データに分ける
    #検証用データとするレースの日付(yyyymmdd)とレース番号で入力
    threshold_date = 20231126
    race_number = 12

    #訓練用データ
    training_data = Z_data[Z_data[:, 3] < threshold_date]
    print(training_data)

    #検証用データ
    validation_data = Z_data[Z_data[:, 3] == threshold_date]
    validation_data = validation_data[validation_data[:, 9] == 1] #同日開催のレース名に1が入るレースを抽出 ex.G1, 1勝クラス
    validation_data = validation_data[validation_data[:, 10] == race_number] #G1のみを抽出するためにレース番号を指定
    print(validation_data)

    #訓練用データのCSV出力
    with open('training_data.csv', 'w', newline='', ) as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(training_data)

    #検証用データのCSV出力
    with open('validation_data.csv', 'w', newline='', ) as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(validation_data)
