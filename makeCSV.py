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
            # 使うデータが揃ってる場合のみ出力
            if all(row_data[col] for col in cols):
                # 馬体重の例外処理
                if(re.search(r'\(([-+]?\d+)\)', row_data[23])):
                    data.append(row_data)
    return data
###
def extract_integer_in_parenthesis(arr):
    result = []
    for string in arr:
        # 正規表現を使って()内の整数を抜き出す
        match = re.search(r'\(([-+]?\d+)\)', string)
        if match:
            result.append(int(match.group(1)))
        else:
            result.append(None)
    return result
###
def extract_integer(arr):
    result = []
    for string in arr:
        # 正規表現を使って%dの部分を抜き出す
        match = re.search(r'\d+', string)
        if match:
            result.append(int(match.group()))
        else:
            result.append(None)
    return result
###
###
def extract_float(arr):
    result = []
    for string in arr:
        # 正規表現を使って%dの部分を抜き出す
        match = re.search(r'\d+\.\d', string)
        if match:
            result.append(float(match.group()))
        else:
            result.append(None)
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
    x3_1 = []
    x3_2 = []
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
            # print("Error at scrape_grades")
            # exit()
            continue ## 戦績表がまだない新馬もいるらしい... 既知のエラーということで続行。

        ## データベクトルの準備
        row, col = np.shape(data)
        vec_id = (n+1)*np.ones(row) # ヒットチャート順位
        vec_date = strdate2intdate(data[:, 0]) # 日付
        vec_num = extract_integer(data[:, 6]) # 頭数
        vec_order = extract_integer(data[:, 8]) # 馬番
        vec_weight_1 = extract_integer(data[:, 13]) # 斤量
        vec_dist = extract_integer(data[:, 14]) # 距離
        vec_time = strtime2fltsec(data[:, 17]) # タイム
        vec_weight_2 = extract_integer_in_parenthesis(data[:, 23]) # 馬体重の変化量
        vec_race = extract_integer(data[:, 4]) # レース名（数字のみを抽出:G1→1）
        vec_race_num = extract_integer(data[:, 3]) # レース番号
        vec_odds = np.log(extract_float(data[:, 9])) # オッズ（必ず正の値を取るので対数）
        ## データの追加
        Y = np.append(Y, vec_time)
        x1 = np.append(x1, vec_id)
        x2 = np.append(x2, vec_date)
        x3_1 = np.append(x3_1, vec_order)
        x3_2 = np.append(x3_2, vec_num)
        x4 = np.append(x4, vec_weight_1)
        x5 = np.append(x5, vec_dist)
        x6 = np.append(x6, vec_weight_2)
        x7 = np.append(x7, vec_odds)
        ## インクリメント
        n += 1
    x0 = np.ones_like(x1) ## バイアス項
    Y = x5/Y # 距離/タイム
    x3 = x3_1/x3_2 # 馬番/頭数
    ## normalize
    mean_data = np.array([Y.mean(), x1.mean(), x2.mean(), x3.mean(), x4.mean(), x5.mean(), x6.mean(), x7.mean()])
    var_data = np.array([Y.var(), x1.var(), x2.var(), x3.var(), x4.var(), x5.var(), x6.var(), x7.var()])
    Y = (Y - mean_data[0])/np.sqrt(var_data[0])
    x1 = (x1 - mean_data[1])/np.sqrt(var_data[1])
    x2 = (x2 - mean_data[2])/np.sqrt(var_data[2])
    x3 = (x3 - mean_data[3])/np.sqrt(var_data[3])
    x4 = (x4 - mean_data[4])/np.sqrt(var_data[4])
    x5 = (x5 - mean_data[5])/np.sqrt(var_data[5])
    x6 = (x6 - mean_data[6])/np.sqrt(var_data[6])
    x7 = (x7 - mean_data[7])/np.sqrt(var_data[7])
    # x8 = (x8 - mean_data[8])/np.sqrt(var_data[8])
    # x9 = (x9 - mean_data[9])/np.sqrt(var_data[9])
    ## reshape
    # X = x6
    # X = np.transpose(np.vstack((x0, x1, x2, x3))) # 重回帰
    # X = np.transpose(np.vstack((x0, x5))) # 重回帰を単回帰と比較
    # X = np.transpose(np.vstack((x5, x6))) # 距離、体重
    X = np.transpose(np.vstack((x3, x4, x6, x7))) # 馬番/頭数、斤量、体重、オッズ
    # Z = np.transpose(np.vstack((Y, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)))
    return X, Y, mean_data, var_data
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
    # 変数リスト
    varlist = ['頭 数', '馬 番', 'オ ッ ズ', '着 順', '斤 量', '上り', '馬体重']
    varlist_wID = varlist[:]
    varlist_wID.insert(0, 'ID')

    ## 1つめはヘッダー込みでデータフレームを作成
    url = f"https://db.netkeiba.com/horse/result/{ids[0]}"
    # pandasでスクレイピング
    df_all = pd.read_html(url, encoding='euc-jp')[0]
    df = df_all[varlist]
    # うまIDを追加
    df_all['ID'] = ids[0]
    df['ID'] = ids[0]
    # 要素を入れ替え
    df = df[varlist_wID]
    for id in ids[1:N]:
        url = f"https://db.netkeiba.com/horse/result/{id}"
        # pandasでスクレイピング
        try:
            df1_all = pd.read_html(url, encoding='euc-jp')[0]
        except ValueError:
            print("Table not found")
            df1 = df1_all[varlist]
            # うまIDを追加
            df1_all['ID'] = id
            df1['ID'] = id
            # 要素を入れ替え
            df1 = df1[varlist_wID]
            # データフレームを結合
            df_all = pd.concat([df_all, df1_all], ignore_index=True)
            df = pd.concat([df, df1], ignore_index=True)

    #CSV出力
    with open('UMA_DB.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

    # Z_data = data[2]
    # #訓練用データと検証用データに分ける
    # #検証用データとするレースの日付(yyyymmdd)とレース番号で入力
    # threshold_date = 20231126
    # race_number = 12

    # #訓練用データ
    # training_data = Z_data[Z_data[:, 3] < threshold_date]
    # print(training_data)

    # #検証用データ
    # validation_data = Z_data[Z_data[:, 3] == threshold_date]
    # validation_data = validation_data[validation_data[:, 9] == 1] #同日開催のレース名に1が入るレースを抽出 ex.G1, 1勝クラス
    # validation_data = validation_data[validation_data[:, 10] == race_number] #G1のみを抽出するためにレース番号を指定
    # print(validation_data)

    # #訓練用データのCSV出力
    # with open('training_data.csv', 'w', newline='', ) as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(training_data)

    # #検証用データのCSV出力
    # with open('validation_data.csv', 'w', newline='', ) as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(validation_data)
