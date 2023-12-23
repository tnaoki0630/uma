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
        #print(table)
        # テーブルの各行を取得
        rows = table.find_all('tr')

        # 行ごとにデータを配列に追加
        for row in rows:
            #print(row)
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
def extract_dataset(arr_id, arr_req):
    ## initialize
    data = []
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
        
        n += 1

        ## データの追加
    return data
###
def scrape_grades_1(data, id, cols):
    # HTTP GETリクエストを送信してHTMLを取得
    url = f"https://db.netkeiba.com/horse/result/{id}"
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
    #data = []
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
    return
###


## ヒットチャートのページから馬ごとのURLを取得
N = 100
data = []
ids = []
required_data = 6, 7

ids = scrape_hitchart("https://db.netkeiba.com/?pid=horse_hitchart")
if not all(ids):
    print("Error at scrape_hitchart")
    exit()

for id in ids[:N]:
    scrape_grades_1(data, id, required_data)
    
#CSV出力
with open('UMA_DB.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)
