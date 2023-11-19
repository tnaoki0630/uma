import requests
from bs4 import BeautifulSoup
# import chardet
import csv

###
def name2id(name):
    # 事前に登録された文字列とIDの対応を定義
    name_to_id_mapping = {
        "うま": 1,
        # 他にも必要な文字列と対応するIDを追加
        "ブレイディヴェーグ": 2020103242,
        "イクイノックス": 2019105219,
    }
    # nameが登録されているか確認し、登録されていれば対応するIDを返す
    if name in name_to_id_mapping:
        return name_to_id_mapping[name]
    else:
        # 登録されていない場合はNULL（ここではNone）を返す
        return None
###

def scrape_grades(url):
    # HTTP GETリクエストを送信してHTMLを取得
    response = requests.get(url)

    # エラーハンドリング
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    # エンコーディングを取得
    # detected_encoding = chardet.detect(response.content)['encoding']
    ## memo: エンコーディング自動検知だとうまくいかない場合がある。GPTいわくchardetならうまくいくかもね、とのことで動作確認した。
    # print("auto:", response.encoding)
    # print("chardet:", detected_encoding)
    # exit()
    ## memo: うまくいきそう。だが実行速度を考えると直接指定した方が早いはず。なのでchardetはアンインストールした。
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
        ## memo: th がテーブルの先頭を意味するっぽい。たぶん ['hoge','fuga'] でor検索できる。
        cells = row.find_all(['th', 'td'])
        ## memo: 各セルの内容を配列に格納
        row_data = [cell.get_text(strip=True) for cell in cells]

        # 空の行は無視する
        if row_data:
            data.append(row_data)
    return data

id = name2id("ブレイディヴェーグ")
# id = name2id("イクイノックス")
url = f"https://db.netkeiba.com/horse/result/{id}"

# スクレイピング
grades_data = scrape_grades(url)

# 結果を表示
if grades_data:
    for row in grades_data:
        print(row[0:10:2])
# csv出力
with open('web_data.csv', mode='wt', newline='') as f:
   csv_writer = csv.writer(f)
   csv_writer.writerows(grades_data) 