import requests
from bs4 import BeautifulSoup
import csv

###
def name2id(name):
    # 事前に登録された文字列とIDの対応を定義
    name_to_id_mapping = {
        "うま": 1,
        # 他にも必要な文字列と対応するIDを追加
        "ブレイディヴェーグ": 2020103242,
        "ジェラルディーナ": 2018105081,
        "ルージュエヴァイユ": 2019104831,
        "ライラック": 2019103588,
        "ハーパー": 2020103458,
        "ディヴィーナ": 2018104947,
        "サリエラ": 2019105207,
        "アートハウス": 2019104074,
        "マリアエレーナ": 2018100927,
        "ゴールドエクリプス": 2019100824,
        "ビッグリボン": 2018100427,
        "シンリョクカ": 2020101608,
        "イズジョーノキセキ": 2017102294,
        "ククナ": 2018105008,
        "ジャンタルマンタル": 2021107171,
        "ドライスタウト": 2019101667,
        "イクイノックス": 2019105219,
        "ダノンキラウェア": 2021100171,
        "ヤマニンウルス": 2020105599,
        "カンティアーモ": 2021105860,
        "ペリエール": 2020101874,
        "セリフォス": 2019104462,
        "ローゼライト": 2018102715,
        "アルセナール": 2021105548,
        "シュネルマイスター": 2018110007,
        "レッドルゼル": 2016105188,
        "ペースセッティング": 2020110007,
        "フルレゾン": 2021104840,
        "リバティアイランド": 2020103656,
        "ナミュール": 2019105213,
        "エルトンバローズ": 2020100680,
        "ステラヴェローチェ": 2018104980,
        "ソウルラッシュ": 2018100382,
        "クイーンズウォーク": 2021105414,
        "サトノダイヤモンド": 2013106101,
        "ガイアメンテ": 2021105808,
        "タガノビューティー": 2017101719,
        "シルトホルン": 2020105424,
        "ホウオウエミーズ": 2017104691,
        "エヴァンジェリーナ": 2021105586
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

name = "ジェラルディーナ"
id = name2id(name)
if id:
    url = f"https://db.netkeiba.com/horse/result/{id}"
else:
    # エラー終了
    print("Error at name2id")
    exit()

# スクレイピング
grades_data = scrape_grades(url)

# 結果
if grades_data:
    # 一部を標準出力
    for row in grades_data:
        print(row[0:10:2])
    # csv出力
    with open(f"{id}.csv", mode='wt', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(grades_data) 
else:
    # エラー終了
    print("Error at scrape_grades")
    exit()