#theadも取得したい
#urlを馬の名前を書いたCSVから取得したい

import requests
from bs4 import BeautifulSoup
import csv
import re

#URLを指定
url = 'https://db.netkeiba.com/horse/result/2019105219/'

#リンク先のhtmlを取得してdataに格納
req = requests.get(url)
data = BeautifulSoup(req.content, "html.parser")
#getData = data.find("ul", "topicsList_main")
#html.parser:解析モジュールの名前

#data内の<tr>要素を抽出
topics = data.find_all("table")


#topicsを文字列string_topicsに変換
string_topics = str(topics)


html1 = [part.split('</td>') for part in string_topics.split('</tr>')]

pattern1 = r'<(.*?)>'

for i in range(len(html1)):
    for j in range(len(html1[i])):
        html1[i][j] = re.sub(pattern1, '', html1[i][j])

html2 = [[cell.replace('[', '') for cell in row] for row in html1]
html3 = [[cell.replace(']', '') for cell in row] for row in html2]
html = [[cell.replace('\n', '') for cell in row] for row in html3]

print(html)


with open('web_data.csv', mode='wt', newline='', encoding = 'utf-8-sig') as f:
   csv_writer = csv.writer(f)
   csv_writer.writerows(html) 