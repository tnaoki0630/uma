import requests
from bs4 import BeautifulSoup
import csv
import re
import argparse
import time
from datetime import date
from io import StringIO
import pandas as pd
from pandas import plotting

JOCKEY_HITCHART_URL = (
    "https://db.sp.netkeiba.com/"
    "?pid=api_get_leading&input=UTF-8&output=jsonp&limit=100"
    f"&category=jockey&page=1&show=leading_detail&bel=&year={date.today().year}&sort_key=win"
)
HORSE_HITCHART_URL = (
    "https://db.sp.netkeiba.com/"
    "?pid=api_get_horse_ranking3&input=UTF-8&output=jsonp"
    "&limit=100&page=1&hr=ninki&sort=daily"
)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; UmaResearch/1.0)",
    "Accept-Language": "ja,en;q=0.8",
}


###
def extract_hitchart_ids(html, entity_type="horse"):
    """Extract unique horse or jockey IDs from a ranking page, preserving order."""
    if entity_type not in {"horse", "jockey"}:
        raise ValueError("entity_type must be 'horse' or 'jockey'")
    if isinstance(html, bytes):
        soup = BeautifulSoup(html, 'html.parser', from_encoding='EUC-JP')
    else:
        soup = BeautifulSoup(html, 'html.parser')
    patterns = {
        "horse": [r'/horse/(?:result/)?(\d{10})(?:/|$)', r'\b(\d{10})\b'],
        "jockey": [
            r'/jockey/(?:result/recent/)?(\d{5})(?:/|$)',
            r'[?&]id=(\d{5})(?:&|$)',
        ],
    }
    ids = []
    seen = set()
    for anchor in soup.find_all('a', href=True):
        href = anchor.get('href', '')
        match = next((re.search(pattern, href) for pattern in patterns[entity_type]
                      if re.search(pattern, href)), None)
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            ids.append(match.group(1))
    # The current horse ranking endpoint returns HTML inside a JSONP string, so
    # links are escaped and are not represented as anchor nodes until JavaScript runs.
    raw_html = html.decode("utf-8", errors="replace") if isinstance(html, bytes) else str(html)
    normalized_html = raw_html.replace(r"\/", "/")
    for pattern in patterns[entity_type]:
        for value in re.findall(pattern, normalized_html):
            if value not in seen:
                seen.add(value)
                ids.append(value)
    return [int(value) for value in ids] if entity_type == "horse" else ids


def scrape_hitchart(url, entity_type="horse"):
    # HTTP GETリクエストを送信してHTMLを取得
    response = requests.get(url, headers=HTTP_HEADERS, timeout=30)

    # エラーハンドリング
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    ids = extract_hitchart_ids(response.content, entity_type)
    if not ids:
        print("Entity links not found")
        return None
    return ids


def read_result_table(url):
    """Fetch the first result table using the headers required by netkeiba."""
    response = requests.get(url, headers=HTTP_HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = "EUC-JP"
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError(f"Result table not found: {url}")
    frame = tables[0]
    horse_name = extract_horse_name(response.text)
    if horse_name and "/horse/result/" in url:
        frame["馬名"] = horse_name
    return frame


def extract_horse_name(html):
    """Extract a horse's display name from a horse result page."""
    soup = BeautifulSoup(html, "html.parser")
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
        for marker in ("の競走成績", "のプロフィール", " | 競走馬データ"):
            if marker in title:
                name = title.split(marker, 1)[0].strip()
                return re.sub(r"\s+\([^)]*\)$", "", name)
    headings = soup.select(".db_head_name h1, h1")
    for heading in headings:
        name = heading.get_text(" ", strip=True).split("(", 1)[0].strip()
        if name:
            return name
    return None


def extract_jockey_name(html):
    """Extract a jockey's display name from a jockey result page."""
    soup = BeautifulSoup(html, "html.parser")
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
        if "の騎手成績" in title:
            return title.split("の騎手成績", 1)[0].strip()
    headings = soup.select(".db_head_name h1, h1")
    for heading in headings:
        name = heading.get_text(" ", strip=True).split("(", 1)[0].strip()
        if name:
            return name
    return None


def collect_jockey_results(
    ids,
    upperbound=1000,
    read_html=None,
    jockey_names=None,
    request_interval=1.0,
):
    """Collect recent-result tables for jockey IDs with pagination safeguards."""
    jockey_names = jockey_names or {}
    jockey_frames = []
    request_count = 0
    for jockey_id in ids:
        jockey_name = jockey_names.get(str(jockey_id).zfill(5))
        pages = []
        seen_pages = set()
        page = 1
        while True:
            url = f"https://db.netkeiba.com/jockey/result/recent/{jockey_id}/?page={page}"
            try:
                if read_html is None:
                    if request_count and request_interval > 0:
                        time.sleep(request_interval)
                    response = requests.get(url, headers=HTTP_HEADERS, timeout=30)
                    request_count += 1
                    if response.status_code == 400:
                        raise RuntimeError(
                            "netkeiba returned HTTP 400. Access may be temporarily restricted; "
                            "stop requests and retry later."
                        )
                    response.raise_for_status()
                    response.encoding = 'EUC-JP'
                    jockey_name = jockey_name or extract_jockey_name(response.text)
                    tables = pd.read_html(StringIO(response.text))
                else:
                    tables = read_html(url, encoding='euc-jp')
                frame = tables[0]
            except (requests.RequestException, ValueError, IndexError):
                break
            if frame.empty:
                break
            fingerprint = tuple(map(str, frame.iloc[0].tolist()))
            if fingerprint in seen_pages:
                print(f"Warning: repeated page for jockey {jockey_id}; stop at page {page}")
                break
            seen_pages.add(fingerprint)
            frame.insert(0, 'ID', str(jockey_id).zfill(5))
            if jockey_name:
                frame['騎手'] = jockey_name
            elif '騎手' not in frame.columns:
                frame['騎手'] = pd.NA
            pages.append(frame)
            if upperbound > 0 and sum(len(item) for item in pages) >= upperbound:
                break
            page += 1
        if not pages:
            print(f"Warning: no result table for jockey {jockey_id}")
            continue
        jockey_frame = pd.concat(pages, ignore_index=True)
        if upperbound > 0:
            jockey_frame = jockey_frame.head(upperbound)
        jockey_frames.append(jockey_frame)
    if not jockey_frames:
        raise RuntimeError("No jockey result tables were collected")
    return pd.concat(jockey_frames, ignore_index=True)
###
def scrape_entrylist(url, BOOL_Jockey=False):
    # HTTP GETリクエストを送信してHTMLを取得
    response = requests.get(url, headers=HTTP_HEADERS, timeout=30)

    # エラーハンドリング
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    # エンコーディングを取得
    # detected_encoding = chardet.detect(response.content)['encoding']
    detected_encoding = 'EUC-JP'

    # BeautifulSoupを使ってHTMLを解析
    soup = BeautifulSoup(response.content, 'html.parser', from_encoding = detected_encoding)

    # 出馬表のデータを取得（この例では出馬表がページ内で１番目の表であると仮定）
    table = soup.find('table')

    # テーブルが見つからない場合
    if not table:
        print("Table not found1")
        return None

    # initialize
    arr = []
    # loop for table
    # テーブルの各行を取得
    rows = table.find_all('tr')
    # 行ごとにデータを配列に追加
    for row in rows:
        ## 各セルのデータを取得，馬のセルのみを取得（騎手情報を取得したい場合はclass値を'Jockey'に変更）
        if BOOL_Jockey:
            cells = row.find_all('td', class_ = 'Jockey') # 引数でフラグ管理
        else:
            cells = row.find_all('td', class_ = 'HorseInfo')
        #row_data = [cell.get_text(strip=True) for cell in cells]
        ## <a> タグの href 属性の値を取得
        for cell in cells:
            ## 空の行は無視する
            if cell:
                anchor_tag = cell.find('a')
                if anchor_tag:
                    href_value = anchor_tag.get('href')
                    # IDを抽出
                    match = re.search(r'0*\d+', href_value)
                    if match:
                        arr.append(match.group())
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
def scrape_pedigree(url):
    # HTTP GETリクエストを送信してHTMLを取得
    response = requests.get(url, headers=HTTP_HEADERS, timeout=30)

    # エラーハンドリング
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    # エンコーディングを取得
    # detected_encoding = chardet.detect(response.content)['encoding']
    detected_encoding = 'EUC-JP'

    # BeautifulSoupを使ってHTMLを解析
    soup = BeautifulSoup(response.content, 'html.parser', from_encoding = detected_encoding)

    # 出馬表のデータを取得（この例では出馬表がページ内で１番目の表であると仮定）
    table = soup.find('table')

    # テーブルが見つからない場合
    if not table:
        print("Table not found1")
        return None

    # テーブルの各行を取得
    rows = table.find_all('tr')

    # 行ごとにデータを配列に追加
    arr = []
    for row in rows:
        # 各セルのデータを取得
        cells = row.find_all('td')
        for cell in cells:
            # 馬名を抽出
            names = cell.find('a') ## 最初に現れる <a> に馬名が入る
            row_data = [name.get_text(strip=True) for name in names]
            # 空の行は無視する
            if row_data:
                arr.append(row_data[0])
    return arr
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
def extract_integer_v2(str):
    # 入出力1つvar
    # 正規表現を使って%dの部分を抜き出す
    match = re.search(r'\d+', str)
    if match:
        result = int(match.group())
    else:
        print(f"Warning: extract_integer_v2 failed! str = {str}")
        result = None
    return result
###
def extract_float(arr):
    result = []
    for string in arr:
        # 正規表現を使って%dの部分を抜き出す
        print(string)
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
def strtime2fltsec_v2(str):
    # 入出力1つvar
    # ":" で文字列を分割して分と秒に分ける
    try:
        minutes, seconds = map(float, str.split(':'))
    except:
        # 例外処理
        print(f"Warning: strtime2fltsec_v2 failed! str = {str}")
        return None
    # 分を秒に変換して合算
    total_seconds = minutes * 60 + seconds
    return total_seconds
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

## 直接実行された場合のみ処理する（関数として呼び出された場合は実行しない）
if __name__ == "__main__":
    ## コマンド引数関連
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ## 作成するcsvの種類
    parser.add_argument("-t", "--type", nargs=1, type=int, choices=range(4), help="\
                        TYPE = 0: EntryList (default) \n\
                        TYPE = 1: DataBase\n\
                        TYPE = 2: Jockey from EntryList\n\
                        TYPE = 3: Jockey DataBase from Hitchart\n\
                        ", default=[0])
    ## 学習に使用するテーブル数
    parser.add_argument("-n", "--number", nargs=1, type=int, help="NUMBER \\in Z (default: 20)", default=[20])
    ## データ数の上限
    parser.add_argument("-u", "--upperbound", nargs=1, type=int, help="\
                        UPPERBOUND \\in Z (default: 1000)\n\
                        (UPPERBOUND = -1: all)\n\
                        ", default=[1000])
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds to wait between result-page requests (default: 1.0)")
    ## 引数の取得
    args = parser.parse_args()

    # EntryList
    if args.type[0] == 0:
        # エントリーリストから馬ごとのURLを取得
        url  = "https://race.netkeiba.com/race/shutuba.html?race_id=202406020711"
        ids = scrape_entrylist(url)
        if not all(ids):
            print("Error at scrape_entrylist")
            exit()

        ## 戦績表
        # 1つめはヘッダー込みでデータフレームを作成
        url = f"https://db.netkeiba.com/horse/result/{ids[0]}"
        # pandasでスクレイピング
        df_all = read_result_table(url)
        # うまIDを追加
        varlist_all = list(df_all)
        varlist_all.insert(0, 'ID')
        df_all['ID'] = ids[0]
        df_all = df_all[varlist_all]
        for id in ids[1:]:
            url = f"https://db.netkeiba.com/horse/result/{id}"
            # pandasでスクレイピング
            try:
                df1_all = read_result_table(url)
            except (requests.RequestException, ValueError, IndexError):
                print("Table not found")
                continue
            # うまIDを追加
            df1_all['ID'] = id
            # 要素を入れ替え
            df1_all = df1_all[varlist_all]
            # データフレームを結合
            df_all = pd.concat([df_all, df1_all], ignore_index=True)
        #CSV出力
        df_all.to_csv('data/UMA_DB_EL.csv', index = False)

        ## 血統情報
        arr_ped = []
        for id in ids:
            url = f"https://db.netkeiba.com/horse/ped/{id}"
            # BeautifulSoupでスクレイピング
            ped = scrape_pedigree(url)
            arr_ped.append(ped)
        # 列名リスト(いいやり方が思いつかなかったのでベタ書き。二分木の深さ優先探索アルゴリズムに近い気がするので、その辺の技術を流用すればもっとスッキリさせれる気がする)
        list = ['f', 'ff', 'fff', 'ffff', 'fffff', 'ffffm']
        list.extend(['fffm', 'fffmf', 'fffmm'])
        list.extend(['ffm', 'ffmf', 'ffmff', 'ffmfm'])
        list.extend(['ffmm', 'ffmmf', 'ffmmm'])
        list.extend(['fm', 'fmf', 'fmff', 'fmfff', 'fmffm', 'fmfm', 'fmfmf', 'fmfmm'])
        list.extend(['fmm', 'fmmf', 'fmmff', 'fmmfm', 'fmmm', 'fmmmf', 'fmmmm'])
        list.extend(['m', 'mf', 'mff', 'mfff', 'mffff', 'mfffm'])
        list.extend(['mffm', 'mffmf', 'mffmm'])
        list.extend(['mfm', 'mfmf', 'mfmff', 'mfmfm'])
        list.extend(['mfmm', 'mfmmf', 'mfmmm'])
        list.extend(['mm', 'mmf', 'mmff', 'mmfff', 'mmffm', 'mmfm', 'mmfmf', 'mmfmm'])
        list.extend(['mmm', 'mmmf', 'mmmff', 'mmmfm', 'mmmm', 'mmmmf', 'mmmmm'])
        # pandas データフレームに変換
        df = pd.DataFrame(arr_ped, columns=list)
        # uma ID を左に追加
        df.insert(0, 'ID', ids)
        #CSV出力_
        df.to_csv('data/UMA_ped_EL.csv', index = False)

    # DataBase
    if args.type[0] == 1:
        # ヒットチャートのページから馬ごとのURLを取得
        ids = scrape_hitchart(HORSE_HITCHART_URL)
        if not all(ids):
            print("Error at scrape_hitchart")
            exit()
        # 出力するテーブル数
        N = min(100, max(1, args.number[0])) # 1~100となるよう例外処理

        ## 戦績表
        # 1つめはヘッダー込みでデータフレームを作成
        url = f"https://db.netkeiba.com/horse/result/{ids[0]}"
        # pandasでスクレイピング
        df_all = read_result_table(url)
        # うまIDを追加
        varlist_all = list(df_all)
        varlist_all.insert(0, 'ID')
        df_all['ID'] = ids[0]
        # 要素を入れ替え
        df_all = df_all[varlist_all]
        for id in ids[1:N]:
            url = f"https://db.netkeiba.com/horse/result/{id}"
            # pandasでスクレイピング
            try:
                df1_all = read_result_table(url)
            except (requests.RequestException, ValueError, IndexError):
                print("Table not found")
                continue
            # うまIDを追加
            df1_all['ID'] = id
            # 要素を入れ替え
            df1_all = df1_all[varlist_all]
            # データフレームを結合
            df_all = pd.concat([df_all, df1_all], ignore_index=True)
        # CSV出力
        df_all.to_csv('data/UMA_DB.csv', index = False)

        # 配列を初期化
        df_all = []

        ## 血統情報
        arr_ped = []
        for id in ids[:N]:
            url = f"https://db.netkeiba.com/horse/ped/{id}"
            # BeautifulSoupでスクレイピング
            ped = scrape_pedigree(url)
            arr_ped.append(ped)
        # 列名リスト(いいやり方が思いつかなかったのでベタ書き。二分木の深さ優先探索アルゴリズムに近い気がするので、その辺の技術を流用すればもっとスッキリさせれる気がする)
        list = ['f', 'ff', 'fff', 'ffff', 'fffff', 'ffffm']
        list.extend(['fffm', 'fffmf', 'fffmm'])
        list.extend(['ffm', 'ffmf', 'ffmff', 'ffmfm'])
        list.extend(['ffmm', 'ffmmf', 'ffmmm'])
        list.extend(['fm', 'fmf', 'fmff', 'fmfff', 'fmffm', 'fmfm', 'fmfmf', 'fmfmm'])
        list.extend(['fmm', 'fmmf', 'fmmff', 'fmmfm', 'fmmm', 'fmmmf', 'fmmmm'])
        list.extend(['m', 'mf', 'mff', 'mfff', 'mffff', 'mfffm'])
        list.extend(['mffm', 'mffmf', 'mffmm'])
        list.extend(['mfm', 'mfmf', 'mfmff', 'mfmfm'])
        list.extend(['mfmm', 'mfmmf', 'mfmmm'])
        list.extend(['mm', 'mmf', 'mmff', 'mmfff', 'mmffm', 'mmfm', 'mmfmf', 'mmfmm'])
        list.extend(['mmm', 'mmmf', 'mmmff', 'mmmfm', 'mmmm', 'mmmmf', 'mmmmm'])
        # pandas データフレームに変換
        df = pd.DataFrame(arr_ped, columns=list)
        # uma ID を左に追加
        df.insert(0, 'ID', ids[:N])
        #CSV出力
        df.to_csv('data/UMA_ped.csv', index = False)

    # Jockey
    if args.type[0] == 2:
        # エントリーリストから馬ごとのURLを取得
        url  = "https://race.netkeiba.com/race/shutuba.html?race_id=202406020711"
        ids = scrape_entrylist(url, True) # 第二引数が True なら騎手情報を取得
        if not all(ids):
            print("Error at scrape_entrylist")
            exit()

        df_all = collect_jockey_results(
            ids, args.upperbound[0], request_interval=max(0.0, args.interval)
        )
        # CSV出力
        df_all.to_csv('data/Jockey_DB_EL.csv', index = False)

    # Jockey DataBase from Hitchart
    if args.type[0] == 3:
        ids = scrape_hitchart(JOCKEY_HITCHART_URL, "jockey")
        if not ids:
            print("Error at scrape_hitchart for jockeys")
            exit()
        # リーディング上位から取得する騎手数
        N = min(len(ids), max(1, args.number[0]))
        df_all = collect_jockey_results(
            ids[:N], args.upperbound[0], request_interval=max(0.0, args.interval)
        )
        df_all.to_csv('data/Jockey_DB.csv', index=False)

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
