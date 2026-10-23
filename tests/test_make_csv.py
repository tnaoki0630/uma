import pandas as pd

from src.makeCSV import (
    collect_jockey_results,
    extract_hitchart_ids,
    extract_horse_name,
    extract_jockey_name,
)


def test_extract_hitchart_ids_handles_horses_and_jockeys_without_duplicates():
    html = b"""
    <table><tr>
      <td><a href='/horse/2021105643/'>Horse A</a></td>
      <td><a href='/horse/2021105643/'>Horse A duplicate</a></td>
      <td><a href='/jockey/result/recent/05339/'>Jockey A</a></td>
      <td><a href='/?pid=jockey_detail&id=01075&page=1'>Jockey B</a></td>
    </tr></table>
    """
    assert extract_hitchart_ids(html, "horse") == [2021105643]
    assert extract_hitchart_ids(html, "jockey") == ["05339", "01075"]


def test_extract_hitchart_ids_handles_escaped_jsonp_horse_links():
    html = r'''("<a href=\"https:\/\/db.sp.netkeiba.com\/horse\/2023107089\/\">A<\/a>"
               "<a href=\"https:\/\/db.sp.netkeiba.com\/horse\/2023107005\/\">B<\/a>")'''

    assert extract_hitchart_ids(html, "horse") == [2023107089, 2023107005]


def test_extract_hitchart_ids_handles_escaped_jsonp_jockey_links():
    html = r'''("<a href=\"https:\/\/db.sp.netkeiba.com\/jockey\/05339\/\">A<\/a>"
               "<a href=\"https:\/\/db.sp.netkeiba.com\/jockey\/01174\/\">B<\/a>")'''

    assert extract_hitchart_ids(html, "jockey") == ["05339", "01174"]


def test_extract_jockey_name_uses_result_page_title():
    html = "<html><head><title>Ｃ．ルメールの騎手成績｜競馬データベース</title></head></html>"

    assert extract_jockey_name(html) == "Ｃ．ルメール"


def test_extract_horse_name_uses_result_page_title():
    html = (
        "<html><head><title>チェルヴィニア (Cervinia)の競走成績"
        "｜競馬データベース</title></head></html>"
    )

    assert extract_horse_name(html) == "チェルヴィニア"


def test_collect_jockey_results_preserves_ids_and_applies_strict_upperbound():
    calls = []

    def fake_read_html(url, encoding):
        calls.append((url, encoding))
        page = int(url.split("page=")[-1])
        if page > 2:
            raise ValueError("end")
        return [pd.DataFrame({"日付": [f"2024/01/0{page}", f"2024/01/0{page}"], "着 順": [1, 2]})]

    result = collect_jockey_results(
        ["01234"],
        upperbound=3,
        read_html=fake_read_html,
        jockey_names={"01234": "テスト騎手"},
    )
    assert result["ID"].tolist() == ["01234", "01234", "01234"]
    assert result["騎手"].tolist() == ["テスト騎手", "テスト騎手", "テスト騎手"]
    assert len(calls) == 2
    assert calls[0][0].endswith("/01234/?page=1")


def test_collect_jockey_results_stops_if_site_repeats_last_page():
    frame = pd.DataFrame({"日付": ["2024/01/01"], "着 順": [1]})
    calls = 0

    def repeated_page(url, encoding):
        nonlocal calls
        calls += 1
        return [frame.copy()]

    result = collect_jockey_results(["00001"], upperbound=-1, read_html=repeated_page)
    assert len(result) == 1
    assert calls == 2
