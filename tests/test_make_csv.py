import pandas as pd

from src.makeCSV import collect_jockey_results, extract_hitchart_ids


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


def test_collect_jockey_results_preserves_ids_and_applies_strict_upperbound():
    calls = []

    def fake_read_html(url, encoding):
        calls.append((url, encoding))
        page = int(url.split("page=")[-1])
        if page > 2:
            raise ValueError("end")
        return [pd.DataFrame({"日付": [f"2024/01/0{page}", f"2024/01/0{page}"], "着 順": [1, 2]})]

    result = collect_jockey_results(["01234"], upperbound=3, read_html=fake_read_html)
    assert result["ID"].tolist() == ["01234", "01234", "01234"]
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
