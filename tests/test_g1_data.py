import csv
from datetime import datetime
from pathlib import Path

from uma_api.g1_data import JST, _history_by_subject, build_g1_demo_catalog


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _history_row(subject: str, date: str, race: str, seconds: str) -> dict[str, str]:
    return {
        "horse_id": subject,
        "jockey_id": f"0{subject}",
        "日付": date,
        "開催": "4東京2",
        "R": race,
        "距離": "芝1600",
        "馬場": "良",
        "タイム": seconds,
    }


def test_history_import_deduplicates_and_blocks_same_day_and_future(tmp_path):
    rows = [
        _history_row("1", "2025/01/01", "1", "1:34.0"),
        _history_row("1", "2025/01/01", "1", "1:34.0"),
        _history_row("1", "2025/02/01", "2", "1:35.0"),
        _history_row("1", "2025/02/02", "3", "1:36.0"),
    ]
    path = tmp_path / "history.csv"
    _write_csv(path, rows)

    histories = _history_by_subject(
        path,
        "horse_id",
        datetime(2025, 2, 1, tzinfo=JST),
        history_limit=40,
    )

    assert len(histories["1"]) == 1
    assert histories["1"][0].occurred_at < datetime(2025, 2, 1, tzinfo=JST)


def test_g1_catalog_uses_2025_holdout_and_name_based_condition_fallback(tmp_path):
    dataset = tmp_path / "g1"
    entries = []
    results = []
    horse_history = []
    jockey_history = []
    for number in range(1, 4):
        entries.append(
            {
                "target_race_id": "202505010811",
                "target_race_date": "2025-02-23",
                "race_name": "フェブラリーS",
                "horse_id": str(number),
                "horse_name": f"テスト馬{number}",
                "馬番": str(number),
                "オッズ": str(2 + number),
                "騎手": f"テスト騎手{number}",
                "jockey_id": f"0{number}",
                "race_venue": "東京",
                "race_surface": "",
                "race_distance_m": "",
                "race_going": "良",
                "race_number": "11",
            }
        )
        results.append(
            {
                **entries[-1],
                "着順": str(number),
            }
        )
        for run_index, date in enumerate(("2024/10/01", "2024/11/01", "2024/12/01"), 1):
            row = _history_row(str(number), date, str(run_index), f"1:{33 + number}.0")
            horse_history.append(row)
            jockey_history.append(row)
    _write_csv(dataset / "entry_sheet.csv", entries)
    _write_csv(dataset / "result_sheet.csv", results)
    _write_csv(dataset / "race_202505010811" / "horse_history.csv", horse_history)
    _write_csv(dataset / "race_202505010811" / "jockey_history.csv", jockey_history)

    catalog = build_g1_demo_catalog(dataset, validation_year=2025, samples=2_000)

    assert catalog["split"]["validation_year"] == 2025
    for mode in catalog["modes"].values():
        assert len(mode["races"]) == 1
        race = mode["races"][0]
        assert race["conditions"]["surface"] == "ダート"
        assert race["conditions"]["distance_m"] == 1600
        assert all(entry["history_count"] == 3 for entry in race["entries"])
