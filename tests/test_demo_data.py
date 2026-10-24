from pathlib import Path

import pytest
from pydantic import ValidationError

from uma_api.demo_data import _read_rows, build_validation_mode
from uma_api.schemas import Entrant, PredictionRequest, RaceConditions

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize(
    ("filename", "basis"),
    [("UMA_DB.csv", "horse"), ("Jockey_DB.csv", "jockey")],
)
def test_csv_validation_mode_is_chronological_and_uses_requested_basis(filename, basis):
    mode = build_validation_mode(ROOT / "data" / filename, basis, max_races=2, samples=2_000)
    assert mode["basis"] == basis
    assert len(mode["races"]) == 2
    assert mode["races"][0]["race_label"][:10] >= mode["races"][1]["race_label"][:10]
    assert all(race["prediction_basis"] == basis for race in mode["races"])
    assert all(race["entries"] for race in mode["races"])
    if basis == "horse":
        assert all(
            not entry["name"].startswith("競走馬 ")
            for race in mode["races"]
            for entry in race["entries"]
        )
        assert all(
            entry["jockey_name"]
            for race in mode["races"]
            for entry in race["entries"]
        )


def test_jockey_basis_requires_jockey_ids():
    conditions = RaceConditions(venue="東京", surface="芝", distance_m=1600, track_condition="良")
    with pytest.raises(ValidationError, match="jockey_id"):
        PredictionRequest(
            race_id="missing-jockey",
            prediction_basis="jockey",
            prediction_at="2026-01-01T00:00:00Z",
            conditions=conditions,
            entrants=[
                Entrant(horse_id=str(index), horse_number=index, name=f"H{index}")
                for index in range(1, 4)
            ],
        )


def test_read_rows_discards_exact_duplicates(tmp_path):
    csv_path = tmp_path / "duplicates.csv"
    csv_path.write_text("ID,馬名\n1,テスト馬\n1,テスト馬\n2,別の馬\n", encoding="utf-8")

    assert _read_rows(csv_path) == [
        {"ID": "1", "馬名": "テスト馬"},
        {"ID": "2", "馬名": "別の馬"},
    ]
