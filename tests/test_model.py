from datetime import UTC, datetime, timedelta

from uma_api.model import predict_speed_distributions
from uma_api.schemas import Entrant, PredictionRequest, RaceConditions, Run

CONDITIONS = RaceConditions(venue="東京", surface="芝", distance_m=1600, track_condition="良")


def run(days: int, speed: float):
    return Run(
        occurred_at=datetime(2025, 1, 10, tzinfo=UTC) + timedelta(days=days),
        conditions=CONDITIONS,
        speed_mps=speed,
    )


def test_future_results_are_excluded_and_sparse_history_falls_back():
    cutoff = datetime(2025, 1, 20, tzinfo=UTC)
    request = PredictionRequest(
        race_id="test",
        prediction_at=cutoff,
        conditions=CONDITIONS,
        entrants=[
            Entrant(
                horse_id="a",
                horse_number=1,
                name="A",
                history=[run(0, 17), run(1, 17.1), run(2, 17.2)],
            ),
            Entrant(horse_id="b", horse_number=2, name="B", history=[run(0, 16), run(50, 25)]),
            Entrant(horse_id="c", horse_number=3, name="C", history=[]),
        ],
    )
    result = predict_speed_distributions(request)
    assert result[0]["model"] == "gp"
    assert result[1]["history_count"] == 1
    assert result[1]["model"] == "fallback"
    assert result[2]["std"] > 0
