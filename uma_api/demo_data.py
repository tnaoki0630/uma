from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from .schemas import Entrant, Odds, PredictionRequest, RaceConditions, Run
from .service import predict

JST = timezone(timedelta(hours=9))


def _number(value: str | None) -> float | None:
    try:
        result = float(value or "")
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def _integer(value: str | None) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y/%m/%d").replace(tzinfo=JST)


def _speed(row: dict[str, str]) -> float | None:
    distance = re.search(r"(\d+)", row.get("距離", ""))
    race_time = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", row.get("タイム", ""))
    if not distance or not race_time:
        return None
    seconds = int(race_time.group(1) or 0) * 60 + float(race_time.group(2))
    return int(distance.group(1)) / seconds if seconds > 0 else None


def _conditions(row: dict[str, str]) -> RaceConditions | None:
    distance = re.search(r"([芝ダ障])(\d+)", row.get("距離", ""))
    if not distance:
        return None
    surfaces = {"芝": "芝", "ダ": "ダート", "障": "障害"}
    venue = re.sub(r"[\d.]+", "", row.get("開催", ""))
    if not venue:
        return None
    return RaceConditions(
        venue=venue,
        surface=surfaces[distance.group(1)],
        distance_m=int(distance.group(2)),
        track_condition=row.get("馬 場", "") or "不明",
    )


def _run(row: dict[str, str]) -> Run | None:
    speed = _speed(row)
    conditions = _conditions(row)
    if speed is None or conditions is None:
        return None
    return Run(occurred_at=_date(row["日付"]), conditions=conditions, speed_mps=speed)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    # Scraping can append an already collected result when a request is retried.
    # Preserve the source order while preventing duplicate histories/entrants.
    unique_rows = []
    seen = set()
    for row in rows:
        fingerprint = tuple(row.items())
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_rows.append(row)
    return unique_rows


def _race_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["日付"], row["開催"], row["R"]


def _stable_numbers(value):
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, list):
        return [_stable_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _stable_numbers(item) for key, item in value.items()}
    return value


def build_validation_mode(
    csv_path: Path, basis: str, max_races: int = 10, samples: int = 15_000
) -> dict:
    rows = _read_rows(csv_path)
    history_by_subject: dict[str, list[tuple[datetime, Run]]] = defaultdict(list)
    for row in rows:
        run = _run(row)
        if run:
            history_by_subject[row["ID"]].append((run.occurred_at, run))
    for history in history_by_subject.values():
        history.sort(key=lambda item: item[0])

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_race_key(row)].append(row)

    candidates = []
    for key, race_rows in grouped.items():
        usable = []
        for row in race_rows:
            horse_number = _integer(row.get("馬 番"))
            finish = _integer(row.get("着 順"))
            if horse_number and finish and _run(row):
                usable.append((row, horse_number, finish))
        if len({item[1] for item in usable}) >= 3 and any(item[2] == 1 for item in usable):
            candidates.append((key, usable))
    candidates.sort(
        key=lambda item: (
            _date(item[0][0]),
            len({horse_number for _, horse_number, _ in item[1]}),
            item[0][1],
            float(item[0][2]),
        ),
        reverse=True,
    )

    validation_races = []
    for key, usable in candidates:
        target_date = _date(key[0])
        conditions = _conditions(usable[0][0])
        if conditions is None:
            continue
        entrants = []
        odds = []
        actual_winner_number = None
        actual_winner_name = None
        for row, horse_number, finish in usable:
            subject_id = row["ID"]
            histories = [
                run
                for occurred_at, run in history_by_subject[subject_id]
                if occurred_at < target_date
            ][-40:]
            if basis == "horse":
                display_name = row.get("馬名") or f"競走馬 {subject_id[-4:]}"
                jockey_id = None
            else:
                display_name = row.get("馬名") or f"馬番 {horse_number}"
                jockey_id = subject_id
            jockey_name = row.get("騎手") or None
            if basis == "jockey" and not jockey_name:
                jockey_name = f"騎手 #{subject_id}"
            entrants.append(
                Entrant(
                    horse_id=subject_id if basis == "horse" else f"{key}-{horse_number}",
                    horse_number=horse_number,
                    name=display_name,
                    jockey_id=jockey_id,
                    jockey_name=jockey_name,
                    history=histories,
                )
            )
            decimal_odds = _number(row.get("オ ッ ズ") or row.get("単勝"))
            if decimal_odds and decimal_odds > 1:
                odds.append(
                    Odds(bet_type="win", selection=[horse_number], decimal_odds=decimal_odds)
                )
            if finish == 1:
                actual_winner_number = horse_number
                actual_winner_name = display_name
        if len(entrants) < 3 or actual_winner_number is None:
            continue
        race_id = "-".join(key).replace("/", "")
        request = PredictionRequest(
            race_id=race_id,
            prediction_basis=basis,
            prediction_at=datetime.combine(target_date.date(), time(15, 0), tzinfo=JST),
            conditions=conditions,
            entrants=entrants,
            odds=odds,
            samples=samples,
            seed=2026 + len(validation_races),
        )
        result = predict(request).model_dump(mode="json")
        result["trio"] = result["trio"][:30]
        result["trifecta"] = result["trifecta"][:30]
        winner_probs = {
            entry["horse_number"]: entry["win_probability"] for entry in result["entries"]
        }
        actual_probability = winner_probs[actual_winner_number]
        brier = sum(
            (probability - float(number == actual_winner_number)) ** 2
            for number, probability in winner_probs.items()
        )
        ranked = sorted(winner_probs, key=winner_probs.get, reverse=True)
        result.update(
            {
                "race_label": f"{key[0]} {key[1]} {key[2]}R {usable[0][0].get('レース名', '')}",
                "conditions": conditions.model_dump(),
                "actual_winner_number": actual_winner_number,
                "actual_winner_name": actual_winner_name,
                "actual_winner_probability": actual_probability,
                "predicted_winner_hit": ranked[0] == actual_winner_number,
                "actual_winner_probability_rank": ranked.index(actual_winner_number) + 1,
                "brier_score": brier,
            }
        )
        validation_races.append(_stable_numbers(result))
        if len(validation_races) >= max_races:
            break
    if not validation_races:
        raise ValueError(f"No validation races could be built from {csv_path}")
    return {
        "basis": basis,
        "label": "馬基準" if basis == "horse" else "ジョッキー基準",
        "source": csv_path.name,
        "max_validation_races": len(validation_races),
        "races": validation_races,
    }


def build_demo_catalog(data_dir: Path, max_races: int = 10) -> dict:
    return {
        "generated_from": ["UMA_DB.csv", "Jockey_DB.csv"],
        "modes": {
            "horse": build_validation_mode(data_dir / "UMA_DB.csv", "horse", max_races),
            "jockey": build_validation_mode(data_dir / "Jockey_DB.csv", "jockey", max_races),
        },
    }
