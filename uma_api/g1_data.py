from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .schemas import Entrant, Odds, PredictionRequest, RaceConditions, Run
from .service import predict

JST = timezone(timedelta(hours=9))

# entry_sheet.csv has missing surface/distance fields for some exports. These are
# fixed JRA G1 race definitions, so using the race name is safer than guessing
# from a runner's past race.
G1_CONDITIONS: dict[str, tuple[str, int]] = {
    "フェブラリーS": ("ダート", 1600),
    "高松宮記念": ("芝", 1200),
    "大阪杯": ("芝", 2000),
    "桜花賞": ("芝", 1600),
    "皐月賞": ("芝", 2000),
    "天皇賞(春)": ("芝", 3200),
    "NHKマイルC": ("芝", 1600),
    "ヴィクトリアマイル": ("芝", 1600),
    "優駿牝馬": ("芝", 2400),
    "東京優駿": ("芝", 2400),
    "安田記念": ("芝", 1600),
    "宝塚記念": ("芝", 2200),
    "スプリンターズS": ("芝", 1200),
    "秋華賞": ("芝", 2000),
    "菊花賞": ("芝", 3000),
    "天皇賞(秋)": ("芝", 2000),
    "エリザベス女王杯": ("芝", 2200),
    "マイルチャンピオンS": ("芝", 1600),
    "ジャパンC": ("芝", 2400),
    "チャンピオンズC": ("ダート", 1800),
    "阪神ジュベナイルF": ("芝", 1600),
    "朝日フューチュリティ": ("芝", 1600),
    "有馬記念": ("芝", 2500),
    "ホープフルS": ("芝", 2000),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _number(value: str | None) -> float | None:
    try:
        number = float((value or "").replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _integer(value: str | None) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _date(value: str | None) -> datetime | None:
    text = (value or "").strip()
    for pattern in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=JST)
        except ValueError:
            pass
    return None


def _race_time_seconds(value: str | None) -> float | None:
    match = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", (value or "").strip())
    if not match:
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2))


def _history_conditions(row: dict[str, str]) -> RaceConditions | None:
    distance = re.search(r"([芝ダ障])(\d+)", row.get("距離", ""))
    if not distance:
        return None
    surface = {"芝": "芝", "ダ": "ダート", "障": "障害"}[distance.group(1)]
    distance_m = int(distance.group(2))
    if not 800 <= distance_m <= 5000:
        return None
    venue = re.sub(r"[\d.]+", "", row.get("開催", "")).strip()
    if not venue:
        return None
    return RaceConditions(
        venue=venue,
        surface=surface,
        distance_m=distance_m,
        track_condition=row.get("馬場", "") or row.get("馬 場", "") or "不明",
    )


def _history_run(row: dict[str, str]) -> Run | None:
    occurred_at = _date(row.get("日付"))
    conditions = _history_conditions(row)
    seconds = _race_time_seconds(row.get("タイム"))
    if occurred_at is None or conditions is None or seconds is None or seconds <= 0:
        return None
    speed = conditions.distance_m / seconds
    if not 5 < speed < 30:
        return None
    return Run(occurred_at=occurred_at, conditions=conditions, speed_mps=speed)


def _target_conditions(
    entry: dict[str, str], summary: dict[str, str] | None
) -> RaceConditions:
    race_name = entry.get("race_name", "")
    fallback = G1_CONDITIONS.get(race_name)
    surface = (summary or {}).get("race_surface") or entry.get("race_surface")
    if surface == "ダ":
        surface = "ダート"
    distance = _integer((summary or {}).get("race_distance_m") or entry.get("race_distance_m"))
    if (not surface or not distance) and fallback:
        surface = surface or fallback[0]
        distance = distance or fallback[1]
    if surface not in {"芝", "ダート", "障害"} or distance is None:
        raise ValueError(f"Race conditions are unavailable for {race_name}")
    return RaceConditions(
        venue=entry.get("race_venue", "") or (summary or {}).get("race_venue", "") or "不明",
        surface=surface,
        distance_m=distance,
        track_condition=entry.get("race_going", "") or "不明",
    )


def _history_by_subject(
    path: Path,
    subject_column: str,
    prediction_at: datetime,
    history_limit: int,
) -> dict[str, list[Run]]:
    histories: dict[str, list[Run]] = defaultdict(list)
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in _read_csv(path):
        subject_id = row.get(subject_column, "").strip()
        run = _history_run(row)
        if not subject_id or run is None or run.occurred_at >= prediction_at:
            continue
        fingerprint = (
            subject_id,
            run.occurred_at.isoformat(),
            row.get("開催", ""),
            row.get("R", ""),
            row.get("距離", ""),
            row.get("タイム", ""),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        histories[subject_id].append(run)
    for subject_id, runs in histories.items():
        histories[subject_id] = sorted(runs, key=lambda run: run.occurred_at)[-history_limit:]
    return histories


def _stable_numbers(value):
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, list):
        return [_stable_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _stable_numbers(item) for key, item in value.items()}
    return value


def build_g1_validation_mode(
    dataset_dir: Path,
    basis: str,
    validation_year: int = 2025,
    max_races: int | None = None,
    samples: int = 15_000,
    history_limit: int = 40,
) -> dict:
    entries = _read_csv(dataset_dir / "entry_sheet.csv")
    results = _read_csv(dataset_dir / "result_sheet.csv")
    summary_path = dataset_dir / "processing_summary.csv"
    summaries = {
        row["target_race_id"]: row for row in _read_csv(summary_path)
    } if summary_path.exists() else {}

    result_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in results:
        result_by_race[row["target_race_id"]].append(row)
    entry_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in entries:
        target_date = _date(row.get("target_race_date"))
        if target_date and target_date.year == validation_year:
            entry_by_race[row["target_race_id"]].append(row)

    race_ids = sorted(
        entry_by_race,
        key=lambda race_id: entry_by_race[race_id][0]["target_race_date"],
        reverse=True,
    )
    validation_races = []
    for race_id in race_ids:
        race_entries = entry_by_race[race_id]
        prediction_at = _date(race_entries[0].get("target_race_date"))
        if prediction_at is None:
            continue
        winner = next(
            (
                row
                for row in result_by_race.get(race_id, [])
                if _integer(row.get("着順")) == 1
            ),
            None,
        )
        if winner is None:
            continue
        conditions = _target_conditions(race_entries[0], summaries.get(race_id))
        history_name = "horse_history.csv" if basis == "horse" else "jockey_history.csv"
        subject_column = "horse_id" if basis == "horse" else "jockey_id"
        history_path = dataset_dir / f"race_{race_id}" / history_name
        if not history_path.exists():
            continue
        history_by_subject = _history_by_subject(
            history_path, subject_column, prediction_at, history_limit
        )

        entrants = []
        odds = []
        for row in sorted(race_entries, key=lambda item: _integer(item.get("馬番")) or 999):
            horse_number = _integer(row.get("馬番"))
            horse_id = row.get("horse_id", "").strip()
            jockey_id = row.get("jockey_id", "").strip()
            subject_id = horse_id if basis == "horse" else jockey_id
            if horse_number is None or not horse_id or (basis == "jockey" and not jockey_id):
                continue
            entrants.append(
                Entrant(
                    horse_id=horse_id,
                    horse_number=horse_number,
                    name=row.get("horse_name", "") or f"馬番 {horse_number}",
                    jockey_id=jockey_id or None,
                    jockey_name=row.get("騎手", "") or None,
                    history=history_by_subject.get(subject_id, []),
                )
            )
            decimal_odds = _number(row.get("オッズ"))
            if decimal_odds and decimal_odds > 1:
                odds.append(
                    Odds(bet_type="win", selection=[horse_number], decimal_odds=decimal_odds)
                )
        if len(entrants) < 3:
            continue

        request = PredictionRequest(
            race_id=race_id,
            prediction_basis=basis,
            prediction_at=prediction_at,
            conditions=conditions,
            entrants=entrants,
            odds=odds,
            samples=samples,
            seed=validation_year + int(race_id[-4:]) + (0 if basis == "horse" else 10_000),
        )
        result = predict(request).model_dump(mode="json")
        result["trio"] = result["trio"][:30]
        result["trifecta"] = result["trifecta"][:30]
        actual_winner_number = _integer(winner.get("馬番"))
        if actual_winner_number is None:
            continue
        winner_probs = {
            entry["horse_number"]: entry["win_probability"] for entry in result["entries"]
        }
        if actual_winner_number not in winner_probs:
            continue
        ranked = sorted(winner_probs, key=winner_probs.get, reverse=True)
        result.update(
            {
                "race_label": (
                    f"{race_entries[0]['target_race_date']} "
                    f"{race_entries[0].get('race_venue', '')} "
                    f"{race_entries[0].get('race_number', '')}R "
                    f"{race_entries[0].get('race_name', '')}"
                ),
                "conditions": conditions.model_dump(),
                "actual_winner_number": actual_winner_number,
                "actual_winner_name": winner.get("horse_name", ""),
                "actual_winner_probability": winner_probs[actual_winner_number],
                "predicted_winner_hit": ranked[0] == actual_winner_number,
                "actual_winner_probability_rank": ranked.index(actual_winner_number) + 1,
                "brier_score": sum(
                    (probability - float(number == actual_winner_number)) ** 2
                    for number, probability in winner_probs.items()
                ),
            }
        )
        result["warnings"].append(
            "単勝オッズは出馬表CSVの実値ですが、取得時点が未記録のため参考値です。"
        )
        validation_races.append(_stable_numbers(result))
        if max_races is not None and len(validation_races) >= max_races:
            break

    if not validation_races:
        raise ValueError(f"No {validation_year} G1 validation races could be built")
    return {
        "basis": basis,
        "label": "馬基準" if basis == "horse" else "ジョッキー基準",
        "source": f"G1協力データ（{validation_year}年ウォークフォワード検証）",
        "max_validation_races": len(validation_races),
        "races": validation_races,
    }


def build_g1_demo_catalog(
    dataset_dir: Path,
    validation_year: int = 2025,
    max_races: int | None = None,
    samples: int = 15_000,
    history_limit: int = 40,
) -> dict:
    return {
        "generated_from": [
            "entry_sheet.csv",
            "result_sheet.csv",
            "race_<id>/horse_history.csv",
            "race_<id>/jockey_history.csv",
        ],
        "split": {
            "policy": "walk-forward",
            "reference_years": "2021-2024",
            "validation_year": validation_year,
            "history_rule": "occurred_at < target_race_date",
            "history_limit_per_subject": history_limit,
        },
        "modes": {
            "horse": build_g1_validation_mode(
                dataset_dir, "horse", validation_year, max_races, samples, history_limit
            ),
            "jockey": build_g1_validation_mode(
                dataset_dir, "jockey", validation_year, max_races, samples, history_limit
            ),
        },
    }
