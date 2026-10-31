from __future__ import annotations

from .model import predict_speed_distributions
from .probability import simulate_rank_probabilities
from .schemas import CombinationPrediction, EntryPrediction, PredictionRequest, PredictionResponse


def predict(request: PredictionRequest) -> PredictionResponse:
    distributions = predict_speed_distributions(request)
    win, trio, trifecta = simulate_rank_probabilities(distributions, request.samples, request.seed)
    odds = {
        (
            item.bet_type,
            tuple(sorted(item.selection)) if item.bet_type == "trio" else tuple(item.selection),
        ): item.decimal_odds
        for item in request.odds
    }

    def combinations(values: dict[tuple[int, ...], float], bet_type: str):
        rows = []
        for selection, probability in values.items():
            key = tuple(sorted(selection)) if bet_type == "trio" else selection
            decimal_odds = odds.get((bet_type, key))
            rows.append(
                CombinationPrediction(
                    selection=list(selection),
                    probability=probability,
                    decimal_odds=decimal_odds,
                    expected_return=probability * decimal_odds
                    if decimal_odds is not None
                    else None,
                )
            )
        return sorted(rows, key=lambda row: row.probability, reverse=True)

    entries = []
    for index, item in enumerate(distributions):
        decimal_odds = odds.get(("win", (item["horse_number"],)))
        entries.append(
            EntryPrediction(
                horse_number=item["horse_number"],
                name=item["name"],
                jockey_id=request.entrants[index].jockey_id,
                jockey_name=request.entrants[index].jockey_name,
                mean_speed_mps=item["mean"],
                std_speed_mps=item["std"],
                win_probability=float(win[index]),
                model=item["model"],
                history_count=item["history_count"],
                decimal_odds=decimal_odds,
                expected_return=float(win[index]) * decimal_odds
                if decimal_odds is not None
                else None,
            )
        )
    warnings = ["予測は娯楽・研究用途です。利益を保証しません。"]
    if any(entry.model == "fallback" for entry in entries):
        subject = "馬" if request.prediction_basis == "horse" else "ジョッキー"
        warnings.append(f"過去3走未満の{subject}には縮約フォールバックを使用しています。")
    if not request.odds:
        warnings.append("券種別オッズがないため期待回収率は表示しません。")
    return PredictionResponse(
        race_id=request.race_id,
        prediction_basis=request.prediction_basis,
        samples=request.samples,
        entries=entries,
        trio=combinations(trio, "trio"),
        trifecta=combinations(trifecta, "trifecta"),
        warnings=warnings,
    )
