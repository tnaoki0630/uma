from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Surface = Literal["芝", "ダート", "障害"]
PredictionBasis = Literal["horse", "jockey"]


class RaceConditions(BaseModel):
    venue: str = Field(min_length=1)
    surface: Surface
    distance_m: int = Field(ge=800, le=5000)
    track_condition: str = Field(min_length=1)


class Run(BaseModel):
    occurred_at: datetime
    conditions: RaceConditions
    speed_mps: float = Field(gt=5, lt=30)


class Entrant(BaseModel):
    horse_id: str = Field(min_length=1)
    horse_number: int = Field(ge=1, le=99)
    name: str = Field(min_length=1)
    jockey_id: str | None = None
    jockey_name: str | None = None
    history: list[Run] = Field(default_factory=list)


class Odds(BaseModel):
    bet_type: Literal["win", "trio", "trifecta"]
    selection: list[int]
    decimal_odds: float = Field(gt=1)


class PredictionRequest(BaseModel):
    race_id: str
    prediction_basis: PredictionBasis = "horse"
    prediction_at: datetime
    conditions: RaceConditions
    entrants: list[Entrant] = Field(min_length=3, max_length=18)
    odds: list[Odds] = Field(default_factory=list)
    samples: int = Field(default=50_000, ge=2_000, le=300_000)
    seed: int = 2024

    @model_validator(mode="after")
    def validate_unique_numbers_and_odds(self):
        numbers = [entrant.horse_number for entrant in self.entrants]
        if len(numbers) != len(set(numbers)):
            raise ValueError("horse_number must be unique")
        if self.prediction_basis == "jockey" and any(
            not entrant.jockey_id for entrant in self.entrants
        ):
            raise ValueError("jockey_id is required when prediction_basis is jockey")
        valid = set(numbers)
        expected = {"win": 1, "trio": 3, "trifecta": 3}
        for odds in self.odds:
            if len(odds.selection) != expected[odds.bet_type]:
                raise ValueError(f"{odds.bet_type} odds has an invalid selection length")
            if len(set(odds.selection)) != len(odds.selection) or not set(odds.selection) <= valid:
                raise ValueError("odds selection must contain distinct entered horse numbers")
        return self


class EntryPrediction(BaseModel):
    horse_number: int
    name: str
    jockey_id: str | None = None
    jockey_name: str | None = None
    mean_speed_mps: float
    std_speed_mps: float
    win_probability: float
    model: Literal["gp", "fallback"]
    history_count: int
    decimal_odds: float | None = None
    expected_return: float | None = None


class CombinationPrediction(BaseModel):
    selection: list[int]
    probability: float
    decimal_odds: float | None = None
    expected_return: float | None = None


class PredictionResponse(BaseModel):
    race_id: str
    prediction_basis: PredictionBasis
    samples: int
    entries: list[EntryPrediction]
    trio: list[CombinationPrediction]
    trifecta: list[CombinationPrediction]
    warnings: list[str]
