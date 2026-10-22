from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import Entrant, PredictionRequest, RaceConditions, Run


def _condition_vector(c: RaceConditions, categories: dict[str, list[str]]) -> np.ndarray:
    values = [1.0, (c.distance_m - 1800.0) / 1000.0]
    for attr in ("venue", "surface", "track_condition"):
        current = getattr(c, attr)
        values.extend(float(current == value) for value in categories[attr])
    return np.asarray(values, dtype=float)


@dataclass
class CommonBaseline:
    categories: dict[str, list[str]]
    weights: np.ndarray
    covariance: np.ndarray
    noise_variance: float

    @classmethod
    def fit(cls, runs: list[Run]) -> CommonBaseline:
        categories = {
            key: sorted({getattr(run.conditions, key) for run in runs})
            for key in ("venue", "surface", "track_condition")
        }
        if not runs:
            size = 2
            return cls(categories, np.array([16.0, 0.0]), np.eye(size), 1.0)
        x = np.vstack([_condition_vector(run.conditions, categories) for run in runs])
        y = np.asarray([run.speed_mps for run in runs])
        ridge = 1.0
        precision = x.T @ x + ridge * np.eye(x.shape[1])
        covariance = np.linalg.inv(precision)
        weights = covariance @ x.T @ y
        residual = y - x @ weights
        noise = max(float(np.mean(residual**2)), 0.04)
        return cls(categories, weights, covariance, noise)

    def predict(self, conditions: RaceConditions) -> tuple[float, float]:
        x = _condition_vector(conditions, self.categories)
        mean = float(x @ self.weights)
        variance = float(self.noise_variance + self.noise_variance * x @ self.covariance @ x)
        return mean, max(variance, 0.04)


def _kernel(a: list[RaceConditions], b: list[RaceConditions], amplitude: float) -> np.ndarray:
    result = np.empty((len(a), len(b)))
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            distance = (left.distance_m - right.distance_m) / 600.0
            category = sum(
                getattr(left, key) != getattr(right, key)
                for key in ("venue", "surface", "track_condition")
            )
            result[i, j] = amplitude * np.exp(-0.5 * distance**2 - 0.7 * category)
    return result


def _residual_prediction(
    entrant: Entrant, target: RaceConditions, baseline: CommonBaseline
) -> tuple[float, float, str]:
    histories = entrant.history
    residuals = np.asarray(
        [run.speed_mps - baseline.predict(run.conditions)[0] for run in histories]
    )
    if len(histories) < 3:
        # Empirical-Bayes shrinkage: sparse horses borrow the population mean (zero residual).
        weight = len(histories) / (len(histories) + 3.0)
        mean = weight * float(residuals.mean()) if len(histories) else 0.0
        variance = 0.35 + (float(residuals.var()) if len(histories) > 1 else 0.35)
        return mean, variance, "fallback"

    amplitude = max(float(np.var(residuals)), 0.08)
    noise = max(amplitude * 0.15, 0.03)
    conditions = [run.conditions for run in histories]
    kernel = _kernel(conditions, conditions, amplitude) + noise * np.eye(len(histories))
    cross = _kernel(conditions, [target], amplitude)
    try:
        solved_y = np.linalg.solve(kernel, residuals)
        solved_k = np.linalg.solve(kernel, cross)
    except np.linalg.LinAlgError:
        solved_y = np.linalg.pinv(kernel) @ residuals
        solved_k = np.linalg.pinv(kernel) @ cross
    mean = (cross.T @ solved_y).item()
    variance = (amplitude + noise - cross.T @ solved_k).item()
    return mean, max(variance, 0.03), "gp"


def predict_speed_distributions(request: PredictionRequest) -> list[dict]:
    # This is the central anti-leakage boundary: future and same-time results never reach fit().
    for entrant in request.entrants:
        entrant.history = [
            run for run in entrant.history if run.occurred_at < request.prediction_at
        ]
    all_runs = [run for entrant in request.entrants for run in entrant.history]
    baseline = CommonBaseline.fit(all_runs)
    baseline_mean, baseline_var = baseline.predict(request.conditions)
    result = []
    for entrant in request.entrants:
        residual_mean, residual_var, model = _residual_prediction(
            entrant, request.conditions, baseline
        )
        result.append(
            {
                "horse_number": entrant.horse_number,
                "name": entrant.name,
                "mean": baseline_mean + residual_mean,
                "std": float(np.sqrt(baseline_var + residual_var)),
                "model": model,
                "history_count": len(entrant.history),
            }
        )
    return result
