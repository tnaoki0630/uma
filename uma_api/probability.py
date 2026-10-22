from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


def simulate_rank_probabilities(
    predictions: list[dict], samples: int, seed: int
) -> tuple[np.ndarray, dict[tuple[int, ...], float], dict[tuple[int, ...], float]]:
    """Draw one coherent finishing order per sample and aggregate every bet market."""
    means = np.asarray([item["mean"] for item in predictions])
    stds = np.asarray([item["std"] for item in predictions])
    numbers = np.asarray([item["horse_number"] for item in predictions])
    rng = np.random.default_rng(seed)
    speeds = rng.normal(means, stds, size=(samples, len(predictions)))
    order = np.argsort(-speeds, axis=1)
    ranked_numbers = numbers[order]
    win = np.bincount(order[:, 0], minlength=len(predictions)) / samples
    trifecta_counts = Counter(map(tuple, ranked_numbers[:, :3].tolist()))
    trio_counts = Counter(tuple(sorted(row)) for row in ranked_numbers[:, :3].tolist())
    trifecta = {selection: count / samples for selection, count in trifecta_counts.items()}
    trio = {selection: count / samples for selection, count in trio_counts.items()}
    # Include zero-observation outcomes so consumers receive a complete probability table.
    for selection in combinations(numbers.tolist(), 3):
        trio.setdefault(tuple(sorted(selection)), 0.0)
    return win, trio, trifecta
