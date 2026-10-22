import numpy as np

from uma_api.probability import simulate_rank_probabilities


def test_probabilities_are_coherent_and_reproducible():
    horses = [
        {"horse_number": 1, "mean": 17.2, "std": 0.4},
        {"horse_number": 2, "mean": 16.8, "std": 0.5},
        {"horse_number": 3, "mean": 16.4, "std": 0.6},
        {"horse_number": 4, "mean": 16.0, "std": 0.7},
    ]
    win, trio, trifecta = simulate_rank_probabilities(horses, 20_000, 7)
    again = simulate_rank_probabilities(horses, 20_000, 7)
    assert np.array_equal(win, again[0])
    assert np.isclose(win.sum(), 1)
    assert np.isclose(sum(trio.values()), 1)
    assert np.isclose(sum(trifecta.values()), 1)
    assert win[0] > win[-1]
    for selection, probability in trio.items():
        matching = sum(value for order, value in trifecta.items() if set(order) == set(selection))
        assert np.isclose(probability, matching)
