"""Small unit tests; requires torch but no MNIST download."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from defenses.robust_aggregation import aggregate
from defenses.fuzzy_reliability import fuzzy_scores


def test_fedavg_and_median():
    updates = [{"x": torch.tensor([value])} for value in (1.0, 2.0, 100.0)]
    mean, _ = aggregate(updates, "fedavg")
    median, _ = aggregate(updates, "median")
    assert torch.allclose(mean["x"], torch.tensor([103 / 3]))
    assert median["x"].item() == 2.0


def test_fuzzy_weights_are_probabilities():
    updates = [{"x": torch.tensor([value, value])} for value in (1.0, 1.1, 20.0)]
    weights, details = fuzzy_scores(updates)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    assert len(details["reliability_score"]) == 3

