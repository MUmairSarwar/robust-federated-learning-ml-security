"""Small unit tests; requires torch but no MNIST download."""
import sys
from pathlib import Path
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from defenses.robust_aggregation import aggregate
from defenses.fuzzy_reliability import fuzzy_scores
from data.mnist import partition_clients


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


def test_iid_partition_assigns_every_sample_once():
    dataset = TensorDataset(torch.randn(20, 1, 28, 28), torch.arange(20) % 10)
    loaders = partition_clients(dataset, num_clients=4, batch_size=2, seed=7, iid=True)
    assert [len(loader.dataset) for loader in loaders] == [5, 5, 5, 5]
    assigned = [index for loader in loaders for index in loader.dataset.indices]
    assert sorted(assigned) == list(range(20))
