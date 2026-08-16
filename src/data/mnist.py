"""Reproducible MNIST loading and federated client partitions."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _limited(dataset: Dataset, maximum: int | None, seed: int = 42) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(dataset), size=maximum, replace=False)
    return Subset(dataset, indices.tolist())


def load_mnist(
    data_dir: str,
    batch_size: int,
    test_batch_size: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
):
    """Download MNIST when needed and return train data plus reference loaders."""
    transform = transforms.ToTensor()
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train = _limited(train, max_train_samples)
    test = _limited(test, max_test_samples, seed=43)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=False)
    return train, train_loader, test_loader


def _targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent = _targets(dataset.dataset)
        return parent[np.asarray(dataset.indices)]
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return np.asarray(targets)
    return np.asarray([int(dataset[index][1]) for index in range(len(dataset))])


def partition_clients(
    dataset: Dataset,
    num_clients: int,
    batch_size: int,
    seed: int = 42,
    iid: bool = True,
) -> list[DataLoader]:
    """Create deterministic IID or label-shard client partitions."""
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if num_clients > len(dataset):
        raise ValueError("num_clients cannot exceed the number of samples")

    rng = np.random.default_rng(seed)
    if iid:
        indices = rng.permutation(len(dataset))
    else:
        labels = _targets(dataset)
        random_tie_breaker = rng.random(len(dataset))
        indices = np.lexsort((random_tie_breaker, labels))

    partitions = np.array_split(indices, num_clients)
    loaders = []
    for client_id, client_indices in enumerate(partitions):
        generator = torch.Generator().manual_seed(seed + client_id)
        loaders.append(
            DataLoader(
                Subset(dataset, client_indices.tolist()),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )
        )
    return loaders
