"""Aggregation algorithms operating on model updates."""
import torch
from defenses.fuzzy_reliability import fuzzy_scores


def aggregate(updates, method="fedavg", sample_counts=None, trim_ratio=0.2):
    if not updates:
        raise ValueError("At least one client update is required.")
    diagnostics = None
    if method == "fuzzy":
        weights, diagnostics = fuzzy_scores(updates)
    elif method == "fedavg":
        weights = torch.tensor(sample_counts or [1] * len(updates), dtype=torch.float64)
        weights /= weights.sum()
    result = {}
    for key in updates[0]:
        stack = torch.stack([update[key].float() for update in updates])
        if method in {"fedavg", "fuzzy"}:
            shape = [len(updates)] + [1] * (stack.ndim - 1)
            result[key] = (stack * weights.to(stack).view(shape)).sum(0)
        elif method == "median":
            result[key] = stack.median(dim=0).values
        elif method == "trimmed_mean":
            trim = int(len(updates) * trim_ratio)
            if 2 * trim >= len(updates):
                raise ValueError("trim_ratio removes all client updates.")
            sorted_values = stack.sort(dim=0).values
            result[key] = sorted_values[trim:len(updates)-trim].mean(0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    return result, diagnostics

