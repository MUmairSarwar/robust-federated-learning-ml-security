"""Label-flipping simulation."""
import torch


def poison_labels(labels, source=1, target=7, fraction=0.5):
    poisoned = labels.clone()
    candidates = torch.where(labels == source)[0]
    count = int(len(candidates) * fraction)
    if count:
        poisoned[candidates[torch.randperm(len(candidates), device=labels.device)[:count]]] = target
    return poisoned

