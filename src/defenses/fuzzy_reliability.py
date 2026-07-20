"""Experimental fuzzy client-reliability scoring.

Indicators are robustly scaled. Membership rewards central, directionally aligned
updates; non-membership rewards norm/distance outliers and negative alignment.
This heuristic is not a security guarantee.
"""
import torch
import torch.nn.functional as F


def flatten_update(update):
    return torch.cat([tensor.float().reshape(-1) for tensor in update.values()])


def _robust_membership(values, lower_is_better=True, eps=1e-8):
    median = values.median()
    mad = (values - median).abs().median().clamp_min(eps)
    z = (values - median) / (1.4826 * mad)
    return torch.sigmoid(-z if lower_is_better else z)


def fuzzy_scores(updates):
    vectors = torch.stack([flatten_update(update) for update in updates])
    center = vectors.mean(0)
    norms = vectors.norm(dim=1)
    distances = (vectors - center).norm(dim=1)
    cosine = F.cosine_similarity(vectors, center.unsqueeze(0), dim=1, eps=1e-8)
    norm_ok = _robust_membership((norms - norms.median()).abs(), True)
    distance_ok = _robust_membership(distances, True)
    alignment_ok = ((cosine + 1) / 2).clamp(0, 1)
    membership = (norm_ok + distance_ok + alignment_ok) / 3
    nonmembership = ((1 - norm_ok) + (1 - distance_ok) + (1 - alignment_ok)) / 3
    uncertainty = (1 - membership - nonmembership).clamp_min(0)
    score = membership - nonmembership
    weights = torch.softmax(score, dim=0)
    return weights, {"norm": norms, "distance": distances, "cosine": cosine,
                     "membership": membership, "nonmembership": nonmembership,
                     "uncertainty": uncertainty, "reliability_score": score}

