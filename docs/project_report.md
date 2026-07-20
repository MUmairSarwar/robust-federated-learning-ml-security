# Robust Federated Learning Under Poisoning and Backdoor Attacks

## Abstract

This educational study implements federated learning on MNIST and measures its behavior under controlled label-poisoning and backdoor simulations. It compares FedAvg, coordinate-wise median, trimmed mean, and an experimental fuzzy client-reliability weighting rule. The project emphasizes reproducible defensive evaluation and does not claim a new or proven security mechanism.

## Introduction and background

Federated learning coordinates model training across clients while keeping their raw datasets local. This changes the trust boundary but does not automatically make learning secure: malicious or corrupted participants can manipulate local optimization. Robust aggregation attempts to reduce the influence of anomalous updates.

The implementation uses a small convolutional classifier, ten clients, ten communication rounds, and one local epoch by default. Both CPU and CUDA are supported.

## Threat model

The server receives parameter deltas from a fixed set of simulated clients. A configurable fraction is malicious. The server does not know their identities. The adversary changes only its own toy MNIST batches; it does not attack communication, infrastructure, real services, or users.

## Attacks

Label poisoning flips a configurable fraction of digit 1 labels to digit 7. The backdoor attack stamps a small white corner square and assigns affected samples to target digit 0. Clean accuracy and attack-specific target rates are measured separately.

## Defenses

FedAvg is the reference method. Coordinate-wise median suppresses coordinate outliers, while trimmed mean removes configurable extremes. The fuzzy prototype derives membership and non-membership evidence from update norm, distance to the mean update, and cosine alignment. It uses the score `membership - nonmembership`, followed by softmax, as client aggregation weights. See [methodology.md](methodology.md) for exact details.

## Experimental setup and metrics

YAML files define seeds, clients, rounds, attacks, and defenses. Reported metrics are clean test loss, clean accuracy, local training loss, backdoor ASR or label source-to-target rate, client update diagnostics, and runtime notes. Results must come from actual runs.

## Results

| Defense | Attack | Clean accuracy | Attack success rate | Runtime |
|---|---|---:|---:|---:|
| FedAvg | None | TBD | TBD | TBD |
| FedAvg | Label poisoning | TBD | TBD | TBD |
| Median | Label poisoning | TBD | TBD | TBD |
| Trimmed mean | Backdoor | TBD | TBD | TBD |
| Fuzzy reliability | Backdoor | TBD | TBD | TBD |

## Discussion, limitations, and future work

Median and trimmed mean are coordinate-wise rules and can discard cross-parameter structure. The fuzzy method compares clients with the mean, which attackers may themselves shift; its uncertainty degree is zero with the current complementary construction and is retained explicitly so richer intuitionistic evidence can be studied later. MNIST, synchronous participation, fixed attackers, and a simple split limit external validity. Future work includes Dirichlet non-IID partitions, partial participation, adaptive attacks, independent validation signals, confidence intervals, stronger baselines, and larger—but still ethical—public benchmarks.

## Ethical statement

All attacks are contained simulations on MNIST for defensive research and education. The code must not be represented as production protection or used to target real systems, users, biometric data, finance, or medicine.

## CV-ready entry

**Robust Federated Learning Under Poisoning and Backdoor Attacks**  
2026 | Python, PyTorch, Torchvision, NumPy, Federated Learning, ML Security

- Built a PyTorch simulation of federated learning on MNIST with multiple clients and server-side aggregation.
- Implemented controlled label-poisoning and backdoor scenarios for defensive evaluation.
- Compared FedAvg, coordinate-wise median, trimmed mean, and fuzzy client-reliability aggregation.
- Designed an experimental score inspired by uncertainty modelling and matrix-game research from an M.Phil. thesis.
- Reported clean accuracy, attack success rate, client-update statistics, and defense comparisons.

