# Methodology

## Data and client split

MNIST is downloaded through `torchvision`. With IID partitioning, a seeded permutation of training indices is divided as evenly as possible among clients. The optional non-IID mode sorts examples by label before forming shards; it is deliberately simple and should be treated as a stress-test partition, not a realistic population model.

## Local and global learning

At each communication round, every client copies the current CNN and performs stochastic-gradient training on its local samples. The client sends a parameter delta—not its raw data—to the simulated server. This is a single-process simulation and therefore demonstrates the algorithmic boundary, not network privacy. FedAvg weights deltas by local sample count; the other rules combine client deltas coordinate by coordinate.

## Controlled attacks

Label poisoning changes a configurable fraction of source-class labels (default 1) to a target class (default 7) on designated malicious clients. The reported source-to-target rate is the fraction of clean source-label test samples predicted as the target.

The backdoor simulation writes a configurable white square into the lower-right corner and replaces the affected training labels with the target label (default 0). Attack success rate (ASR) is evaluated on triggered test images whose true label is not already the target. Thus, ASR measures targeted misclassification without counting trivial target-class examples.

## Robust aggregation

Coordinate-wise median selects the median client value at every parameter coordinate. Trimmed mean sorts values at each coordinate, removes `floor(n * trim_ratio)` observations from both ends, and averages the rest.

## Fuzzy reliability prototype

Each client update is flattened to a vector. The method records update norm, Euclidean distance from the mean update, and cosine alignment with the mean. Norm deviation and distance are transformed through robust median/MAD logistic membership functions; cosine similarity is mapped from `[-1, 1]` to `[0, 1]`.

The experimental degrees are:

- membership: mean evidence from normal norm, central distance, and directional alignment;
- non-membership: mean complementary suspicious evidence;
- uncertainty: `max(0, 1 - membership - non-membership)`;
- reliability score: `membership - non-membership`.

Softmax converts reliability scores to nonnegative weights that sum to one. The final update is their weighted mean. These scores are heuristic indicators. They do not identify attackers with certainty and provide no formal security guarantee.

## Reproducibility and outputs

Python, NumPy, and PyTorch random generators are seeded. CUDA deterministic mode is enabled when applicable. Per-round metrics and fuzzy diagnostics are saved as CSV; figures are generated from these measured values. Hardware and dependency versions can still cause small numerical differences.

