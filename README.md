# Robust Federated Learning Under Poisoning and Backdoor Attacks

**Project period: August 2025 – December 2025** (developed locally; published on GitHub in July 2026)

[![tests](https://github.com/MUmairSarwar/robust-federated-learning-ml-security/actions/workflows/tests.yml/badge.svg)](https://github.com/MUmairSarwar/robust-federated-learning-ml-security/actions/workflows/tests.yml)

An educational PyTorch study of federated-learning security on MNIST, including controlled label poisoning, a corner-trigger backdoor, robust aggregation baselines and an experimental fuzzy client-reliability weighting method.

![Experiment architecture](docs/architecture.svg)

## Project focus

This project studies how malicious or corrupted clients can influence federated learning and compares server-side aggregation strategies under heterogeneous data, poisoning and backdoor scenarios.

Implemented components include:

- Configurable multi-client CNN simulation
- IID and simple non-IID client partitions
- Label-poisoning and backdoor scenarios
- FedAvg, coordinate-wise median and trimmed-mean aggregation
- Experimental fuzzy client-reliability aggregation
- Clean accuracy and attack-success metrics
- Automated tests, YAML configurations and per-run metrics for reproducibility

## Why it matters

Federated learning allows participants to train a shared model without pooling raw data, but malicious clients can still manipulate model updates. This repository provides a controlled, reproducible environment for studying robust aggregation and adversarial behaviour.

The fuzzy reliability prototype connects uncertainty modelling and optimisation with machine-learning security. It is a research exercise, not a proven security guarantee or a claim of academic novelty.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example runs

```bash
python src/train_centralized.py
python src/main.py --config experiments/baseline_fedavg.yaml
python src/main.py --config experiments/label_poisoning_fedavg.yaml
python src/main.py --config experiments/backdoor_fedavg.yaml
python src/main.py --config experiments/fuzzy_reliability_defense.yaml
```

## Results

| Defense | Attack | Clean accuracy (mean ± std) | Attack success rate (mean ± std) |
|---|---|---:|---:|
| FedAvg | None | 97.66% ± 0.15 | 0.00% ± 0.00 |
| FedAvg | Label poisoning | 97.61% ± 0.09 | 0.15% ± 0.13 |
| Median | Label poisoning | 97.61% ± 0.05 | 0.03% ± 0.05 |
| FedAvg | Backdoor | 97.55% ± 0.13 | 98.99% ± 0.26 |
| Trimmed mean | Backdoor | 97.57% ± 0.12 | 92.65% ± 2.80 |
| Fuzzy reliability | Backdoor | 97.68% ± 0.12 | 91.97% ± 3.52 |

Results are means over 3 seeds (42, 43, 44) on MNIST; per-run values are in results/benchmark_runs.csv.

All methods keep about 97.6% clean accuracy. In this setup, label poisoning barely affects the model. The backdoor succeeds about 99% of the time under FedAvg. Trimmed-mean and fuzzy-reliability aggregation reduce this only to about 92%, so these defences are not sufficient against this backdoor.

## Technologies

Python, PyTorch, NumPy, pandas, scikit-learn, pytest, federated learning, robust statistics and adversarial ML evaluation.

## Limitations

MNIST is a toy benchmark and is not representative of modern deployed workloads. The simulation assumes synchronous participation and a controlled malicious-client setup. Stronger future work could include larger public datasets, more seeds, partial participation, additional robust baselines and adaptive attacks.

## Other selected projects

- [Fair Multi-Agent Battery Control (MARL)](https://github.com/MUmairSarwar/fair-multiagent-battery-control)
- [Rhine-Main Heat Risk Forecasting](https://github.com/MUmairSarwar/germany-heat-risk-forecasting)
- [German Energy Operations Analytics](https://github.com/MUmairSarwar/german-energy-operations)
- [Retail Customer & Operations Analytics](https://github.com/MUmairSarwar/retail-customer-analytics)
- [Strategic Classification](https://github.com/MUmairSarwar/strategic-classification)

## Author

**Muhammad Umair Sarwar**  
M.Sc. Mathematics student (Mathematics in Data Science), TU Darmstadt.  
Open to Werkstudent and data science roles in Germany, and to funded PhD positions.
