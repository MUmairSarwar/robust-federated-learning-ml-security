# Robust Federated Learning Under Poisoning and Backdoor Attacks

An educational PyTorch study of federated learning security on MNIST, including controlled label poisoning, a corner-trigger backdoor, robust aggregation baselines, and an experimental fuzzy client-reliability weighting method.

> **Status:** implementation complete; benchmark values are intentionally `TBD` until the experiments are run. No numerical results are fabricated.

## Motivation

Federated learning lets participants train a shared model without pooling raw data, but malicious or corrupted clients may still manipulate model updates. This repository makes that problem inspectable in a small, reproducible simulation and compares several server-side aggregation strategies.

The fuzzy reliability prototype connects Muhammad Umair Sarwar's mathematics background—particularly uncertainty modelling, score functions, matrix games, and optimization—with machine-learning security. It is a research exercise, **not a proven security guarantee or a claim of academic novelty**.

## Ethical scope

This code is limited to defensive education and research using the public MNIST toy benchmark. It does not target real users, services, production models, biometric systems, medical systems, or financial systems. The attacks modify only local MNIST batches inside a controlled simulation.

## What is implemented

- A readable CNN and centralized reference trainer
- Local, synchronous federated simulation with configurable clients and rounds
- Seeded IID and simple label-shard non-IID partitions
- Label poisoning: configurable `source_label → target_label` flips
- Backdoor: configurable white square in the lower-right corner
- FedAvg, coordinate-wise median, and trimmed mean
- Experimental fuzzy client-reliability aggregation
- Clean accuracy, loss, backdoor ASR, and label source-to-target rate
- Per-client norms, distance, cosine similarity, fuzzy degrees, and scores
- CSV outputs, plots, YAML experiments, a quick demo, and research documentation

## Architecture

Each round broadcasts the current global CNN. Every client trains a private copy on its local partition and returns a parameter delta. The server combines deltas using the configured defense and applies the result to the global model. This is a local single-process simulation; it does not itself provide network or data privacy.

## Attacks and metrics

**Label poisoning** changes a configurable fraction of digit 1 labels to digit 7 on malicious clients. Its attack metric is the proportion of clean digit-1 test images predicted as 7.

**Backdoor poisoning** adds a small white corner trigger and relabels poisoned samples as digit 0. Backdoor attack success rate (ASR) is measured on triggered non-target test images, preventing true target-class images from inflating the metric.

## Defenses

| Method | Rule | Main caution |
|---|---|---|
| FedAvg | Sample-count-weighted mean update | Sensitive to extreme malicious updates |
| Median | Median at every coordinate | Ignores relationships among coordinates |
| Trimmed mean | Drop both coordinate-wise tails and average | Requires a suitable trim ratio |
| Fuzzy reliability | Score and softly weight complete client updates | Experimental heuristic; no guarantee |

### Fuzzy client-reliability scoring

For client update vector $u_i$ and mean update $\bar{u}$, the implementation measures:

1. $\lVert u_i\rVert_2$ and its deviation from the median client norm;
2. $\lVert u_i-\bar{u}\rVert_2$;
3. cosine similarity $\cos(u_i,\bar{u})$.

Median/MAD logistic transformations convert the first two indicators to degrees in $[0,1]$; cosine alignment is mapped to the same range. Their mean is the membership degree $\mu_i$. Complementary suspicious evidence gives non-membership $\nu_i$, and uncertainty is recorded as $\pi_i=\max(0,1-\mu_i-\nu_i)$. The experimental score is

$$s_i=\mu_i-\nu_i,$$

and aggregation weights are $w_i=\operatorname{softmax}(s)_i$. See [methodology](docs/methodology.md) for exact operational details and limitations.

## Installation

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MNIST downloads automatically on the first run. CUDA is selected automatically when available; otherwise all code runs on CPU.

## Run

```bash
# Centralized reference
python src/train_centralized.py

# Federated baseline and attacks
python src/main.py --config experiments/baseline_fedavg.yaml
python src/main.py --config experiments/label_poisoning_fedavg.yaml
python src/main.py --config experiments/backdoor_fedavg.yaml

# Custom prototype
python src/main.py --config experiments/fuzzy_reliability_defense.yaml

# Tiny first-run check
python src/main.py --config experiments/quick_demo.yaml --smoke-test
```

Run median and trimmed-mean variants with the other files in `experiments/`. Results are written to `results/tables/`; run-specific learning curves are written to `results/figures/`.

## Results

Do not fill this table until the corresponding configurations have run. For a credible comparison, repeat multiple seeds and report mean ± standard deviation.

| Defense | Attack | Clean accuracy | Attack success rate | Runtime |
|---|---|---:|---:|---:|
| FedAvg | None | TBD | TBD | TBD |
| FedAvg | Label poisoning | TBD | TBD | TBD |
| FedAvg | Backdoor | TBD | TBD | TBD |
| Median | Label poisoning | TBD | TBD | TBD |
| Trimmed mean | Backdoor | TBD | TBD | TBD |
| Fuzzy reliability | Backdoor | TBD | TBD | TBD |

## Figures

Successful runs generate `accuracy_vs_round.png`, `loss_vs_round.png`, `attack_success_rate_comparison.png`, and, for fuzzy aggregation, `reliability_scores.png`. The plotting module can also build `defense_comparison.png` after measured comparison data is supplied. An optional confusion matrix is left as future work.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── experiments/              # Eight full YAML scenarios + quick demo
├── results/
│   ├── figures/
│   └── tables/
├── docs/
│   ├── project_report.md
│   ├── methodology.md
│   ├── experiment_log.md
│   ├── linkedin_summary.md
│   └── github_from_zero.md
├── tests/test_aggregation.py
└── src/
    ├── main.py
    ├── config.py
    ├── model.py
    ├── train_centralized.py
    ├── data/
    ├── federated/
    ├── attacks/
    ├── defenses/
    ├── evaluation/
    └── utils/
```

## Technologies

Python, PyTorch, Torchvision, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, PyYAML, federated learning, robust statistics, and adversarial ML evaluation.

## CV-ready entry

**Robust Federated Learning Under Poisoning and Backdoor Attacks**  
2026 | Python, PyTorch, Torchvision, NumPy, Federated Learning, ML Security

- Built a PyTorch simulation of federated learning on MNIST with multiple clients and server-side aggregation.
- Implemented controlled label-poisoning and backdoor-attack scenarios to evaluate vulnerabilities in collaborative machine learning.
- Compared FedAvg, coordinate-wise median, trimmed mean, and fuzzy client-reliability aggregation as defenses against malicious clients.
- Designed a reliability-scoring extension inspired by uncertainty modelling and matrix-game research from an M.Phil. thesis.
- Reported clean accuracy, attack success rate, client-update statistics, and defense-comparison results.

## LinkedIn draft

I built a PyTorch research project to explore a practical question: how robust is federated learning when some clients submit poisoned updates?

Using a controlled MNIST simulation, I implemented label-poisoning and backdoor scenarios and compared FedAvg, coordinate-wise median, trimmed mean, and an experimental fuzzy client-reliability method. The custom method connects update statistics with my mathematics research background in uncertainty modelling and matrix games.

The project separates clean accuracy from attack success, saves reproducible configurations and per-client diagnostics, and clearly documents limitations. It is a defensive educational study—not a production security guarantee.

Repository: [add GitHub URL]

`#FederatedLearning #MachineLearningSecurity #PyTorch #RobustML #AdversarialML #DataScience`

## Limitations and future work

MNIST is not representative of modern deployed workloads. The simulation assumes synchronous full participation, a fixed malicious set, and no communication failures. The fuzzy center can be shifted by attackers and its evidence design needs ablations. Stronger follow-up work should add Dirichlet non-IID partitions, partial participation, multiple seeds and confidence intervals, adaptive attacks, independent server validation, published robust baselines, privacy mechanisms, and larger public research datasets.

## Author and license

Muhammad Umair Sarwar — incoming M.Sc. Mathematics student, specialization in Mathematics in Data Science, TU Darmstadt. Licensed under the MIT License.

