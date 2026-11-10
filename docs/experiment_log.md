# Experiment log

| Date | Commit | Configs | Device | Output | Notes |
|---|---|---|---|---|---|
| 2026-08-17 | `17d4e9b` | the six configs in `.github/workflows/benchmark.yml`, seeds 42, 43, 44 | GitHub Actions, `ubuntu-latest` (CPU) | `results/benchmark_runs.csv` | 18 runs. Means and standard deviations are in the README. |

When comparing defences, keep the seed, data split, attack settings, number of rounds, model and client training settings the same, and repeat over several seeds.
