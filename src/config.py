"""Configuration loading and validation."""
from dataclasses import asdict, dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    experiment_name: str = "baseline_fedavg"
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "results"
    num_clients: int = 10
    rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 64
    test_batch_size: int = 256
    learning_rate: float = 0.01
    momentum: float = 0.9
    aggregation: str = "fedavg"
    weighted_fedavg: bool = True
    trim_ratio: float = 0.2
    attack: str = "none"
    malicious_client_fraction: float = 0.0
    poison_fraction: float = 0.5
    source_label: int = 1
    target_label: int = 7
    backdoor_target: int = 0
    trigger_size: int = 3
    iid: bool = True
    max_train_samples: int | None = None
    max_test_samples: int | None = None

    def validate(self):
        if self.aggregation not in {"fedavg", "median", "trimmed_mean", "fuzzy"}:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        if self.attack not in {"none", "label_poisoning", "backdoor"}:
            raise ValueError(f"Unknown attack: {self.attack}")
        if not 0 <= self.trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5).")
        if self.num_clients < 1 or self.rounds < 1:
            raise ValueError("num_clients and rounds must be positive.")


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    unknown = set(values) - set(asdict(Config()))
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    cfg = Config(**values)
    cfg.validate()
    return cfg

