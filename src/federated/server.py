"""Federated experiment orchestration."""
from pathlib import Path
import pandas as pd
import torch
from evaluation.metrics import backdoor_asr, evaluate, source_to_target_rate
from evaluation.plots import save_run_plots
from federated.client import train_client
from defenses.robust_aggregation import aggregate


def run_federated(model, client_loaders, test_loader, cfg, device):
    malicious_count = int(cfg.num_clients * cfg.malicious_client_fraction)
    malicious = set(range(malicious_count))
    history, reliability_rows = [], []
    for round_id in range(1, cfg.rounds + 1):
        updates, losses, counts = [], [], []
        for client_id, loader in enumerate(client_loaders):
            update, loss, count = train_client(model, loader, cfg, device, client_id in malicious)
            updates.append(update); losses.append(loss); counts.append(count)
        aggregated, diagnostics = aggregate(updates, cfg.aggregation,
                                            counts if cfg.weighted_fedavg else None,
                                            cfg.trim_ratio)
        state = model.state_dict()
        model.load_state_dict({key: state[key].cpu() + aggregated[key] for key in state})
        model.to(device)
        test_loss, accuracy = evaluate(model, test_loader, device)
        attack_rate = 0.0
        if cfg.attack == "backdoor":
            attack_rate = backdoor_asr(model, test_loader, device, cfg.backdoor_target, cfg.trigger_size)
        elif cfg.attack == "label_poisoning":
            attack_rate = source_to_target_rate(model, test_loader, device, cfg.source_label, cfg.target_label)
        history.append({"round": round_id, "train_loss": sum(losses)/len(losses),
                        "test_loss": test_loss, "clean_accuracy": accuracy,
                        "attack_success_rate": attack_rate})
        if diagnostics:
            for client_id in range(len(updates)):
                reliability_rows.append({"round": round_id, "client": client_id,
                    **{key: float(value[client_id]) for key, value in diagnostics.items()},
                    "malicious": client_id in malicious})
        print(f"Round {round_id:02d}: accuracy={accuracy:.4f}, loss={test_loss:.4f}, attack_rate={attack_rate:.4f}")
    tables = Path(cfg.output_dir) / "tables"; tables.mkdir(parents=True, exist_ok=True)
    history_df, reliability_df = pd.DataFrame(history), pd.DataFrame(reliability_rows)
    history_df.to_csv(tables / f"{cfg.experiment_name}_metrics.csv", index=False)
    if not reliability_df.empty:
        reliability_df.to_csv(tables / f"{cfg.experiment_name}_reliability.csv", index=False)
    save_run_plots(history_df, reliability_df, cfg.output_dir)
    return history_df

