"""Command-line entry point for federated experiments."""
import argparse
import torch
from config import load_config
from data.mnist import load_mnist, partition_clients
from federated.server import run_federated
from model import MNISTCNN
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled MNIST federated-learning security study")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file")
    parser.add_argument("--smoke-test", action="store_true", help="Override with a tiny 2-client, 1-round run")
    return parser.parse_args()


def main():
    args = parse_args(); cfg = load_config(args.config)
    if args.smoke_test:
        cfg.num_clients, cfg.rounds, cfg.max_train_samples, cfg.max_test_samples = 2, 1, 256, 128
        cfg.batch_size = 32
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _, test_loader = load_mnist(cfg.data_dir, cfg.batch_size, cfg.test_batch_size,
                                       cfg.max_train_samples, cfg.max_test_samples)
    clients = partition_clients(train, cfg.num_clients, cfg.batch_size, cfg.seed, cfg.iid)
    print(f"Device: {device}; clients: {cfg.num_clients}; malicious fraction: {cfg.malicious_client_fraction}")
    run_federated(MNISTCNN().to(device), clients, test_loader, cfg, device)


if __name__ == "__main__":
    main()

