"""Centralized MNIST reference training."""
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.mnist import load_mnist
from evaluation.metrics import evaluate
from model import MNISTCNN
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true"); args = parser.parse_args()
    set_seed(42); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _, test_loader = load_mnist("data", 64, 256, 256 if args.smoke_test else None,
                                      128 if args.smoke_test else None)
    loader = DataLoader(train, batch_size=64, shuffle=True); model = MNISTCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); rows = []
    for epoch in range(1, (1 if args.smoke_test else args.epochs) + 1):
        model.train(); total = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels); loss.backward(); optimizer.step()
            total += loss.item() * len(images)
        test_loss, accuracy = evaluate(model, test_loader, device)
        rows.append({"epoch": epoch, "train_loss": total/len(train), "test_loss": test_loss,
                     "clean_accuracy": accuracy})
        print(rows[-1])
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("results/tables/centralized_metrics.csv", index=False)


if __name__ == "__main__":
    main()

