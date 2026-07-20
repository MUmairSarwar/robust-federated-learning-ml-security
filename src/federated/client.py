"""Local client training."""
from copy import deepcopy
import torch
import torch.nn.functional as F
from attacks.backdoor import poison_batch
from attacks.label_poisoning import poison_labels


def train_client(global_model, loader, cfg, device, malicious=False):
    model = deepcopy(global_model).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
    total_loss, seen = 0.0, 0
    for _ in range(cfg.local_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if malicious and cfg.attack == "label_poisoning":
                labels = poison_labels(labels, cfg.source_label, cfg.target_label, cfg.poison_fraction)
            elif malicious and cfg.attack == "backdoor":
                images, labels = poison_batch(images, labels, cfg.backdoor_target,
                                               cfg.poison_fraction, cfg.trigger_size)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
            seen += len(images)
    update = {name: model.state_dict()[name].detach().cpu() - global_model.state_dict()[name].detach().cpu()
              for name in global_model.state_dict()}
    return update, total_loss / max(seen, 1), len(loader.dataset)

