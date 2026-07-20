"""Clean and attack-specific evaluation metrics."""
import torch
import torch.nn.functional as F
from attacks.backdoor import add_trigger


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(images)
    return loss / count, correct / count


@torch.no_grad()
def backdoor_asr(model, loader, device, target=0, trigger_size=3):
    model.eval()
    successes, count = 0, 0
    for images, labels in loader:
        mask = labels != target
        images = images[mask].to(device)
        if not len(images):
            continue
        predictions = model(add_trigger(images, trigger_size)).argmax(1)
        successes += (predictions == target).sum().item()
        count += len(images)
    return successes / max(count, 1)


@torch.no_grad()
def source_to_target_rate(model, loader, device, source=1, target=7):
    model.eval()
    hits, count = 0, 0
    for images, labels in loader:
        mask = labels == source
        images = images[mask].to(device)
        if not len(images):
            continue
        hits += (model(images).argmax(1) == target).sum().item()
        count += len(images)
    return hits / max(count, 1)

