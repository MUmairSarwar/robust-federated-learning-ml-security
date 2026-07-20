"""Corner-trigger backdoor simulation for MNIST only."""
import torch


def add_trigger(images, size=3):
    triggered = images.clone()
    triggered[..., -size:, -size:] = 1.0
    return triggered


def poison_batch(images, labels, target=0, fraction=0.5, trigger_size=3):
    count = int(len(images) * fraction)
    if not count:
        return images, labels
    ids = torch.randperm(len(images), device=images.device)[:count]
    output_images, output_labels = images.clone(), labels.clone()
    output_images[ids] = add_trigger(output_images[ids], trigger_size)
    output_labels[ids] = target
    return output_images, output_labels

