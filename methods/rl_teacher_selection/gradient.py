"""Gradient computation utilities for RL teacher selection.

This module provides functions to compute gradients for reward calculation:
- compute_gradient: Compute gradient vector for a single sample (last N layers only)
- compute_validation_gradient: Compute average gradient direction from validation set
"""

import torch
import torch.nn as nn
from typing import Optional, List


def get_last_n_layers(model: nn.Module, n: int = 3) -> List[nn.Module]:
    """Get the last N layers of a model.

    Args:
        model: The model to extract layers from
        n: Number of layers to extract

    Returns:
        List of the last N layers
    """
    # Get all named modules that have parameters
    layers_with_params = []
    for name, module in model.named_modules():
        if list(module.parameters(recurse=False)):
            layers_with_params.append((name, module))

    # Return last N layers
    return [m for _, m in layers_with_params[-n:]]


def compute_gradient(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    last_n_layers: int = 3,
) -> torch.Tensor:
    """Compute gradient vector for a single sample (last N layers only).

    Args:
        model: Student model (must have forward(input_ids, labels) -> loss)
        input_ids: Input token ids
        labels: Target labels
        last_n_layers: Number of last layers to compute gradient for

    Returns:
        Flattened gradient vector (1D tensor)
    """
    model.zero_grad()

    # Forward pass
    loss = model(input_ids, labels)

    # Backward pass
    loss.backward()

    # Get last N layers
    target_layers = get_last_n_layers(model, last_n_layers)

    # Collect gradients from last N layers only
    grads = []
    for layer in target_layers:
        for param in layer.parameters(recurse=False):
            if param.grad is not None:
                grads.append(param.grad.flatten())

    # Concatenate into single vector
    g = torch.cat(grads)

    # Clean up
    model.zero_grad()

    return g


def compute_validation_gradient(
    model: nn.Module,
    val_loader,
    num_samples: int = 200,
    last_n_layers: int = 3,
) -> torch.Tensor:
    """Compute average gradient direction from validation set (last N layers only).

    Args:
        model: Student model
        val_loader: Validation dataloader
        num_samples: Number of samples to use
        last_n_layers: Number of last layers to compute gradient for

    Returns:
        Normalized gradient vector (unit vector)
    """
    grad_sum = None
    samples_collected = 0

    # Collect gradients from validation samples
    for input_ids, labels in val_loader:
        # Compute gradient for this batch
        g = compute_gradient(model, input_ids, labels, last_n_layers)

        # Accumulate
        if grad_sum is None:
            grad_sum = g
        else:
            grad_sum = grad_sum + g

        samples_collected += 1

        # Stop when we have enough samples
        if samples_collected >= num_samples:
            break

    # Average
    g_val = grad_sum / samples_collected

    # Normalize to unit vector
    g_val_normalized = g_val / torch.norm(g_val)

    return g_val_normalized
