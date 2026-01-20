"""Gradient computation utilities for RL teacher selection.

This module provides functions to compute gradients for reward calculation:
- compute_gradient: Compute gradient vector for a single sample
- compute_validation_gradient: Compute average gradient direction from validation set
"""

import torch
import torch.nn as nn
from typing import Optional


def compute_gradient(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute gradient vector for a single sample.

    Args:
        model: Student model (must have forward(input_ids, labels) -> loss)
        input_ids: Input token ids
        labels: Target labels

    Returns:
        Flattened gradient vector (1D tensor)
    """
    model.zero_grad()

    # Forward pass
    loss = model(input_ids, labels)

    # Backward pass
    loss.backward()

    # Collect all parameter gradients
    grads = []
    for param in model.parameters():
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
) -> torch.Tensor:
    """Compute average gradient direction from validation set.

    Args:
        model: Student model
        val_loader: Validation dataloader
        num_samples: Number of samples to use

    Returns:
        Normalized gradient vector (unit vector)
    """
    grad_sum = None
    samples_collected = 0

    # Collect gradients from validation samples
    for input_ids, labels in val_loader:
        # Compute gradient for this batch
        g = compute_gradient(model, input_ids, labels)

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
