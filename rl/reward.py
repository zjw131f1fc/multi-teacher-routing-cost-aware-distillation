"""Reward computation for RL teacher selection.

Reward function:
    R = Cosine(g_sample, g_val) * log(1 + ||g_sample||) - λ * Cost

Where:
    - g_sample: Gradient from synthetic sample (last N layers)
    - g_val: Average gradient direction from validation set (unit vector)
    - Cosine(·,·): Gradient alignment ∈ [-1, 1]
    - log(1 + ||g_sample||): Log-scaled gradient magnitude (prevents outliers)
    - Cost: Teacher invocation cost
    - λ: Cost weight coefficient
"""

import torch
import torch.nn.functional as F


def compute_reward(
    g_sample: torch.Tensor,
    g_val: torch.Tensor,
    cost: float,
    lambda_cost: float = 0.05,
) -> float:
    """Compute reward for a teacher selection action.

    Args:
        g_sample: Gradient vector from synthetic sample (1D tensor)
        g_val: Validation gradient direction (1D tensor, unit vector)
        cost: Cost of the action
        lambda_cost: Cost weight coefficient

    Returns:
        Reward value (scalar)
    """
    # Gradient alignment (cosine similarity)
    cosine_sim = F.cosine_similarity(g_sample, g_val, dim=0)

    # Gradient magnitude with log scaling (prevents outliers from dominating)
    grad_norm = torch.log1p(torch.norm(g_sample))

    # Quality term - Cost term
    reward = cosine_sim * grad_norm - lambda_cost * cost

    return reward.item()
