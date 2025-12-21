"""Training step for distillation router."""

import torch
from typing import Dict, Any


def train_step(batch: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Training step function.

    Args:
        batch: Batch data from dataloader
            - instruction: str, the problem/question
            - responses: dict, teacher model responses
                - {teacher_name: {messages: [...], rewards: {...}}}
            - metadata: dict, additional metadata
        info: Training info dict containing:
            - models: dict of registered models
            - config: training configuration
            - step: current training step
            - epoch: current epoch

    Returns:
        Dictionary of loss dictionaries for each parameter group:
        {
            "router": {
                "routing_loss": tensor,
                "cost_loss": tensor,
                ...
            },
            "student": {
                "distill_loss": tensor,
                "task_loss": tensor,
                ...
            }
        }
    """
    # Extract models from info
    router = info["models"]["router"]
    student = info["models"]["student"]
    config = info["config"]

    # TODO: Implement training step
    # 1. Extract teacher responses and rewards
    # 2. Route to best teacher(s) based on input
    # 3. Compute distillation loss from selected teacher
    # 4. Compute cost-aware routing loss
    # 5. Return loss dict for each parameter group

    raise NotImplementedError("Training step not implemented yet")

    # Example return structure:
    # return {
    #     "router": {
    #         "routing_loss": routing_loss,
    #         "cost_loss": cost_loss,
    #     },
    #     "student": {
    #         "distill_loss": distill_loss,
    #     }
    # }
