"""Evaluation step for distillation router."""

import torch
from typing import Dict, Any


def eval_step(batch: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluation step function.

    Args:
        batch: Batch data from dataloader
            - instruction: str, the problem/question
            - responses: dict, teacher model responses
            - metadata: dict, additional metadata
        info: Evaluation info dict containing:
            - models: dict of registered models
            - config: training configuration
            - dataset_bundle: dataset metadata and judge function

    Returns:
        Dictionary of evaluation metrics:
        {
            "accuracy": float,
            "routing_accuracy": float,
            "avg_cost": float,
            ...
        }
    """
    # Extract models from info
    router = info["models"]["router"]
    student = info["models"]["student"]
    config = info["config"]

    # TODO: Implement evaluation step
    # 1. Generate student predictions
    # 2. Evaluate routing decisions
    # 3. Compute accuracy and other metrics
    # 4. Return metrics dict

    raise NotImplementedError("Evaluation step not implemented yet")

    # Example return structure:
    # return {
    #     "accuracy": accuracy,
    #     "routing_accuracy": routing_accuracy,
    #     "avg_cost": avg_cost,
    # }
