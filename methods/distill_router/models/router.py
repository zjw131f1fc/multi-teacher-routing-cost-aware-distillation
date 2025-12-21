"""Router model for multi-teacher selection."""

import torch
import torch.nn as nn
from typing import Dict, Any


class Router(nn.Module):
    """Router for selecting teacher models based on input features.

    Args:
        config: Full configuration dict containing:
            - backbone_settings.mllm_settings.hidden_dim: Input feature dimension
            - method_settings.num_teachers: Number of teacher models
            - method_settings.router_d_hidden: Hidden dimension for router network
            - method_settings.router_dropout: Dropout rate
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Extract parameters from config
        # Support both mllm_settings and llm_settings
        backbone_cfg = config["backbone_settings"]
        if "mllm_settings" in backbone_cfg:
            self.d_input = backbone_cfg["mllm_settings"]["hidden_dim"]
        elif "llm_settings" in backbone_cfg:
            self.d_input = backbone_cfg["llm_settings"]["hidden_dim"]
        else:
            raise ValueError("No hidden_dim found in backbone_settings")

        self.num_teachers = config["method_settings"]["num_teachers"]
        self.d_hidden = config["method_settings"]["router_d_hidden"]
        self.dropout = config["method_settings"]["router_dropout"]

        # TODO: Implement router network architecture
        # Example structure:
        # self.network = nn.Sequential(
        #     nn.Linear(self.d_input, self.d_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.d_hidden, self.num_teachers)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [batch_size, d_input]

        Returns:
            Teacher selection logits [batch_size, num_teachers]
        """
        # TODO: Implement forward pass
        raise NotImplementedError("Router forward pass not implemented yet")
