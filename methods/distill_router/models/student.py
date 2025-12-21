"""Student model for distillation."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class Student(nn.Module):
    """Student model that learns from multiple teachers.

    Args:
        config: Full configuration dict containing:
            - backbone_settings: Backbone configuration
            - method_settings.student_use_adapter: Whether to use adapter
            - method_settings: Additional method-specific settings
        backbone: Pretrained backbone model (optional)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        backbone: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config
        self.backbone = backbone

        # Extract settings
        self.use_adapter = config["method_settings"].get("student_use_adapter", False)

        # TODO: Add any additional learnable components
        # Example:
        # if self.use_adapter:
        #     hidden_dim = config["backbone_settings"]["mllm_settings"]["hidden_dim"]
        #     self.adapter = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            Model outputs
        """
        # TODO: Implement forward pass
        raise NotImplementedError("Student forward pass not implemented yet")
