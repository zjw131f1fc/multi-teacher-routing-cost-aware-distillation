"""Student model for knowledge distillation."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class StudentModel(nn.Module):
    """Wrapper for student LLM model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns loss.

        Args:
            input_ids: Input token ids (batch_size, seq_len)
            labels: Target labels (batch_size, seq_len)

        Returns:
            Loss scalar
        """
        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs.loss


def load_student(config: dict) -> StudentModel:
    """Load student model from config.

    Args:
        config: Configuration dict with:
            - student.model_name: HuggingFace model name
            - student.device: Device to load model on (default: "cuda")

    Returns:
        StudentModel instance
    """
    student_config = config.get("student", {})
    model_name = student_config.get("model_name", "gpt2")
    device = student_config.get("device", "cuda")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    return StudentModel(model)


def load_student_tokenizer(config: dict):
    """Load tokenizer for student model.

    Args:
        config: Configuration dict with:
            - student.model_name: HuggingFace model name

    Returns:
        Tokenizer instance
    """
    student_config = config.get("student", {})
    model_name = student_config.get("model_name", "gpt2")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
