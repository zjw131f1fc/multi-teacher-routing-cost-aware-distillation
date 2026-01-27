"""Teacher model for knowledge distillation."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class TeacherModel:
    """Wrapper for teacher LLM model, only provides generate."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, instruction: str) -> str:
        """Generate response for instruction.

        Args:
            instruction: Input instruction text

        Returns:
            Generated response string
        """
        inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode, skip input tokens
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        return response


def load_teacher(config: dict, teacher_name: str) -> TeacherModel:
    """Load teacher model from config.

    Args:
        config: Configuration dict with:
            - teachers.<teacher_name>.model_name: HuggingFace model name
            - teachers.<teacher_name>.device: Device (default: "cuda")
            - teachers.<teacher_name>.max_new_tokens: Max tokens to generate
            - teachers.<teacher_name>.temperature: Sampling temperature
            - teachers.<teacher_name>.top_p: Top-p sampling
        teacher_name: Name of the teacher to load

    Returns:
        TeacherModel instance
    """
    teachers_config = config.get("teachers", {})
    teacher_config = teachers_config.get(teacher_name, {})

    model_name = teacher_config.get("model_name")
    if model_name is None:
        raise ValueError(f"No model_name specified for teacher '{teacher_name}'")

    device = teacher_config.get("device", "cuda")
    max_new_tokens = teacher_config.get("max_new_tokens", 512)
    temperature = teacher_config.get("temperature", 0.7)
    top_p = teacher_config.get("top_p", 0.9)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return TeacherModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def load_teachers(config: dict) -> dict:
    """Load all teachers from config.

    Args:
        config: Configuration dict

    Returns:
        Dict of {teacher_name: TeacherModel}
    """
    teachers_config = config.get("teachers", {})
    teachers = {}

    for teacher_name in teachers_config.keys():
        teachers[teacher_name] = load_teacher(config, teacher_name)

    return teachers
