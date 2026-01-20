"""Policy Network for Teacher Selection

Architecture:
    Input (instruction text)
        ↓
    [Frozen DeBERTa Encoder]
        ↓
    [Trainable MLP Head]
        ↓
    Output: [P(T₁), P(T₂), ..., P(Tₙ), P(Reuse)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class PolicyNetwork(nn.Module):
    """Policy network for selecting teachers or reusing data.

    Args:
        encoder: Pretrained encoder model (e.g., DeBERTa)
        tokenizer: Tokenizer for the encoder
        num_actions: Number of actions (num_teachers + 1 for Reuse)
        hidden_dims: List of hidden dimensions for MLP head
        freeze_encoder: Whether to freeze encoder parameters
    """

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        num_actions: int,
        hidden_dims: list[int] = [512, 256],
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        self.tokenizer = tokenizer
        self.num_actions = num_actions
        self.freeze_encoder = freeze_encoder

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # Build MLP head
        encoder_hidden_size = self.encoder.config.hidden_size
        self.mlp_head = self._build_mlp(encoder_hidden_size, hidden_dims, num_actions)

    def _build_mlp(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Module:
        """Build MLP head with specified architecture."""
        layers = []

        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def encode(self, instruction_text: str | list[str]) -> torch.Tensor:
        """Encode instruction text using frozen encoder.

        Args:
            instruction_text: Single instruction or batch of instructions

        Returns:
            Encoded representation (batch_size, hidden_size)
        """
        # Tokenize
        if isinstance(instruction_text, str):
            instruction_text = [instruction_text]

        inputs = self.tokenizer(
            instruction_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to same device as model
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        # Encode
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(**inputs)
        else:
            outputs = self.encoder(**inputs)

        # Use [CLS] token representation
        encoded = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        return encoded

    def forward(self, instruction_text: str | list[str]) -> torch.Tensor:
        """Forward pass: instruction -> action probabilities.

        Args:
            instruction_text: Single instruction or batch of instructions

        Returns:
            Action probabilities (batch_size, num_actions)
        """
        # Encode instruction
        encoded = self.encode(instruction_text)

        # MLP head
        logits = self.mlp_head(encoded)

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        return probs

    def sample_action(self, instruction_text: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy distribution.

        Args:
            instruction_text: Single instruction or batch of instructions

        Returns:
            actions: Sampled action indices (batch_size,)
            log_probs: Log probabilities of sampled actions (batch_size,)
        """
        probs = self.forward(instruction_text)

        # Sample from categorical distribution
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs

    def get_action_probs(self, instruction_text: str | list[str], actions: torch.Tensor) -> torch.Tensor:
        """Get probabilities of specific actions.

        Args:
            instruction_text: Single instruction or batch of instructions
            actions: Action indices (batch_size,)

        Returns:
            Action probabilities (batch_size,)
        """
        probs = self.forward(instruction_text)
        action_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return action_probs
