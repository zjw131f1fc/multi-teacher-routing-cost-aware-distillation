"""Policy Network for Teacher Selection

Architecture:
    Input (instruction text + time_step)
        ↓
    [Frozen DeBERTa Encoder]
        ↓
    [Concat with Time Step Embedding]
        ↓
    [Trainable MLP Head]
        ↓
    Output: [P(T₁), P(T₂), ..., P(Tₙ)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class PolicyNetwork(nn.Module):
    """Policy network for selecting teachers.

    Args:
        encoder: Pretrained encoder model (e.g., DeBERTa)
        tokenizer: Tokenizer for the encoder
        num_actions: Number of actions (num_teachers)
        hidden_dims: List of hidden dimensions for MLP head
        freeze_encoder: Whether to freeze encoder parameters
        max_time_steps: Maximum number of time steps
        time_embedding_dim: Dimension of time step embedding
    """

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        num_actions: int,
        hidden_dims: list[int] = [512, 256],
        freeze_encoder: bool = True,
        max_time_steps: int = 100000,
        time_embedding_dim: int = 64,
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

        # Time step embedding
        self.time_embedding = nn.Embedding(max_time_steps, time_embedding_dim)

        # Build MLP head (encoder output + time embedding)
        encoder_hidden_size = self.encoder.config.hidden_size
        input_dim = encoder_hidden_size + time_embedding_dim
        self.mlp_head = self._build_mlp(input_dim, hidden_dims, num_actions)

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

    def forward(self, instruction_text: str | list[str], time_step: int | torch.Tensor) -> torch.Tensor:
        """Forward pass: instruction + time_step -> action probabilities.

        Args:
            instruction_text: Single instruction or batch of instructions
            time_step: Current time step (int or tensor)

        Returns:
            Action probabilities (batch_size, num_actions)
        """
        # Encode instruction
        encoded = self.encode(instruction_text)
        batch_size = encoded.size(0)

        # Get time embedding
        if isinstance(time_step, int):
            time_step = torch.tensor([time_step], device=encoded.device)
        if time_step.dim() == 0:
            time_step = time_step.unsqueeze(0)
        if time_step.size(0) == 1 and batch_size > 1:
            time_step = time_step.expand(batch_size)

        time_emb = self.time_embedding(time_step)  # (batch_size, time_embedding_dim)

        # Concat instruction encoding and time embedding
        combined = torch.cat([encoded, time_emb], dim=-1)

        # MLP head
        logits = self.mlp_head(combined)

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        return probs

    def sample_action(self, instruction_text: str | list[str], time_step: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy distribution.

        Args:
            instruction_text: Single instruction or batch of instructions
            time_step: Current time step

        Returns:
            actions: Sampled action indices (batch_size,)
            log_probs: Log probabilities of sampled actions (batch_size,)
        """
        probs = self.forward(instruction_text, time_step)

        # Sample from categorical distribution
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs

    def get_action_probs(self, instruction_text: str | list[str], time_step: int | torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get probabilities of specific actions.

        Args:
            instruction_text: Single instruction or batch of instructions
            time_step: Current time step
            actions: Action indices (batch_size,)

        Returns:
            Action probabilities (batch_size,)
        """
        probs = self.forward(instruction_text, time_step)
        action_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return action_probs
