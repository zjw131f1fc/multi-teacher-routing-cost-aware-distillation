"""Mock data and models for testing RL teacher selection framework

This module provides simple mock implementations for:
- Instructions (questions/tasks)
- Student model (simple transformer for gradient computation)
- Teacher models (mock generators)
- Validation dataloader

Usage:
    from methods.rl_teacher_selection.mock_data import (
        get_mock_instructions,
        get_mock_student_model,
        get_mock_teachers,
        get_mock_val_loader
    )
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple


# ============================================================================
# Mock Instructions
# ============================================================================

def get_mock_instructions(num_samples: int = 1000) -> List[str]:
    """Generate mock instructions for testing.

    Args:
        num_samples: Number of instructions to generate

    Returns:
        List of instruction strings
    """
    templates = [
        "Solve the equation: {}x + {} = {}",
        "Calculate: {} * {} + {}",
        "What is {} divided by {}?",
        "Find the value of x: {}x - {} = {}",
        "Compute: ({} + {}) * {}",
    ]

    instructions = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        # Generate random numbers for the template
        nums = [torch.randint(1, 20, (1,)).item() for _ in range(3)]
        instruction = template.format(*nums)
        instructions.append(instruction)

    return instructions


# ============================================================================
# Mock Student Model
# ============================================================================

class MockStudentModel(nn.Module):
    """Simple student model for testing gradient computation.

    A small transformer-like model (~10M parameters) for fast testing.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass with optional loss computation.

        Args:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len), optional

        Returns:
            If labels provided: loss (scalar)
            Otherwise: logits (batch_size, seq_len, vocab_size)
        """
        # Embedding
        x = self.embedding(input_ids)

        # Transformer
        x = self.transformer(x)

        # LM head
        logits = self.lm_head(x)

        if labels is not None:
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss
        else:
            return logits


def get_mock_student_model(device: str = "cpu") -> MockStudentModel:
    """Create and initialize a mock student model.

    Args:
        device: Device to place the model on

    Returns:
        Initialized MockStudentModel
    """
    model = MockStudentModel()
    model.to(device)
    model.train()
    return model


# ============================================================================
# Mock Teacher Models
# ============================================================================

class MockTeacher:
    """Mock teacher model that generates simple responses with caching."""

    def __init__(self, name: str, quality: float = 0.8, cost: float = 1.0):
        """Initialize mock teacher.

        Args:
            name: Teacher name
            quality: Quality factor (0-1), affects response quality
            cost: Cost of calling this teacher
        """
        self.name = name
        self.quality = quality
        self.cost = cost
        self.vocab_size = 32000
        self.cache = {}  # {instruction: (input_ids, labels)}
        self.cache_hits = 0
        self.cache_misses = 0

    def generate(self, instruction: str, max_length: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate response for instruction (with caching).

        Args:
            instruction: Input instruction
            max_length: Maximum response length

        Returns:
            input_ids: (seq_len,) tensor
            labels: (seq_len,) tensor
        """
        # Check cache first
        if instruction in self.cache:
            self.cache_hits += 1
            return self.cache[instruction]

        # Cache miss - generate new response
        self.cache_misses += 1
        seq_len = torch.randint(20, max_length, (1,)).item()

        # Input: random tokens representing instruction
        input_ids = torch.randint(0, self.vocab_size, (seq_len,))

        # Labels: similar to input but with some variation based on quality
        if torch.rand(1).item() < self.quality:
            # High quality: labels are similar to input
            labels = input_ids.clone()
        else:
            # Low quality: labels are more random
            labels = torch.randint(0, self.vocab_size, (seq_len,))

        # Store in cache
        self.cache[instruction] = (input_ids, labels)
        return input_ids, labels

    def prefill_cache(self, instructions: List[str], max_length: int = 50):
        """Prefill cache with responses for given instructions.

        Simulates having run previous experiments with this teacher.

        Args:
            instructions: List of instructions to prefill
            max_length: Maximum response length
        """
        for instruction in instructions:
            if instruction not in self.cache:
                self.generate(instruction, max_length)

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self):
        """Clear cache and reset statistics."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


def get_mock_teachers() -> List[MockTeacher]:
    """Create a pool of mock teachers with different quality/cost trade-offs.

    Returns:
        List of MockTeacher instances
    """
    teachers = [
        MockTeacher("gpt-4", quality=0.95, cost=1.0),
        MockTeacher("gpt-3.5", quality=0.80, cost=0.1),
        MockTeacher("llama-70b", quality=0.85, cost=0.5),
    ]
    return teachers


# ============================================================================
# Mock Validation Dataloader
# ============================================================================

def get_mock_val_loader(
    num_samples: int = 500,
    batch_size: int = 32,
    seq_len: int = 50,
    vocab_size: int = 32000,
) -> DataLoader:
    """Create a mock validation dataloader.

    Args:
        num_samples: Number of validation samples
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size

    Returns:
        DataLoader with (input_ids, labels) batches
    """
    # Generate random validation data
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))

    dataset = TensorDataset(input_ids, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


# ============================================================================
# Convenience function
# ============================================================================

def get_mock_data(
    num_instructions: int = 1000,
    num_val_samples: int = 500,
    device: str = "cpu",
):
    """Get all mock data and models in one call.

    Args:
        num_instructions: Number of training instructions
        num_val_samples: Number of validation samples
        device: Device for models

    Returns:
        Dictionary with:
            - instructions: List of instruction strings
            - student_model: MockStudentModel
            - teachers: List of MockTeacher
            - val_loader: DataLoader
    """
    return {
        "instructions": get_mock_instructions(num_instructions),
        "student_model": get_mock_student_model(device),
        "teachers": get_mock_teachers(),
        "val_loader": get_mock_val_loader(num_val_samples),
    }


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("Loading mock data...")

    # Get all mock data
    data = get_mock_data(num_instructions=100, num_val_samples=50)

    print(f"\nInstructions: {len(data['instructions'])} samples")
    print(f"Example: {data['instructions'][0]}")

    print(f"\nStudent model: {sum(p.numel() for p in data['student_model'].parameters()):,} parameters")

    print(f"\nTeachers: {len(data['teachers'])} teachers")
    for teacher in data["teachers"]:
        print(f"  - {teacher.name}: quality={teacher.quality}, cost={teacher.cost}")

    print(f"\nValidation loader: {len(data['val_loader'])} batches")

    # Test student model forward pass
    print("\nTesting student model...")
    batch = next(iter(data["val_loader"]))
    input_ids, labels = batch
    loss = data["student_model"](input_ids, labels)
    print(f"Loss: {loss.item():.4f}")

    # Test teacher generation
    print("\nTesting teacher generation...")
    teacher = data["teachers"][0]
    instruction = data["instructions"][0]
    input_ids, labels = teacher.generate(instruction)
    print(f"Generated sequence length: {len(input_ids)}")

    print("\n✓ All mock data loaded successfully!")
