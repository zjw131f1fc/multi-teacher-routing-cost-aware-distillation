"""Gymnasium environment for RL-based teacher selection.

This environment integrates all components:
- Policy network (handled by SB3)
- Teacher pool
- Gradient computation
- Reward computation
- Student model training
"""

import gymnasium as gym
import numpy as np
import torch
from typing import List, Optional, Dict, Any, Tuple

from .teacher_pool import TeacherPool
from .gradient import compute_gradient, compute_validation_gradient
from .reward import compute_reward


class TeacherSelectionEnv(gym.Env):
    """Gymnasium environment for teacher selection.

    Observation: Current instruction (returned as dict with text)
    Action: Discrete - [0, num_teachers-1] for selecting teachers
    """

    def __init__(
        self,
        instructions: List[str],
        student_model: torch.nn.Module,
        teacher_pool: TeacherPool,
        val_dataloader,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize environment.

        Args:
            instructions: List of training instructions
            student_model: Student model to train
            teacher_pool: Pool of teacher models
            val_dataloader: Validation dataloader
            config: Configuration dict with:
                - val_samples: Number of validation samples for gradient (default: 200)
                - lambda_cost: Cost weight coefficient (default: 0.05)
                - student_update_freq: Train student every N steps (default: 100)
                - student_train_steps: Number of training steps for student (default: 10)
        """
        super().__init__()

        self.instructions = instructions
        self.student_model = student_model
        self.teacher_pool = teacher_pool
        self.val_dataloader = val_dataloader

        # Configuration
        self.config = config or {}
        self.val_samples = self.config.get("val_samples", 200)
        self.lambda_cost = self.config.get("lambda_cost", 0.05)
        self.student_update_freq = self.config.get("student_update_freq", 100)
        self.student_train_steps = self.config.get("student_train_steps", 10)

        # Action space: [0, num_teachers-1] for selecting teachers
        num_teachers = len(teacher_pool)
        self.action_space = gym.spaces.Discrete(num_teachers)

        # Observation space: instruction text (we'll return as dict)
        self.observation_space = gym.spaces.Dict({
            "instruction": gym.spaces.Text(max_length=1000)
        })

        # Training state
        self.current_idx = 0
        self.dataset_D = []  # Accumulated synthetic dataset
        self.g_val = None  # Validation gradient direction
        self.step_count = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment to initial state.

        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)

        # Reset to first instruction
        self.current_idx = 0
        self.step_count = 0

        # Compute initial validation gradient
        self.g_val = compute_validation_gradient(
            self.student_model, self.val_dataloader, self.val_samples
        )

        obs = {"instruction": self.instructions[self.current_idx]}
        info = {}

        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step.

        Args:
            action: Teacher index to use

        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        instruction = self.instructions[self.current_idx]

        # Call teacher to generate data
        (input_ids, labels), cost = self.teacher_pool.generate(action, instruction)

        # Compute gradient for this sample
        g_sample = compute_gradient(self.student_model, input_ids, labels)

        # Compute reward
        reward = compute_reward(g_sample, self.g_val, cost, self.lambda_cost)

        # Add to dataset D
        self.dataset_D.append((input_ids, labels))

        # Move to next instruction
        self.current_idx += 1
        self.step_count += 1

        # Check if episode is done
        terminated = self.current_idx >= len(self.instructions)
        truncated = False

        # Prepare next observation
        if not terminated:
            next_obs = {"instruction": self.instructions[self.current_idx]}
        else:
            next_obs = {"instruction": ""}  # Dummy observation

        # Train student and update g_val periodically
        if self.step_count % self.student_update_freq == 0:
            self.train_student(self.student_train_steps)
            self.g_val = compute_validation_gradient(
                self.student_model, self.val_dataloader, self.val_samples
            )

        # Info dict
        info = {
            "cost": cost,
            "action": action,
            "dataset_size": len(self.dataset_D),
            "instruction_idx": self.current_idx - 1,
        }

        return next_obs, reward, terminated, truncated, info

    def get_dataset(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get accumulated dataset D.

        Returns:
            List of (input_ids, labels) tuples
        """
        return self.dataset_D

    def train_student(self, num_steps: int = 1):
        """Train student model on accumulated dataset.

        Args:
            num_steps: Number of training steps
        """
        if len(self.dataset_D) == 0:
            return

        # Simple training loop
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-4)

        for _ in range(num_steps):
            # Sample batch from dataset D
            batch_size = min(32, len(self.dataset_D))
            indices = np.random.choice(len(self.dataset_D), batch_size, replace=False)
            batch = [self.dataset_D[i] for i in indices]

            # Stack batch
            input_ids = torch.stack([x[0] for x in batch])
            labels = torch.stack([x[1] for x in batch])

            # Forward pass
            loss = self.student_model(input_ids, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
