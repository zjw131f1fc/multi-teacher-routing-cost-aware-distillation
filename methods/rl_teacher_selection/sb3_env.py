"""Gymnasium environment for RL-based teacher selection.

This environment integrates all components:
- Policy network (handled by SB3)
- Teacher pool
- Gradient computation (last N layers only)
- Reward computation (with log-scaled gradient norm)
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

    Observation: Current instruction + time step
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
                - last_n_layers: Number of last layers for gradient computation (default: 3)
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
        self.last_n_layers = self.config.get("last_n_layers", 3)

        # Action space: [0, num_teachers-1] for selecting teachers
        num_teachers = len(teacher_pool)
        self.action_space = gym.spaces.Discrete(num_teachers)

        # Observation space: instruction text + time step
        self.observation_space = gym.spaces.Dict({
            "instruction": gym.spaces.Text(max_length=1000),
            "time_step": gym.spaces.Discrete(100000),
        })

        # Training state
        self.current_idx = 0
        self.batch_data = []  # Current batch data for student training
        self.g_val = None  # Validation gradient direction
        self.global_step = 0

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
        self.global_step = 0
        self.batch_data = []

        # Compute initial validation gradient (last N layers only)
        self.g_val = compute_validation_gradient(
            self.student_model, self.val_dataloader,
            self.val_samples, self.last_n_layers
        )

        obs = {
            "instruction": self.instructions[self.current_idx],
            "time_step": self.global_step,
        }
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

        # Compute gradient for this sample (last N layers only)
        g_sample = compute_gradient(
            self.student_model, input_ids, labels, self.last_n_layers
        )

        # Compute reward (with log-scaled gradient norm)
        reward = compute_reward(g_sample, self.g_val, cost, self.lambda_cost)

        # Add to batch data
        self.batch_data.append((input_ids, labels))

        # Move to next instruction
        self.current_idx += 1
        self.global_step += 1

        # Check if episode is done
        terminated = self.current_idx >= len(self.instructions)
        truncated = False

        # Prepare next observation
        if not terminated:
            next_obs = {
                "instruction": self.instructions[self.current_idx],
                "time_step": self.global_step,
            }
        else:
            next_obs = {
                "instruction": "",
                "time_step": self.global_step,
            }

        # Train student and update g_val periodically
        if self.global_step % self.student_update_freq == 0:
            self.train_student(self.student_train_steps)
            self.g_val = compute_validation_gradient(
                self.student_model, self.val_dataloader,
                self.val_samples, self.last_n_layers
            )

        # Info dict
        info = {
            "cost": cost,
            "action": action,
            "batch_size": len(self.batch_data),
            "instruction_idx": self.current_idx - 1,
            "global_step": self.global_step,
        }

        return next_obs, reward, terminated, truncated, info

    def get_batch_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get current batch data.

        Returns:
            List of (input_ids, labels) tuples
        """
        return self.batch_data

    def train_student(self, num_steps: int = 1):
        """Train student model on batch data.

        Args:
            num_steps: Number of training steps
        """
        if len(self.batch_data) == 0:
            return

        # Simple training loop
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-4)

        for _ in range(num_steps):
            # Sample batch from data
            batch_size = min(32, len(self.batch_data))
            indices = np.random.choice(len(self.batch_data), batch_size, replace=False)
            batch = [self.batch_data[i] for i in indices]

            # Stack batch
            input_ids = torch.stack([x[0] for x in batch])
            labels = torch.stack([x[1] for x in batch])

            # Forward pass
            loss = self.student_model(input_ids, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
