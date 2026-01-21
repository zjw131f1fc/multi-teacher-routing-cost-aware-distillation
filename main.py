"""Main entry point for RL-based teacher selection training.

Usage:
    python main.py [--config CONFIG_PATH]

Example:
    python main.py --config configs/rl_teacher_selection.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from methods.rl_teacher_selection.sb3_env import TeacherSelectionEnv
from methods.rl_teacher_selection.teacher_pool import Teacher, TeacherPool
from methods.rl_teacher_selection.mock_data import (
    get_mock_instructions,
    get_mock_student_model,
    get_mock_teachers,
    get_mock_val_loader,
)


class TrainingCallback(BaseCallback):
    """Callback for logging training progress."""

    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.total_cost = 0.0
        self.teacher_counts = {}

    def _on_step(self) -> bool:
        # Track costs and teacher selections
        if "cost" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]
            self.total_cost += info.get("cost", 0)
            action = info.get("action", -1)
            self.teacher_counts[action] = self.teacher_counts.get(action, 0) + 1

        # Log periodically
        if self.n_calls % self.eval_freq == 0:
            print(f"\n[Step {self.n_calls}]")
            print(f"  Total cost: {self.total_cost:.2f}")
            print(f"  Teacher selections: {self.teacher_counts}")

        return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "policy_settings": {
            "bert_model": "bert-base-uncased",
            "mlp_hidden_dims": [512, 256],
            "freeze_bert": True,
            "max_time_steps": 100000,
            "time_embedding_dim": 64,
        },
        "teacher_pool": [
            {"name": "gpt-4", "cost": 1.0},
            {"name": "gpt-3.5", "cost": 0.1},
            {"name": "llama-70b", "cost": 0.5},
        ],
        "training_settings": {
            "batch_size": 1000,
            "policy_update_freq": 200,
            "eval_freq": 5,
            "total_timesteps": 10000,
        },
        "gradient_settings": {
            "last_n_layers": 3,
        },
        "reward_settings": {
            "lambda_cost": 0.05,
            "val_samples": 200,
        },
        "ppo_settings": {
            "learning_rate": 1e-4,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "n_steps": 128,
            "batch_size": 64,
        },
    }


def setup_environment(config: Dict[str, Any], device: str = "cpu"):
    """Setup training environment with mock data.

    Args:
        config: Configuration dictionary
        device: Device to use

    Returns:
        TeacherSelectionEnv instance
    """
    # Get mock data
    instructions = get_mock_instructions(config["training_settings"]["batch_size"])
    student_model = get_mock_student_model(device)
    mock_teachers = get_mock_teachers()
    val_loader = get_mock_val_loader()

    # Create teacher pool
    teachers = [
        Teacher(
            name=t.name,
            model=t,
            cost=t.cost,
            generate_fn=lambda model, instr: model.generate(instr),
        )
        for t in mock_teachers
    ]
    teacher_pool = TeacherPool(teachers)

    # Environment config
    env_config = {
        "val_samples": config["reward_settings"]["val_samples"],
        "lambda_cost": config["reward_settings"]["lambda_cost"],
        "student_update_freq": config["training_settings"]["policy_update_freq"],
        "student_train_steps": 10,
        "last_n_layers": config["gradient_settings"]["last_n_layers"],
    }

    # Create environment
    env = TeacherSelectionEnv(
        instructions=instructions,
        student_model=student_model,
        teacher_pool=teacher_pool,
        val_dataloader=val_loader,
        config=env_config,
    )

    return env


def train(config: Dict[str, Any], device: str = "cpu"):
    """Run training.

    Args:
        config: Configuration dictionary
        device: Device to use
    """
    print("Setting up environment...")
    env = setup_environment(config, device)

    print("Creating PPO model...")
    ppo_config = config["ppo_settings"]
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=ppo_config["learning_rate"],
        clip_range=ppo_config["clip_range"],
        vf_coef=ppo_config["vf_coef"],
        ent_coef=ppo_config["ent_coef"],
        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        verbose=1,
    )

    print("Starting training...")
    callback = TrainingCallback(eval_freq=100)
    model.learn(
        total_timesteps=config["training_settings"]["total_timesteps"],
        callback=callback,
    )

    print("\nTraining complete!")
    print(f"Total cost: {callback.total_cost:.2f}")
    print(f"Teacher selections: {callback.teacher_counts}")

    return model


def main():
    parser = argparse.ArgumentParser(description="RL-based Teacher Selection Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        print(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        print("Using default config")
        config = get_default_config()

    print(f"Using device: {args.device}")
    print(f"Config: {config}")

    # Train
    model = train(config, args.device)

    # Save model
    save_path = "rl_teacher_selection_model.zip"
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
