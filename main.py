"""Main entry point for RL-based teacher selection training."""

from pathlib import Path

from configs import load_config
from datas import load_dataset
from models import load_student, load_student_tokenizer, load_teachers
from teachers import TeacherPool
from rl import TeacherSelectionEnv, PolicyNetwork

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from transformers import AutoModel, AutoTokenizer


class TrainingCallback(BaseCallback):
    """Callback for logging training progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            print(f"Step {self.n_calls}")
        return True


def main(config_path: str):
    # 1. 加载配置
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")

    # 2. 加载数据集
    dataset = load_dataset(config)
    print(f"Loaded dataset with {len(dataset)} samples")

    # 3. 加载 student 模型和 tokenizer
    student = load_student(config)
    tokenizer = load_student_tokenizer(config)
    print(f"Loaded student model")

    # 4. 加载 teacher 模型
    teachers = load_teachers(config)
    print(f"Loaded {len(teachers)} teachers: {list(teachers.keys())}")

    # 5. 创建 TeacherPool
    teacher_pool = TeacherPool(config, dataset, teachers)
    print("Created TeacherPool")

    # 6. 准备 instructions 和 val_dataloader
    instructions = [sample["instruction"] for sample in dataset.samples]
    val_dataloader = None  # TODO: 从 config 加载验证集

    # 7. 创建环境
    env_config = {
        "teacher_names": list(teachers.keys()),
        "teacher_costs": config.get("teacher_costs", {}),
        "val_samples": config.get("rl", {}).get("val_samples", 200),
        "lambda_cost": config.get("rl", {}).get("lambda_cost", 0.05),
        "student_update_freq": config.get("rl", {}).get("student_update_freq", 100),
        "student_train_steps": config.get("rl", {}).get("student_train_steps", 10),
        "last_n_layers": config.get("rl", {}).get("last_n_layers", 3),
    }
    env = TeacherSelectionEnv(
        instructions=instructions,
        student_model=student,
        teacher_pool=teacher_pool,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        config=env_config,
    )
    print("Created TeacherSelectionEnv")

    # 8. 创建 Policy Network (用于 PPO 的 feature extractor)
    policy_config = config.get("policy", {})
    encoder_name = policy_config.get("encoder_name", "bert-base-uncased")
    encoder = AutoModel.from_pretrained(encoder_name)
    policy_tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # 9. 训练 PPO
    ppo_config = config.get("ppo", {})
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=ppo_config.get("learning_rate", 3e-4),
        n_steps=ppo_config.get("n_steps", 128),
        batch_size=ppo_config.get("batch_size", 64),
        verbose=1,
    )
    print("Created PPO model")

    total_timesteps = config.get("training", {}).get("total_timesteps", 10000)
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=TrainingCallback(),
    )

    # 10. 保存模型
    save_path = config.get("training", {}).get("save_path", "./output/model")
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    main(args.config)
