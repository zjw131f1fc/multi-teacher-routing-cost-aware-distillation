"""HuggingFace Trainer 包装器。

将 HuggingFace Transformers 的 Trainer 集成到框架中，提供：
- 成熟的训练管道（DeepSpeed、FSDP、梯度累积等）
- 完善的分布式支持
- 丰富的回调系统
- 自动优化（混合精度、gradient accumulation）

使用方式:
    在配置文件中设置:
    trainer_settings:
      type: "dl"
      name: "hf-trainer"
      dl_settings:
        batch_size: 4
        epochs: 3
      hf_settings:  # 直接映射到 TrainingArguments
        gradient_accumulation_steps: 4
        fp16: true
        learning_rate: 5.0e-5
        warmup_ratio: 0.1
        logging_steps: 100
        eval_steps: 500
        save_steps: 500
        save_total_limit: 3
        ...

注意：
- 不强制复用 train_step/eval_step 函数
- 按照 HuggingFace Trainer 的自然方式使用
- 通过继承 Trainer 和自定义 data_collator 来适配你的数据格式
"""

from typing import Any, Dict, Optional
import os
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """适配器：将我们的数据格式转换为 HuggingFace Dataset 格式"""

    def __init__(self, samples):
        """
        Args:
            samples: 可以是 DistillDataset、QADataset 或普通列表
        """
        if hasattr(samples, 'samples'):
            # DistillDataset / QADataset
            self.samples = samples.samples
        elif isinstance(samples, list):
            self.samples = samples
        else:
            # 假设是可迭代对象
            self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class HFTrainerWrapper:
    """HuggingFace Trainer 包装器

    提供配置驱动的接口，兼容框架的 load_trainer 调用方式。
    """

    def __init__(self, config: Any, dataset_bundle: Dict[str, Any]):
        """
        Args:
            config: 配置对象（包含 trainer_settings, global_settings 等）
            dataset_bundle: 数据集 bundle（包含 splits, meta, judge 等）
        """
        self.config = config
        self.logger = getattr(config, "logger", None) or config.get("logger")
        self.dataset_bundle = dataset_bundle
        self.splits = dataset_bundle["splits"]
        self.meta = dataset_bundle["meta"]

        # 注册的模型和 tokenizer
        self.model = None
        self.tokenizer = None
        self.models = {}  # 兼容接口：存储额外的注册模型

        # HuggingFace Trainer 实例（延迟初始化）
        self.hf_trainer: Optional[Trainer] = None

        # 从配置中提取设置
        if isinstance(config, dict):
            ts = config["trainer_settings"]["dl_settings"]
            self.hf_settings = config["trainer_settings"].get("hf_settings", {})
            gs = config["global_settings"]
        else:
            ts = (
                config.trainer_settings["dl_settings"]
                if isinstance(config.trainer_settings, dict)
                else config.trainer_settings.dl_settings
            )
            self.hf_settings = (
                config.trainer_settings.get("hf_settings", {})
                if isinstance(config.trainer_settings, dict)
                else getattr(config.trainer_settings, "hf_settings", {})
            )
            gs = config.global_settings

        self.batch_size = int(ts["batch_size"])
        self.epochs = int(ts["epochs"])
        self.save_dir = gs["save_dir"] if isinstance(gs, dict) else gs.save_dir
        self.experiment_tag = gs["experiment_tag"] if isinstance(gs, dict) else gs.experiment_tag

        if self.logger:
            self.logger.info("HFTrainerWrapper 初始化完成")
            self.logger.info(f"  - Batch size: {self.batch_size}, Epochs: {self.epochs}")

    # ==================== 模型注册（兼容接口）====================

    def register_model(self, name: str, model_or_obj: Any):
        """注册模型或其他对象

        Args:
            name: 名称（"student", "tokenizer" 等）
            model_or_obj: 模型实例、tokenizer 或其他对象
        """
        self.models[name] = model_or_obj

        # 特殊处理：自动识别主模型和 tokenizer
        if name == "student" and isinstance(model_or_obj, torch.nn.Module):
            self.model = model_or_obj
        elif name == "tokenizer":
            self.tokenizer = model_or_obj

        if self.logger:
            if isinstance(model_or_obj, torch.nn.Module):
                total_params = sum(p.numel() for p in model_or_obj.parameters())
                trainable_params = sum(
                    p.numel() for p in model_or_obj.parameters() if p.requires_grad
                )
                self.logger.info(
                    f"注册模型: {name} | 总参数量: {total_params:,} | 可训练参数量: {trainable_params:,}"
                )
            else:
                self.logger.info(f"注册对象: {name}")

    def add_param_group(self, name: str, params, **kwargs):
        """兼容接口：HF Trainer 自动管理参数组"""
        if self.logger:
            self.logger.info(f"参数组 {name} 已注册（HF Trainer 自动管理优化器）")

    def setup_optimizers(self):
        """兼容接口：HF Trainer 自动设置优化器"""
        if self.logger:
            self.logger.info("优化器将由 HF Trainer 自动创建")

    def register_train_step(self, fn):
        """兼容接口：HF Trainer 使用内置的训练逻辑"""
        if self.logger:
            self.logger.warning(
                "HF Trainer 使用内置的训练逻辑，train_step 函数将被忽略。"
                "如需自定义训练逻辑，请继承 Trainer 类并重写 compute_loss 方法。"
            )

    def register_eval_step(self, fn):
        """兼容接口：HF Trainer 使用内置的评估逻辑"""
        if self.logger:
            self.logger.warning(
                "HF Trainer 使用内置的评估逻辑，eval_step 函数将被忽略。"
                "如需自定义评估逻辑，请继承 Trainer 类并重写 evaluation_loop 方法。"
            )

    # ==================== 核心方法 ====================

    def _prepare_hf_trainer(self):
        """准备 HuggingFace Trainer 实例"""
        if self.hf_trainer is not None:
            return  # 已经初始化

        if self.model is None:
            raise ValueError("未注册主模型！请使用 register_model('student', model) 注册模型")

        # 准备数据集
        train_dataset = CustomDataset(self.splits["train"])
        eval_dataset = CustomDataset(self.splits["test"]) if "test" in self.splits else None

        # 构建输出目录
        output_dir = os.path.join(self.save_dir, self.experiment_tag)
        os.makedirs(output_dir, exist_ok=True)

        # 默认 TrainingArguments 参数
        default_args = {
            "output_dir": output_dir,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "num_train_epochs": self.epochs,
            "logging_dir": os.path.join(output_dir, "logs"),
            "logging_steps": 100,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch" if eval_dataset else "no",
            "save_total_limit": 3,
            "load_best_model_at_end": True if eval_dataset else False,
            "report_to": "none",  # 不使用 wandb/tensorboard
            "remove_unused_columns": False,  # 保留所有数据列
            "dataloader_num_workers": 0,  # 避免多进程问题
        }

        # 合并用户自定义的 hf_settings
        # 用户配置会覆盖默认配置
        training_args_dict = {**default_args, **self.hf_settings}
        training_args = TrainingArguments(**training_args_dict)

        # 自定义 data collator
        # 这里可以根据你的数据格式定制
        def custom_collate_fn(batch):
            """保持原始数据格式，不做额外处理"""
            return batch

        # 创建 Trainer
        self.hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,  # 用于自动保存
            data_collator=custom_collate_fn,
        )

        if self.logger:
            self.logger.info(f"HuggingFace Trainer 已准备")
            self.logger.info(f"  - 输出目录: {output_dir}")
            self.logger.info(f"  - 训练集大小: {len(train_dataset)}")
            if eval_dataset:
                self.logger.info(f"  - 评估集大小: {len(eval_dataset)}")

    def run(self):
        """启动训练（兼容 BasicPytorchTrainer.run() 接口）"""
        self._prepare_hf_trainer()

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("开始训练（使用 HuggingFace Trainer）")
            self.logger.info("=" * 60)

        # 训练
        train_result = self.hf_trainer.train()

        # 保存模型
        self.hf_trainer.save_model()

        # 评估
        eval_result = {}
        if "test" in self.splits:
            eval_result = self.hf_trainer.evaluate()
            if self.logger:
                self.logger.info(f"评估结果: {eval_result}")

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("训练完成！")
            self.logger.info("=" * 60)

        # 返回结果（兼容接口）
        return {
            "score": eval_result.get("eval_loss", 0),
            "train_result": train_result,
            "eval_result": eval_result,
        }

    def evaluate(self, max_samples: Optional[int] = None, trigger_info: Optional[Dict] = None):
        """评估模型（兼容接口）"""
        self._prepare_hf_trainer()

        if "test" not in self.splits:
            if self.logger:
                self.logger.info("测试集不存在，跳过评估")
            return {}

        # TODO: 支持 max_samples 抽样
        if max_samples is not None and self.logger:
            self.logger.warning("HF Trainer 暂不支持 max_samples 参数，将评估全部测试集")

        eval_result = self.hf_trainer.evaluate()

        if self.logger:
            self.logger.info(f"评估结果: {eval_result}")

        return eval_result
