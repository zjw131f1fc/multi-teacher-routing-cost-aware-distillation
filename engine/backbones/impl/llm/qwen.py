"""Qwen2.5-0.5B-Instruct LLM Backbone 实现。

实现 Qwen2.5 模型的基础功能：
- 加载模型和分词器
- forward: 前向传播
- generate: 文本生成
"""

import torch
import logging
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...base.llm import BaseLLMBackbone


class QwenLLMBackbone(BaseLLMBackbone):
    """Qwen2.5-0.5B-Instruct LLM Backbone 实现。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化 Qwen Backbone。

        参数:
            config: Config 实例，包含 backbone_settings 等配置
        """
        super().__init__(config)
        self.device = self.config.global_settings["device"]  # type: ignore
        self.llm_cfg = self.backbone_cfg["llm_settings"]

        self._load_model()

        # 获取模型实际所在的设备
        if self.model is not None:
            model_device = next(self.model.parameters()).device
            self.device = str(model_device)

        self.output_device = torch.device(self.device)

    def _load_model(self):
        """加载 Qwen 模型和分词器。"""
        logger = getattr(self.config, "logger", None) or logging.getLogger(__name__)

        # 从配置获取模型 ID
        model_id = self.backbone_cfg.get("model_id", None)
        if model_id is None:
            # 如果未指定，根据 name 映射
            model_name = self.model_name
            model_map = {
                "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
                "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            }
            model_id = model_map.get(model_name, "Qwen/Qwen2.5-0.5B-Instruct")

        logger.info(f"Loading Qwen LLM: {model_id}...")
        device_map = self.llm_cfg.get("device_map", "auto")

        # 从全局配置获取 dtype
        dtype_str = self.config.global_settings.get("dtype", "float16")
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        model_dtype = dtype_map.get(dtype_str, torch.float16)

        # 如果是 CPU，强制使用 FP32
        if device_map == "cpu" or self.device == "cpu":
            model_dtype = torch.float32
            device_map = "cpu"

        logger.info(f"Using dtype: {model_dtype}")

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 打印模型信息
        num_layers = len(self.model.model.layers)
        hidden_size = self.model.config.hidden_size
        vocab_size = self.model.config.vocab_size

        logger.info(f"[Qwen] Model: {model_id}")
        logger.info(f"[Qwen] Layers: {num_layers} | Hidden: {hidden_size} | Vocab: {vocab_size}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """前向传播。

        参数:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: 其他参数传递给模型

        返回:
            模型输出
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **kwargs
    ):
        """生成文本。

        参数:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: 最大生成长度
            **kwargs: 其他生成参数

        返回:
            生成的 token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
