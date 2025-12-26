"""DeBERTa-v3 Encoder Backbone 实现。

实现 DeBERTa-v3 模型的基础功能：
- 加载模型和分词器
- forward: 前向传播，返回隐藏状态
- encode: 编码文本，返回句子表示
"""

import torch
import logging
from typing import Dict, Any, Optional, Union
from transformers import AutoModel, AutoTokenizer

from ...base.encoder import BaseEncoderBackbone


class DeBERTaEncoderBackbone(BaseEncoderBackbone):
    """DeBERTa-v3 Encoder Backbone 实现。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化 DeBERTa Backbone。

        参数:
            config: Config 实例，包含 backbone_settings 等配置
        """
        super().__init__(config)

        # 只有在提供配置时才加载模型
        if config is not None:
            self.device = self.config.global_settings["device"]  # type: ignore
            self.encoder_cfg = self.backbone_cfg["encoder_settings"]

            self._load_model()

            # 获取模型实际所在的设备
            if self.model is not None:
                model_device = next(self.model.parameters()).device
                self.device = str(model_device)

            self.output_device = torch.device(self.device)

            # 池化模式
            self.pooling_mode = self.encoder_cfg.get("pooling_mode", "mean")
        else:
            self.device = "cpu"
            self.output_device = torch.device("cpu")
            self.encoder_cfg = {}
            self.pooling_mode = "mean"

    def _load_model(self):
        """加载 DeBERTa 模型和分词器。"""
        logger = getattr(self.config, "logger", None) or logging.getLogger(__name__)

        # 从配置获取模型 ID
        model_id = self.backbone_cfg.get("model_id", None)
        if model_id is None:
            # 如果未指定，根据 name 映射
            model_name = self.model_name
            model_map = {
                "deberta-v3-base": "microsoft/deberta-v3-base",
                "deberta-v3-large": "microsoft/deberta-v3-large",
                "deberta-v3-small": "microsoft/deberta-v3-small",
            }
            model_id = model_map.get(model_name, "microsoft/deberta-v3-base")

        logger.info(f"Loading DeBERTa Encoder: {model_id}...")
        device_map = self.encoder_cfg.get("device_map", "auto")

        # 从全局配置获取 dtype
        dtype_str = self.config.global_settings.get("dtype", "float32")
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        model_dtype = dtype_map.get(dtype_str, torch.float32)

        # 如果是 CPU，强制使用 FP32
        if device_map == "cpu" or self.device == "cpu":
            model_dtype = torch.float32
            device_map = "cpu"

        logger.info(f"Using dtype: {model_dtype}")

        # 加载模型
        self.model = AutoModel.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=model_dtype,
            trust_remote_code=True,
        )

        # 加载分词器（直接使用 DebertaV2Tokenizer 避免 tiktoken 依赖）
        from transformers import DebertaV2Tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # 打印模型信息
        num_layers = len(self.model.encoder.layer)
        hidden_size = self.model.config.hidden_size
        vocab_size = self.model.config.vocab_size

        logger.info(f"[DeBERTa] Model: {model_id}")
        logger.info(f"[DeBERTa] Layers: {num_layers} | Hidden: {hidden_size} | Vocab: {vocab_size}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """前向传播。

        参数:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            return_dict: 是否返回字典格式
            **kwargs: 其他参数传递给模型

        返回:
            模型输出（BaseModelOutput 或 tuple）
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            **kwargs
        )

    def encode(
        self,
        texts: Union[str, list],
        pooling_mode: Optional[str] = None,
        return_tensor: bool = True,
        **kwargs
    ):
        """编码文本为向量表示。

        参数:
            texts: 单个文本或文本列表
            pooling_mode: 池化模式 ("mean", "cls", "max")，默认使用配置中的模式
            return_tensor: 是否返回 tensor，否则返回 numpy
            **kwargs: 传递给 tokenizer 的其他参数

        返回:
            编码向量 [batch_size, hidden_size] 或 [hidden_size]（单文本时）
        """
        # 确保 texts 是列表
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # 分词
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **kwargs
        )

        # 移动到模型设备
        encoded = {k: v.to(self.output_device) for k, v in encoded.items()}

        # 前向传播
        with torch.no_grad():
            outputs = self.forward(**encoded, return_dict=True)

        # 获取隐藏状态
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        attention_mask = encoded["attention_mask"]  # [batch_size, seq_len]

        # 池化
        pooling_mode = pooling_mode or self.pooling_mode
        if pooling_mode == "cls":
            # 使用 [CLS] token 的表示
            pooled = last_hidden_state[:, 0, :]
        elif pooling_mode == "mean":
            # 平均池化（考虑 attention_mask）
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif pooling_mode == "max":
            # 最大池化
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[mask_expanded == 0] = -1e9  # 忽略 padding
            pooled = torch.max(last_hidden_state, dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling_mode}")

        # 转换为 numpy（如果需要）
        if not return_tensor:
            pooled = pooled.cpu().numpy()

        # 如果是单文本，返回单个向量
        if is_single:
            return pooled[0]

        return pooled
