"""RouterClassifier 模型: 使用 DeBERTa Encoder 预测教师 NTE 桶（分类模式）

架构:
    输入: instruction (问题文本，带prompt)
        ↓
    DeBERTa Encoder (microsoft/deberta-v3-base, 全量微调)
        ↓
    Mean Pooling (对所有非padding token取平均)
        ↓
    LayerNorm (稳定数值范围)
        ↓
    每个教师独立的分类头:
        - Teacher 1 Head: hidden_dim → hidden_dim//4 → num_buckets
        - Teacher 2 Head: hidden_dim → hidden_dim//4 → num_buckets
        - ...
        ↓
    输出: {
        teacher1: [logits for bucket_0, bucket_1, ..., bucket_K-1],
        teacher2: [logits for bucket_0, bucket_1, ..., bucket_K-1],
        ...
    }

桶定义（使用自适应分桶）:
    - Bucket 0 (D级): 最低 NTE 区间
    - Bucket 1 (C级): 较低 NTE 区间
    - Bucket 2 (B级): 中等 NTE 区间
    - Bucket 3 (A级): 较高 NTE 区间
    - Bucket 4 (S级): 最高 NTE 区间
"""

import torch
import torch.nn as nn
from typing import List, Dict


class RouterClassifier(nn.Module):
    """路由分类模型: 使用 DeBERTa Encoder 预测每个教师的 NTE 桶（分类模式）

    参数:
        encoder_backbone: Encoder backbone (DeBERTa)
        num_teachers: 教师数量
        teacher_names: 教师名称列表 (用于索引)
        num_buckets: 桶数量 (default=5)
        hidden_dim: Encoder hidden dimension
        dropout: Dropout rate (default=0.1)
    """

    def __init__(
        self,
        encoder_backbone,
        num_teachers: int,
        teacher_names: List[str],
        num_buckets: int = 5,
        hidden_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()

        # DeBERTa Encoder backbone (将被全量微调)
        self.encoder = encoder_backbone
        self.tokenizer = encoder_backbone.tokenizer

        # 获取 encoder 的 device 和 dtype
        encoder_device = encoder_backbone.output_device
        encoder_dtype = next(encoder_backbone.model.parameters()).dtype

        # LayerNorm: 稳定 hidden state 的数值范围
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 为每个教师创建独立的分类头
        # 这样可以让不同教师学习不同的决策边界
        self.teacher_heads = nn.ModuleDict()
        intermediate_dim = hidden_dim // 4  # 768 -> 192

        # 创建教师名称到模块键的映射（PyTorch ModuleDict 不允许键名中有 "."）
        self.teacher_name_to_key = {}
        for teacher_name in teacher_names:
            # 将 "." 和 "-" 替换为 "_"
            module_key = teacher_name.replace(".", "_").replace("-", "_")
            self.teacher_name_to_key[teacher_name] = module_key

            self.teacher_heads[module_key] = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, num_buckets)  # 输出 K 个桶的 logits
            )

        # 将 LayerNorm 和所有分类头转换为与 encoder 相同的 dtype 和设备
        self.layer_norm = self.layer_norm.to(dtype=encoder_dtype, device=encoder_device)
        for head in self.teacher_heads.values():
            head.to(dtype=encoder_dtype, device=encoder_device)

        self.num_teachers = num_teachers
        self.teacher_names = teacher_names
        self.num_buckets = num_buckets
        self.hidden_dim = hidden_dim

    def forward(
        self,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """前向传播

        参数:
            texts: 批次文本列表

        返回:
            logits: {
                teacher_name: [batch_size, num_buckets] - 该教师的桶分类 logits
            }
        """
        # 使用 encoder 的 encode() 方法，内部已实现 tokenization + mean pooling
        pooled = self.encoder.encode(
            texts,
            pooling_mode="mean",
            return_tensor=True
        )  # [batch_size, hidden_dim]

        # LayerNorm 稳定数值范围
        pooled = self.layer_norm(pooled)

        # 为每个教师独立预测桶分类 logits
        logits = {}
        for teacher_name in self.teacher_names:
            module_key = self.teacher_name_to_key[teacher_name]
            logits[teacher_name] = self.teacher_heads[module_key](pooled)  # [batch_size, num_buckets]

        return logits


# 向后兼容：保留旧名称的别名
RouterRegressor = RouterClassifier
