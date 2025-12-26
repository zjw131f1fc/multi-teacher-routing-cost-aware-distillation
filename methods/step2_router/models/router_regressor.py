"""RouterRegressor 模型: 使用 DeBERTa Encoder 预测教师 NTE 分数

架构:
    输入: instruction (问题文本，带prompt)
        ↓
    DeBERTa Encoder (microsoft/deberta-v3-base, 全量微调)
        ↓
    Mean Pooling (对所有非padding token取平均)
        ↓
    MLP Head: hidden_dim → hidden_dim//4 → num_teachers
        ↓
    Sigmoid + Scale (约束到0-1范围)
        ↓
    输出: [score_teacher1, score_teacher2, ...] (每个教师的 NTE 分数, 0-1)
"""

import torch
import torch.nn as nn
from typing import List


class RouterRegressor(nn.Module):
    """路由回归模型: 使用 DeBERTa Encoder 预测每个教师的 NTE 分数

    参数:
        encoder_backbone: Encoder backbone (DeBERTa)
        num_teachers: 教师数量
        hidden_dim: Encoder hidden dimension
        dropout: Dropout rate (default=0.1)
        score_scale: 分数缩放因子 (default=1.0)
    """

    def __init__(
        self,
        encoder_backbone,
        num_teachers: int,
        hidden_dim: int,
        dropout: float = 0.1,
        score_scale: float = 1.0
    ):
        super().__init__()

        # DeBERTa Encoder backbone (将被全量微调)
        self.encoder = encoder_backbone
        self.tokenizer = encoder_backbone.tokenizer

        # 获取 encoder 的 device 和 dtype
        encoder_device = encoder_backbone.output_device
        encoder_dtype = next(encoder_backbone.model.parameters()).dtype

        # LayerNorm: 稳定 hidden state 的数值范围，防止梯度爆炸
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 回归头: 输出 num_teachers 个分数
        intermediate_dim = hidden_dim // 4  # 768 -> 192
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, num_teachers)
        )

        # 将 LayerNorm 和回归头转换为与 encoder 相同的 dtype 和设备
        self.layer_norm = self.layer_norm.to(dtype=encoder_dtype, device=encoder_device)
        self.head = self.head.to(dtype=encoder_dtype, device=encoder_device)

        self.num_teachers = num_teachers
        self.hidden_dim = hidden_dim
        self.score_scale = score_scale

    def forward(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """前向传播

        参数:
            texts: 批次文本列表

        返回:
            scores: [batch_size, num_teachers] - 预测的NTE分数 (0-1)
        """
        # 使用 encoder 的 encode() 方法，内部已实现 tokenization + mean pooling
        # pooling_mode="mean" 会对所有非padding token取平均
        pooled = self.encoder.encode(
            texts,
            pooling_mode="mean",
            return_tensor=True
        )  # [batch_size, hidden_dim]

        # LayerNorm 稳定数值范围
        pooled = self.layer_norm(pooled)

        # 回归头: 使用sigmoid约束到[0, 1]，然后缩放到[0, score_scale]
        logits = self.head(pooled)  # [batch_size, num_teachers]
        scores = torch.sigmoid(logits) * self.score_scale

        return scores
