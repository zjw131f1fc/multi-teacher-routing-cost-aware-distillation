"""RouterRegressor 模型: 使用 LLM 预测教师 NTE 分数

架构:
    输入: instruction (问题文本)
        ↓
    LLM (Qwen-1.5B, 全量微调)
        ↓
    Last Token Pooling (取最后一个非padding token的hidden state)
        ↓
    Linear Head: hidden_dim → num_teachers
        ↓
    输出: [score_teacher1, score_teacher2, ...] (每个教师的 NTE 分数, 0-10)
"""

import torch
import torch.nn as nn
from typing import Optional


class RouterRegressor(nn.Module):
    """路由回归模型: 预测每个教师的 NTE 分数

    参数:
        llm_backbone: LLM backbone (例如 Qwen)
        num_teachers: 教师数量
        hidden_dim: LLM hidden dimension
        dropout: Dropout rate (default=0.1)
    """

    def __init__(
        self,
        llm_backbone,
        num_teachers: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # LLM backbone (将被全量微调)
        self.llm = llm_backbone.model
        self.tokenizer = llm_backbone.tokenizer

        # 回归头
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_teachers)
        )

        self.num_teachers = num_teachers
        self.hidden_dim = hidden_dim

    def last_token_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Last Token Pooling: 提取每个序列最后一个非padding token的hidden state

        参数:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        返回:
            pooled: [batch_size, hidden_dim]
        """
        # 获取每个序列的最后一个非padding位置
        # attention_mask: 1表示有效token, 0表示padding
        seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size], 最后一个有效token的索引

        batch_size = hidden_states.shape[0]

        # 提取最后一个token的hidden state
        # hidden_states[i, seq_lengths[i], :] 取出第i个样本的最后一个有效token
        last_hidden = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            seq_lengths,
            :
        ]  # [batch_size, hidden_dim]

        return last_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        参数:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        返回:
            scores: [batch_size, num_teachers] 预测的NTE分数 (0-10)
        """
        # 前向传播获取 hidden states
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 获取最后一层的 hidden states
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]

        # Last Token Pooling
        pooled = self.last_token_pooling(last_hidden_states, attention_mask)  # [batch_size, hidden_dim]

        # 回归头预测分数
        scores = self.head(pooled)  # [batch_size, num_teachers]

        return scores
