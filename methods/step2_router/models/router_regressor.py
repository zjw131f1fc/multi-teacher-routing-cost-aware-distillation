"""RouterRegressor 模型: 使用 LLM 预测教师 NTE 分数

架构:
    输入: instruction (问题文本，带prompt)
        ↓
    LLM (Qwen-1.5B, 全量微调)
        ↓
    Last Token Pooling (取最后一个非padding token的hidden state)
        ↓
    MLP Head: hidden_dim → hidden_dim//4 → num_teachers
        ↓
    Sigmoid + Scale (约束到0-10范围)
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
        score_scale: 分数缩放因子 (default=10.0, NTE分数范围0-10)
    """

    def __init__(
        self,
        llm_backbone,
        num_teachers: int,
        hidden_dim: int,
        dropout: float = 0.1,
        score_scale: float = 10.0
    ):
        super().__init__()

        # LLM backbone (将被全量微调)
        self.llm = llm_backbone.model
        self.tokenizer = llm_backbone.tokenizer

        # 获取LLM的dtype和最后一层所在的设备
        # 对于device_map分布的模型，最后一层的输出设备用于放置回归头
        llm_dtype = next(self.llm.parameters()).dtype

        # 尝试获取最后一层的设备（通常是输出层）
        try:
            # 对于decoder模型，最后一层通常是lm_head或类似的
            if hasattr(self.llm, 'lm_head'):
                last_device = next(self.llm.lm_head.parameters()).device
            else:
                # 如果没有lm_head，使用最后一个transformer层的设备
                last_device = next(self.llm.model.layers[-1].parameters()).device
        except:
            # 如果获取失败，使用第一个参数的设备
            last_device = next(self.llm.parameters()).device

        # LayerNorm: 稳定 hidden state 的数值范围，防止梯度爆炸
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 增强的回归头: 添加隐藏层以提高表达能力
        intermediate_dim = hidden_dim // 4  # 1536 -> 384
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, num_teachers)
        )

        # 将回归头和LayerNorm转换为与LLM相同的dtype和设备
        self.layer_norm = self.layer_norm.to(dtype=llm_dtype, device=last_device)
        self.head = self.head.to(dtype=llm_dtype, device=last_device)

        self.num_teachers = num_teachers
        self.hidden_dim = hidden_dim
        self.score_scale = score_scale

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
        # 使用clamp确保索引不会为负数
        seq_lengths = (attention_mask.sum(dim=1) - 1).clamp(min=0)  # [batch_size]

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
        # 获取 LLM 第一层的设备（embedding 层所在设备）
        llm_device = next(self.llm.parameters()).device

        # 将输入移到 LLM 的设备上
        input_ids = input_ids.to(llm_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(llm_device)

        # 前向传播获取 hidden states
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 获取最后一层的 hidden states 并立即 detach 以释放计算图
        last_hidden_states = outputs.hidden_states[-1].detach()  # [batch_size, seq_len, hidden_dim]

        # 清理 outputs 以释放显存
        del outputs

        # Last Token Pooling
        pooled = self.last_token_pooling(last_hidden_states, attention_mask)  # [batch_size, hidden_dim]

        # 清理 last_hidden_states
        del last_hidden_states

        # LayerNorm 稳定数值范围
        pooled = self.layer_norm(pooled)

        # 回归头预测分数
        logits = self.head(pooled)  # [batch_size, num_teachers]

        # 使用sigmoid约束到[0, 1]，然后缩放到[0, score_scale]
        scores = torch.sigmoid(logits) * self.score_scale

        return scores
