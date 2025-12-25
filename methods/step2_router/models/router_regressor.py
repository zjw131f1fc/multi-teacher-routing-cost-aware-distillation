"""RouterRegressor 模型: 使用 LLM 预测教师 NTE 分数

支持三种模式:
1. 回归模式 (use_bucketing=False, use_score_diff=False):
    输入: instruction (问题文本，带prompt)
        ↓
    LLM (Qwen-1.5B, 全量微调)
        ↓
    Pooling (取最后一个非padding token的hidden state 或 mean pooling)
        ↓
    MLP Head: hidden_dim → hidden_dim//4 → num_teachers
        ↓
    Sigmoid + Scale (约束到0-10范围)
        ↓
    输出: [score_teacher1, score_teacher2, ...] (每个教师的 NTE 分数, 0-10)

2. 分类模式 (use_bucketing=True, use_score_diff=False):
    输入: instruction (问题文本，带prompt)
        ↓
    LLM (Qwen-1.5B, 全量微调)
        ↓
    Pooling (取最后一个非padding token的hidden state 或 mean pooling)
        ↓
    每个教师独立的 MLP Head:
        Teacher 1: hidden_dim → hidden_dim//4 → num_buckets
        Teacher 2: hidden_dim → hidden_dim//4 → num_buckets
        ...
        ↓
    Softmax (每个教师独立)
        ↓
    输出: [batch, num_teachers, num_buckets] (每个教师的桶概率分布)

3. 分数差预测模式 (use_score_diff=True, 仅支持2个教师):
    输入: instruction (问题文本，带prompt)
        ↓
    LLM (Qwen-1.5B, 全量微调)
        ↓
    Pooling
        ↓
    两个独立的MLP Head:
        - Strong Teacher Head: hidden_dim → hidden_dim//4 → 1 (预测强教师的绝对分数)
        - Diff Head: hidden_dim → hidden_dim//4 → 1 (预测弱教师与强教师的分数差)
        ↓
    输出: [score_strong, score_strong - diff] (重构两个教师的分数)

优势：
- 降低任务难度：1个绝对分数 + 1个相对差值
- 适合双教师场景（一强一弱）
- 自动保证强教师分数 ≥ 弱教师分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class RouterRegressor(nn.Module):
    """路由回归/分类模型: 预测每个教师的 NTE 分数或桶分布

    参数:
        llm_backbone: LLM backbone (例如 Qwen)
        num_teachers: 教师数量
        hidden_dim: LLM hidden dimension
        dropout: Dropout rate (default=0.1)
        score_scale: 分数缩放因子 (default=10.0, 仅用于回归模式)
        use_bucketing: 是否使用桶化分类 (default=False)
        num_buckets: 桶的数量 (仅用于分类模式, default=5)
        teacher_bucket_ranges: 教师特定桶边界字典 {teacher_name: [ranges]} (仅用于分类模式)
        teacher_bucket_centers: 教师特定桶中心字典 {teacher_name: [centers]} (仅用于分类模式)
        use_score_diff: 是否使用分数差预测模式 (default=False, 仅支持2个教师)
        strong_teacher_idx: 强教师的索引（用于分数差模式，default=0，值为0或1）
    """

    def __init__(
        self,
        llm_backbone,
        num_teachers: int,
        hidden_dim: int,
        dropout: float = 0.1,
        score_scale: float = 10.0,
        use_bucketing: bool = False,
        num_buckets: int = 5,
        teacher_bucket_ranges: Optional[Dict[str, List[float]]] = None,
        teacher_bucket_centers: Optional[Dict[str, List[float]]] = None,
        pooling_strategy: str = "mean",  # "last" 或 "mean"
        use_score_diff: bool = False,
        strong_teacher_idx: int = 0
    ):
        super().__init__()

        # LLM backbone (将被全量微调)
        self.llm = llm_backbone.model
        self.tokenizer = llm_backbone.tokenizer
        self.pooling_strategy = pooling_strategy

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

        # 根据模式选择不同的输出头
        intermediate_dim = hidden_dim // 4  # 1536 -> 384

        if use_score_diff:
            # 分数差预测模式: 预测强教师分数 + 弱教师与强教师的差值
            # 注意: 此模式与 use_bucketing 互斥，且仅支持 2 个教师
            if use_bucketing:
                raise ValueError("use_score_diff 和 use_bucketing 不能同时为 True")

            if num_teachers != 2:
                raise ValueError(f"分数差预测模式仅支持 2 个教师，当前教师数量: {num_teachers}")

            # Head 1: 预测强教师的分数
            self.strong_head = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, 1)
            )

            # Head 2: 预测弱教师与强教师的分数差（差值应为非正数）
            self.diff_head = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, 1)
            )

            # 将两个head转换为与LLM相同的dtype和设备
            self.strong_head = self.strong_head.to(dtype=llm_dtype, device=last_device)
            self.diff_head = self.diff_head.to(dtype=llm_dtype, device=last_device)

        elif use_bucketing:
            # 分类模式: 每个教师独立的 MLP Head
            # 使用 ModuleList 存储每个教师的分类器
            self.teacher_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, intermediate_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(intermediate_dim, num_buckets)
                )
                for _ in range(num_teachers)
            ])

            # 将所有教师的head转换为与LLM相同的dtype和设备
            for teacher_head in self.teacher_heads:
                teacher_head.to(dtype=llm_dtype, device=last_device)
        else:
            # 回归模式: 输出 num_teachers 个分数（保持不变）
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, num_teachers)
            )
            # 将回归头转换为与LLM相同的dtype和设备
            self.head = self.head.to(dtype=llm_dtype, device=last_device)

        # 将LayerNorm转换为与LLM相同的dtype和设备
        self.layer_norm = self.layer_norm.to(dtype=llm_dtype, device=last_device)

        self.num_teachers = num_teachers
        self.hidden_dim = hidden_dim
        self.score_scale = score_scale
        self.use_bucketing = use_bucketing
        self.use_score_diff = use_score_diff
        self.strong_teacher_idx = strong_teacher_idx
        self.num_buckets = num_buckets

        # 存储教师特定的桶配置（用于推理时将桶概率转换为期望分数）
        if use_bucketing:
            if teacher_bucket_ranges is not None:
                # 教师特定分桶：每个教师有独立的桶边界
                self.teacher_bucket_ranges = teacher_bucket_ranges
            else:
                # 向后兼容：所有教师共享桶边界（设为 None）
                self.teacher_bucket_ranges = None

            if teacher_bucket_centers is not None:
                # 教师特定桶中心
                self.teacher_bucket_centers = teacher_bucket_centers
            else:
                # 向后兼容：所有教师共享桶中心（设为 None）
                self.teacher_bucket_centers = None

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

    def mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean Pooling: 对所有非padding token的hidden state取平均

        参数:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        返回:
            pooled: [batch_size, hidden_dim]
        """
        # 扩展mask到hidden_dim维度
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()  # [batch, seq_len, hidden_dim]

        # 只对非padding位置求和
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # [batch, hidden_dim]

        # 计算每个样本的有效token数
        seq_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]

        # 平均
        mean_hidden = sum_hidden / seq_lengths  # [batch, hidden_dim]

        return mean_hidden

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
            回归模式: scores [batch_size, num_teachers] - 预测的NTE分数 (0-10)
            分类模式: logits [batch_size, num_teachers, num_buckets] - 每个教师的桶 logits
            分数差模式: scores [batch_size, num_teachers] - 预测的NTE分数 (0-10)
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

        # 根据策略选择 Pooling 方法
        if self.pooling_strategy == "mean":
            pooled = self.mean_pooling(last_hidden_states, attention_mask)  # [batch_size, hidden_dim]
        else:  # "last"
            pooled = self.last_token_pooling(last_hidden_states, attention_mask)  # [batch_size, hidden_dim]

        # 清理 last_hidden_states
        del last_hidden_states

        # LayerNorm 稳定数值范围
        pooled = self.layer_norm(pooled)

        if self.use_score_diff:
            # 分数差预测模式：预测强教师分数 + 差值（仅支持2个教师）
            # 1. 预测强教师的分数
            strong_logit = self.strong_head(pooled)  # [batch, 1]
            strong_score = torch.sigmoid(strong_logit) * self.score_scale  # [batch, 1], 范围 [0, 10]

            # 2. 预测弱教师与强教师的分数差（使用 sigmoid 约束到 [0, score_scale]，然后取负）
            diff_logit = self.diff_head(pooled)  # [batch, 1]
            # 差值应该是非正数（弱教师总是不如或等于强教师）
            diff = -(torch.sigmoid(diff_logit) * self.score_scale)  # [batch, 1], 范围 [-10, 0]

            # 3. 重构弱教师的分数
            weak_score = strong_score + diff  # [batch, 1]

            # 4. 根据 strong_teacher_idx 拼接（0=强教师在前，1=弱教师在前）
            if self.strong_teacher_idx == 0:
                # 强教师在第0位，弱教师在第1位
                scores = torch.cat([strong_score, weak_score], dim=-1)  # [batch, 2]
            else:
                # 弱教师在第0位，强教师在第1位
                scores = torch.cat([weak_score, strong_score], dim=-1)  # [batch, 2]

            return scores

        elif self.use_bucketing:
            # 分类模式：每个教师独立预测
            logits_list = []

            for teacher_idx in range(self.num_teachers):
                # 每个教师独立的 MLP
                teacher_logits = self.teacher_heads[teacher_idx](pooled)  # [batch, num_buckets]
                logits_list.append(teacher_logits)

            # 堆叠为 [batch, num_teachers, num_buckets]
            logits = torch.stack(logits_list, dim=1)  # [batch, num_teachers, num_buckets]
            return logits
        else:
            # 回归模式：使用sigmoid约束到[0, 1]，然后缩放到[0, score_scale]
            logits = self.head(pooled)  # [batch_size, num_teachers]
            scores = torch.sigmoid(logits) * self.score_scale
            return scores

    def get_expected_scores(self, logits: torch.Tensor, teacher_names: Optional[List[str]] = None) -> torch.Tensor:
        """将桶 logits 转换为期望分数（仅用于分类模式）

        参数:
            logits: [batch_size, num_teachers, num_buckets]
            teacher_names: 教师名称列表（用于教师特定分桶），顺序与 num_teachers 对应

        返回:
            expected_scores: [batch_size, num_teachers] - 基于桶概率分布的期望分数
        """
        if not self.use_bucketing:
            raise ValueError("get_expected_scores 只能在分类模式下使用")

        # Softmax 获取概率分布
        probs = F.softmax(logits, dim=-1)  # [batch, teachers, buckets]

        batch_size, num_teachers, num_buckets = probs.shape
        device = probs.device

        # 如果有教师特定桶中心，逐个教师计算期望分数
        if self.teacher_bucket_centers is not None and teacher_names is not None:
            expected_scores = torch.zeros(batch_size, num_teachers, device=device, dtype=probs.dtype)

            for t, teacher_name in enumerate(teacher_names):
                if teacher_name in self.teacher_bucket_centers:
                    # 使用该教师的桶中心
                    centers = torch.tensor(
                        self.teacher_bucket_centers[teacher_name],
                        dtype=probs.dtype,
                        device=device
                    )
                    # 期望分数 = Σ(bucket_center * prob)
                    expected_scores[:, t] = (probs[:, t, :] * centers).sum(dim=-1)
                else:
                    # 如果找不到该教师的配置，使用默认方法（均匀分布中心）
                    centers = torch.linspace(0.5, 9.5, num_buckets, dtype=probs.dtype, device=device)
                    expected_scores[:, t] = (probs[:, t, :] * centers).sum(dim=-1)
        else:
            # 向后兼容：所有教师共享桶中心（使用均匀分布）
            centers = torch.linspace(0.5, 9.5, num_buckets, dtype=probs.dtype, device=device)
            expected_scores = (probs * centers.view(1, 1, -1)).sum(dim=-1)  # [batch, teachers]

        return expected_scores
