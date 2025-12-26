"""训练和评估步骤函数"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List
import numpy as np


def _build_prompt(instruction: str, mode: str = "routing", strong_teacher: str = None, weak_teacher: str = None) -> str:
    """构建路由器的输入prompt

    参数:
        instruction: 问题文本
        mode: "routing" (路由模式) 或 "score_diff" (分数差预测模式)
        strong_teacher: 强教师名称（仅score_diff模式需要）
        weak_teacher: 弱教师名称（仅score_diff模式需要）

    返回:
        prompt: 构建好的prompt
    """
    if mode == "score_diff":
        # 分数差预测模式：清晰描述任务
        return f"""You are evaluating which AI model performs better on a given problem.

Task: Predict how much better {strong_teacher} performs compared to {weak_teacher} on the following problem.

Instructions:
- Output a single number representing the score difference
- Positive number: {strong_teacher} is better (e.g., +3.5 means {strong_teacher} scores 3.5 points higher)
- Negative number: {weak_teacher} is better (e.g., -2.0 means {weak_teacher} scores 2.0 points higher)
- Zero: Both models perform equally well

Problem:
{instruction}

Score difference:"""
    else:
        # 原始路由模式
        return f"""Analyze the following problem and determine which AI model would be most suitable to answer it.

Problem:
{instruction}

Analysis:"""


def score_to_bucket(score: float, bucket_ranges: List[float]) -> int:
    """将连续分数转换为桶索引

    参数:
        score: NTE 分数 (0-1)
        bucket_ranges: 桶边界，例如 [0.2, 0.4, 0.6, 0.8]

    返回:
        bucket_idx: 桶索引 (0 到 num_buckets-1)

    示例:
        bucket_ranges = [0.2, 0.4, 0.6, 0.8]
        score=0.15 -> bucket_0: [0.0, 0.2)
        score=0.32 -> bucket_1: [0.2, 0.4)
        score=0.58 -> bucket_2: [0.4, 0.6)
        score=0.71 -> bucket_3: [0.6, 0.8)
        score=0.95 -> bucket_4: [0.8, 1.0]
    """
    for i, threshold in enumerate(bucket_ranges):
        if score < threshold:
            return i
    return len(bucket_ranges)  # 最后一个桶


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, reduction: str = 'mean') -> torch.Tensor:
    """Focal Loss: 聚焦于难分类样本，降低易分类样本的权重

    FL(p) = -(1-p)^γ * log(p)

    参数:
        logits: [batch, num_classes] 未归一化的logits
        targets: [batch] 类别标签
        gamma: 聚焦参数，越大对易分类样本惩罚越强（默认2.0）
        reduction: 'mean' | 'sum' | 'none'

    返回:
        focal_loss: 损失值
    """
    # 计算 softmax 概率
    probs = F.softmax(logits, dim=-1)  # [batch, num_classes]

    # 获取目标类别的概率
    batch_size = logits.shape[0]
    target_probs = probs[torch.arange(batch_size, device=logits.device), targets]  # [batch]

    # 计算 focal weight: (1 - p)^γ
    focal_weight = (1 - target_probs) ** gamma  # [batch]

    # 计算 cross entropy (不使用reduction，以便应用focal weight)
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [batch]

    # 应用 focal weight
    focal_loss = focal_weight * ce_loss  # [batch]

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def pairwise_ranking_loss(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    masks: torch.Tensor,
    margin: float = 0.5,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Pairwise Ranking Loss: 保持教师之间的相对排序关系

    对于每个样本，构造所有教师对 (i, j)，如果 target[i] > target[j]，
    则希望 pred[i] > pred[j] + margin

    Loss = max(0, margin - (pred[i] - pred[j]))  当 target[i] > target[j]

    参数:
        pred_scores: [batch, num_teachers] 预测分数
        target_scores: [batch, num_teachers] 真实分数
        masks: [batch, num_teachers] 有效性掩码
        margin: 排序边界，鼓励正确的排序对分数差大于margin（默认0.5）
        reduction: 'mean' | 'sum' | 'none'

    返回:
        ranking_loss: 损失值
    """
    batch_size, num_teachers = pred_scores.shape
    device = pred_scores.device

    total_loss = torch.tensor(0.0, device=device, dtype=pred_scores.dtype)
    num_pairs = 0

    for b in range(batch_size):
        # 获取当前样本的有效教师
        valid_mask = masks[b] > 0  # [num_teachers]
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) < 2:
            continue  # 至少需要2个有效教师才能构造配对

        # 构造所有教师对 (i, j)，其中 i < j
        for idx_i in range(len(valid_indices)):
            for idx_j in range(idx_i + 1, len(valid_indices)):
                i = valid_indices[idx_i]
                j = valid_indices[idx_j]

                target_i = target_scores[b, i]
                target_j = target_scores[b, j]
                pred_i = pred_scores[b, i]
                pred_j = pred_scores[b, j]

                # 如果 target[i] > target[j]，希望 pred[i] > pred[j] + margin
                if target_i > target_j:
                    loss_ij = F.relu(margin - (pred_i - pred_j))
                    total_loss += loss_ij
                    num_pairs += 1
                # 如果 target[j] > target[i]，希望 pred[j] > pred[i] + margin
                elif target_j > target_i:
                    loss_ji = F.relu(margin - (pred_j - pred_i))
                    total_loss += loss_ji
                    num_pairs += 1
                # 如果相等，不施加排序约束

    if num_pairs == 0:
        return torch.tensor(0.0, device=device, dtype=pred_scores.dtype)

    if reduction == 'mean':
        return total_loss / num_pairs
    elif reduction == 'sum':
        return total_loss
    else:
        return total_loss


def _process_batch(batch, tokenizer, required_teachers, max_seq_length, device, use_bucketing=False, teacher_bucket_ranges=None, use_score_diff=False, strong_teacher_idx=0):
    """处理batch数据：tokenization + 提取target scores/buckets

    参数:
        use_score_diff: 是否使用分数差模式
        strong_teacher_idx: 强教师索引（仅score_diff模式需要）

    返回:
        如果 use_bucketing=False (回归模式):
            {
                "input_ids": [batch_size, seq_len],
                "attention_mask": [batch_size, seq_len],
                "target_scores": [batch_size, num_teachers],
                "masks": [batch_size, num_teachers]
            }

        如果 use_bucketing=True (分类模式):
            {
                "input_ids": [batch_size, seq_len],
                "attention_mask": [batch_size, seq_len],
                "target_buckets": [batch_size, num_teachers],  # 桶索引 (long)
                "masks": [batch_size, num_teachers]
            }
    """
    # 根据模式选择prompt
    if use_score_diff:
        # 分数差模式：使用特殊的prompt
        strong_teacher = required_teachers[strong_teacher_idx]
        weak_teacher = required_teachers[1 - strong_teacher_idx]
        prompts = [
            _build_prompt(sample["instruction"], mode="score_diff",
                         strong_teacher=strong_teacher, weak_teacher=weak_teacher)
            for sample in batch
        ]
    else:
        # 回归/分类模式：使用原始prompt
        prompts = [_build_prompt(sample["instruction"]) for sample in batch]

    # Tokenize
    encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )

    # 提取目标 NTE 分数或桶
    target_scores = []
    target_buckets = []
    masks = []

    for sample in batch:
        scores = []
        buckets = []
        mask = []

        for teacher_idx, teacher in enumerate(required_teachers):
            # 检查教师响应是否存在，且包含 nte_scores
            if (teacher in sample.get("responses", {}) and
                "nte_scores" in sample["responses"][teacher]):
                nte_score = sample["responses"][teacher]["nte_scores"]["nte_score"]
                scores.append(nte_score)

                # 如果使用桶化，转换分数为桶索引
                if use_bucketing:
                    # 使用教师特定的桶边界（如果有）
                    if teacher_bucket_ranges is not None and teacher in teacher_bucket_ranges:
                        ranges = teacher_bucket_ranges[teacher]
                    else:
                        # 向后兼容：使用默认边界
                        ranges = [2.0, 4.0, 6.0, 8.0]

                    bucket_idx = score_to_bucket(nte_score, ranges)
                    buckets.append(bucket_idx)

                mask.append(1.0)
            else:
                # 没有分数，用0占位，mask标记为0
                scores.append(0.0)
                if use_bucketing:
                    buckets.append(0)  # 占位
                mask.append(0.0)

        target_scores.append(scores)
        if use_bucketing:
            target_buckets.append(buckets)
        masks.append(mask)

    # 不对input_ids和attention_mask指定设备，让模型的device_map自动处理
    # 只将target和mask移到指定设备（用于loss计算）
    result = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "masks": torch.tensor(masks, dtype=torch.float32, device=device)
    }

    if use_bucketing:
        result["target_buckets"] = torch.tensor(target_buckets, dtype=torch.long, device=device)
        result["target_scores"] = torch.tensor(target_scores, dtype=torch.float32, device=device)  # 桶化模式也返回分数（用于计算Top-1准确率）
    else:
        result["target_scores"] = torch.tensor(target_scores, dtype=torch.float32, device=device)

    return result


def train_step(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """训练步骤

    参数:
        batch: 批次数据（原始数据，list of samples）
        device: 设备（由trainer传入）
        info: 训练信息，包含:
            - models: 注册的模型字典（包含router, tokenizer等）
            - config: 配置对象
            - ...

    返回:
        回归模式:
            losses: {
                "router": {
                    "loss": 总损失,
                    "mse": MSE损失,
                    "mae": MAE (用于监控)
                }
            }

        分类模式:
            losses: {
                "router": {
                    "loss": 总损失 (CrossEntropy),
                    "ce": CrossEntropy损失 (用于监控),
                    "acc": 桶分类准确率 (用于监控)
                }
            }
    """
    # 获取模型和配置
    router = info["models"]["router"]
    tokenizer = info["models"]["tokenizer"]
    required_teachers = info["models"]["required_teachers"]
    max_seq_length = info["models"]["max_seq_length"]
    config = info["config"]

    # 获取桶化配置
    use_bucketing = config["method_settings"].get("use_bucketing", False)
    use_score_diff = config["method_settings"].get("use_score_diff", False)
    teacher_bucket_ranges = config["method_settings"].get("teacher_bucket_ranges", None)
    strong_teacher_idx = config["method_settings"].get("strong_teacher_idx", 0)

    # 处理batch：tokenization + 提取target
    processed = _process_batch(
        batch, tokenizer, required_teachers, max_seq_length, device,
        use_bucketing=use_bucketing,
        teacher_bucket_ranges=teacher_bucket_ranges,
        use_score_diff=use_score_diff,
        strong_teacher_idx=strong_teacher_idx
    )

    input_ids = processed["input_ids"]
    attention_mask = processed["attention_mask"]
    masks = processed["masks"]

    # 获取模型的dtype（用于target）
    model_dtype = next(router.parameters()).dtype
    masks = masks.to(dtype=model_dtype)

    # input_ids和attention_mask保持在CPU，让LLM的device_map自动处理
    # 前向传播（device_map会自动处理输入的设备分配）
    outputs = router(input_ids, attention_mask)

    if use_bucketing:
        # ============== 分类模式 ==============
        logits = outputs  # [batch, num_teachers, num_buckets]
        target_buckets = processed["target_buckets"]  # [batch, num_teachers]

        # 将预测结果移到target所在的设备
        logits = logits.to(target_buckets.device)

        batch_size, num_teachers, num_buckets = logits.shape

        # 获取 Focal Loss 配置
        focal_gamma = config["method_settings"].get("focal_gamma", 2.0)

        # 为每个教师独立计算 Focal Loss
        focal_total_loss = 0
        valid_count = 0

        for t in range(num_teachers):
            teacher_mask = masks[:, t]  # [batch]
            valid_samples = teacher_mask > 0

            if valid_samples.sum() > 0:
                teacher_logits = logits[valid_samples, t, :]  # [valid_batch, num_buckets]
                teacher_targets = target_buckets[valid_samples, t]  # [valid_batch]

                # 使用 Focal Loss 替代 CrossEntropy
                focal_total_loss += focal_loss(teacher_logits, teacher_targets, gamma=focal_gamma, reduction='mean')
                valid_count += 1

        focal_total_loss = focal_total_loss / (valid_count + 1e-8)

        # 计算Top-1准确率（桶化模式）
        with torch.no_grad():
            # 将logits转换为期望分数
            expected_scores = router.get_expected_scores(logits, teacher_names=required_teachers)  # [batch, num_teachers]

            # 找到预测的最佳教师和真实的最佳教师
            pred_best = torch.argmax(expected_scores, dim=1)  # [batch]
            true_best = torch.argmax(target_scores, dim=1)  # [batch]

            # 计算准确率（只考虑所有教师都有效的样本）
            all_valid = masks.sum(dim=1) == num_teachers  # [batch]
            if all_valid.sum() > 0:
                top1_acc = (pred_best[all_valid] == true_best[all_valid]).float().mean().item()
            else:
                top1_acc = 0.0

        # 返回损失字典和指标
        return {
            "router": {
                "loss": focal_total_loss,
                "focal_loss": focal_total_loss.item()
            },
            "metrics": {
                "top1_acc": top1_acc
            }
        }
    else:
        # ============== 回归模式 / 分数差预测模式 ==============
        use_score_diff = config["method_settings"].get("use_score_diff", False)

        if use_score_diff:
            # 分数差模式：直接监督 diff，而不是重构后的分数
            pred_scores = outputs  # [batch, 2]，已经是重构后的分数
            target_scores = processed["target_scores"]  # [batch, 2]

            # 转换target为模型的dtype
            target_scores = target_scores.to(dtype=model_dtype)
            pred_scores = pred_scores.to(target_scores.device)

            # 获取 strong_teacher_idx
            strong_teacher_idx = config["method_settings"].get("strong_teacher_idx", 0)

            # 计算真实的 diff 和预测的 diff
            true_diff = target_scores[:, strong_teacher_idx] - target_scores[:, 1 - strong_teacher_idx]  # [batch]
            pred_diff = pred_scores[:, strong_teacher_idx] - pred_scores[:, 1 - strong_teacher_idx]  # [batch]

            # 直接监督 diff（MSE Loss）
            diff_loss = F.mse_loss(pred_diff, true_diff)

            # 计算Top-1准确率
            with torch.no_grad():
                pred_best = torch.argmax(pred_scores, dim=1)  # [batch]
                true_best = torch.argmax(target_scores, dim=1)  # [batch]

                num_teachers = masks.shape[1]
                all_valid = masks.sum(dim=1) == num_teachers  # [batch]
                if all_valid.sum() > 0:
                    top1_acc = (pred_best[all_valid] == true_best[all_valid]).float().mean().item()
                else:
                    top1_acc = 0.0

            return {
                "router": {
                    "loss": diff_loss,
                    "diff_loss": diff_loss.item()
                },
                "metrics": {
                    "top1_acc": top1_acc
                }
            }
        else:
            # 回归模式：监督完整分数
            pred_scores = outputs  # [batch, num_teachers]
            target_scores = processed["target_scores"]  # [batch, num_teachers]

            # 转换target为模型的dtype
            target_scores = target_scores.to(dtype=model_dtype)

            # 将预测结果移到target所在的设备
            pred_scores = pred_scores.to(target_scores.device)

            # 计算损失 (只对有效的教师计算)
            mse_loss = F.mse_loss(pred_scores, target_scores, reduction='none')  # [batch, num_teachers]
            mse_loss = (mse_loss * masks).sum() / (masks.sum() + 1e-8)  # 平均到有效的教师上

            # 计算Top-1准确率
            with torch.no_grad():
                # 找到预测的最佳教师和真实的最佳教师
                pred_best = torch.argmax(pred_scores, dim=1)  # [batch]
                true_best = torch.argmax(target_scores, dim=1)  # [batch]

                # 计算准确率（只考虑所有教师都有效的样本）
                num_teachers = masks.shape[1]
                all_valid = masks.sum(dim=1) == num_teachers  # [batch]
                if all_valid.sum() > 0:
                    top1_acc = (pred_best[all_valid] == true_best[all_valid]).float().mean().item()
                else:
                    top1_acc = 0.0

            # 返回损失字典和指标
            return {
                "router": {
                    "loss": mse_loss,
                    "mse_loss": mse_loss.item()
                },
                "metrics": {
                    "top1_acc": top1_acc
                }
            }


def eval_step(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """评估步骤

    参数:
        batch: 批次数据（原始数据，list of samples）
        device: 设备（由trainer传入）
        info: 评估信息

    返回:
        回归模式:
            metrics: {
                "mse": MSE损失,
                "mae": MAE,
                "top1_acc": Top-1准确率,
                "pearson": Pearson相关系数,
            }

        分数差模式:
            metrics: {
                "top1_acc": Top-1准确率（最重要）,
                "diff_mae": 差值的MAE,
                "diff_mse": 差值的MSE,
            }

        分类模式:
            metrics: {
                "bucket_acc": 桶分类准确率,
                "top1_acc": Top-1准确率（基于期望分数）,
                "pearson": Pearson相关系数（期望分数 vs 真实分数）,
            }
    """
    # 获取模型和配置
    router = info["models"]["router"]
    tokenizer = info["models"]["tokenizer"]
    required_teachers = info["models"]["required_teachers"]
    max_seq_length = info["models"]["max_seq_length"]
    config = info["config"]

    # 获取桶化配置
    use_bucketing = config["method_settings"].get("use_bucketing", False)
    use_score_diff = config["method_settings"].get("use_score_diff", False)
    teacher_bucket_ranges = config["method_settings"].get("teacher_bucket_ranges", None)
    strong_teacher_idx = config["method_settings"].get("strong_teacher_idx", 0)

    # 处理batch：tokenization + 提取target
    processed = _process_batch(
        batch, tokenizer, required_teachers, max_seq_length, device,
        use_bucketing=use_bucketing,
        teacher_bucket_ranges=teacher_bucket_ranges,
        use_score_diff=use_score_diff,
        strong_teacher_idx=strong_teacher_idx
    )

    input_ids = processed["input_ids"]
    attention_mask = processed["attention_mask"]
    masks = processed["masks"]

    # 获取模型的dtype
    model_dtype = next(router.parameters()).dtype
    masks = masks.to(dtype=model_dtype)

    # 前向传播（device_map会自动处理输入的设备分配）
    with torch.no_grad():
        outputs = router(input_ids, attention_mask)

    if use_bucketing:
        # ============== 分类模式 ==============
        logits = outputs  # [batch, num_teachers, num_buckets]
        target_buckets = processed["target_buckets"]  # [batch, num_teachers]

        # 将预测结果移到target所在的设备
        logits = logits.to(target_buckets.device)

        batch_size, num_teachers, num_buckets = logits.shape

        # 1. 计算桶分类准确率
        pred_buckets = logits.argmax(dim=-1)  # [batch, num_teachers]
        bucket_correct = (pred_buckets == target_buckets).float()
        bucket_acc = (bucket_correct * masks).sum() / (masks.sum() + 1e-8)

        # 2. 统计每个教师的预测桶分布和真实桶分布（只返回counts，让trainer累积）
        teacher_bucket_distributions = {}
        for t, teacher_name in enumerate(required_teachers):
            teacher_mask = masks[:, t] > 0  # [batch]
            if teacher_mask.sum() > 0:
                # 获取该教师的预测桶和真实桶
                teacher_pred_buckets = pred_buckets[teacher_mask, t]  # [valid_samples]
                teacher_true_buckets = target_buckets[teacher_mask, t]  # [valid_samples]

                # 统计预测桶的数量
                pred_bucket_counts = torch.zeros(num_buckets, device=device)
                for b in range(num_buckets):
                    pred_bucket_counts[b] = (teacher_pred_buckets == b).sum().item()

                # 统计真实桶的数量
                true_bucket_counts = torch.zeros(num_buckets, device=device)
                for b in range(num_buckets):
                    true_bucket_counts[b] = (teacher_true_buckets == b).sum().item()

                # 只返回counts（整数），让trainer累积后再计算百分比
                teacher_bucket_distributions[teacher_name] = {
                    "predicted_counts": pred_bucket_counts.cpu().numpy().tolist(),
                    "ground_truth_counts": true_bucket_counts.cpu().numpy().tolist()
                }

        # 3. 计算期望分数（用于 Top-1 和 Pearson）
        expected_scores = router.get_expected_scores(logits, teacher_names=required_teachers)  # [batch, num_teachers]

        # 需要真实分数来计算 Pearson
        # 从 target_scores 获取（在处理时也计算了）
        target_scores_list = []
        for sample in batch:
            scores = []
            for teacher in required_teachers:
                if (teacher in sample.get("responses", {}) and
                    "nte_scores" in sample["responses"][teacher]):
                    nte_score = sample["responses"][teacher]["nte_scores"]["nte_score"]
                    scores.append(nte_score)
                else:
                    scores.append(0.0)
            target_scores_list.append(scores)

        target_scores = torch.tensor(target_scores_list, dtype=model_dtype, device=device)

        # 4. 计算 Top-1 准确率（基于期望分数）
        pred_best_teacher = expected_scores.argmax(dim=1)  # [batch]
        true_best_teacher = target_scores.argmax(dim=1)  # [batch]
        top1_correct = (pred_best_teacher == true_best_teacher).float().mean()

        # 5. 计算 Pearson 相关系数（期望分数 vs 真实分数）
        valid_mask = masks.view(-1) > 0
        pred_flat = expected_scores.view(-1)[valid_mask]
        target_flat = target_scores.view(-1)[valid_mask]

        if len(pred_flat) > 1:
            pred_mean = pred_flat.mean()
            target_mean = target_flat.mean()
            pred_centered = pred_flat - pred_mean
            target_centered = target_flat - target_mean

            numerator = (pred_centered * target_centered).sum()
            denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

            if denominator > 1e-8:
                pearson = numerator / denominator
            else:
                pearson = torch.tensor(0.0, device=device)
        else:
            pearson = torch.tensor(0.0, device=device)

        return {
            "bucket_acc": bucket_acc.item(),  # 桶分类准确率
            "top1_acc": top1_correct.item(),   # Top-1 准确率（最重要！）
            "pearson": pearson.item(),         # 相关系数
            "teacher_bucket_dist": teacher_bucket_distributions,  # 每个教师的桶分布
        }
    else:
        # ============== 回归模式 / 分数差预测模式 ==============
        # 分数差模式在forward中已经重构为完整分数，因此评估指标与回归模式相同
        pred_scores = outputs  # [batch, num_teachers]
        target_scores = processed["target_scores"]  # [batch, num_teachers]

        # 转换target为模型的dtype
        target_scores = target_scores.to(dtype=model_dtype)

        # 将预测结果移到target所在的设备
        pred_scores = pred_scores.to(target_scores.device)

        # 计算更实际的指标：Top-1 准确率
        # 对于每个样本，比较预测的最佳教师和真实的最佳教师是否一致
        pred_best_teacher = pred_scores.argmax(dim=1)  # [batch]
        true_best_teacher = target_scores.argmax(dim=1)  # [batch]
        top1_correct = (pred_best_teacher == true_best_teacher).float().mean()

        if use_score_diff:
            # ============== 分数差模式：只关注diff相关指标 ==============
            # 获取strong_teacher_idx
            strong_teacher_idx = config["method_settings"].get("strong_teacher_idx", 0)
            weak_teacher_idx = 1 - strong_teacher_idx

            # 计算真实的diff和预测的diff
            true_diff = target_scores[:, strong_teacher_idx] - target_scores[:, weak_teacher_idx]  # [batch]
            pred_diff = pred_scores[:, strong_teacher_idx] - pred_scores[:, weak_teacher_idx]  # [batch]

            # 计算diff的MAE
            diff_mae = torch.abs(pred_diff - true_diff).mean()

            # 计算diff的MSE
            diff_mse = F.mse_loss(pred_diff, true_diff)

            return {
                "top1_acc": top1_correct.item(),  # Top-1 准确率：最重要的指标
                "diff_mae": diff_mae.item(),       # 差值的MAE
                "diff_mse": diff_mse.item(),       # 差值的MSE
            }
        else:
            # ============== 回归模式：返回完整的指标 ==============
            # 计算整体 MSE 和 MAE
            mse_loss = F.mse_loss(pred_scores, target_scores, reduction='none')  # [batch, num_teachers]
            mse = (mse_loss * masks).sum() / (masks.sum() + 1e-8)

            mae = torch.abs(pred_scores - target_scores)
            mae = (mae * masks).sum() / (masks.sum() + 1e-8)

            # 计算 Pearson 相关系数（整体）
            # 将所有有效的 pred 和 target 展平
            valid_mask = masks.view(-1) > 0
            pred_flat = pred_scores.view(-1)[valid_mask]
            target_flat = target_scores.view(-1)[valid_mask]

            if len(pred_flat) > 1:
                # Pearson correlation
                pred_mean = pred_flat.mean()
                target_mean = target_flat.mean()
                pred_centered = pred_flat - pred_mean
                target_centered = target_flat - target_mean

                numerator = (pred_centered * target_centered).sum()
                denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

                if denominator > 1e-8:
                    pearson = numerator / denominator
                else:
                    pearson = torch.tensor(0.0, device=device)
            else:
                pearson = torch.tensor(0.0, device=device)

            return {
                "mse": mse.item(),
                "mae": mae.item(),
                "top1_acc": top1_correct.item(),  # Top-1 准确率：最重要的指标
                "pearson": pearson.item(),  # 相关系数：越接近 1 越好
            }
