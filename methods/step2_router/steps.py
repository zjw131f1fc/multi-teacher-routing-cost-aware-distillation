"""训练和评估步骤函数（桶化分类模式）"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List
import numpy as np


def _build_prompt(instruction: str) -> str:
    """构建路由器的输入prompt

    参数:
        instruction: 问题文本

    返回:
        prompt: 构建好的prompt
    """
    return f"""Analyze the following problem and determine which AI model would be most suitable to answer it.

Problem:
{instruction}

Analysis:"""


def _score_to_bucket(score: float, bucket_ranges: List[float]) -> int:
    """将连续分数映射到桶索引

    参数:
        score: NTE 分数 [0, 1]
        bucket_ranges: 桶边界列表，例如 [0.2, 0.4, 0.6, 0.8] 表示 5 个桶

    返回:
        bucket_index: 桶索引 [0, num_buckets-1]
    """
    for i, threshold in enumerate(bucket_ranges):
        if score < threshold:
            return i
    return len(bucket_ranges)  # 最后一个桶


def _process_batch(batch, required_teachers, teacher_bucket_ranges, device):
    """处理batch数据：构建prompts + 提取target bucket indices

    参数:
        batch: list of samples from dataset
        required_teachers: list of teacher names
        teacher_bucket_ranges: dict {teacher_name: [bucket_ranges]}，每个教师的桶边界
        device: 设备

    返回:
        {
            "prompts": [batch_size] 文本列表,
            "target_buckets": {teacher_name: [batch_size]} 目标桶索引,
            "masks": {teacher_name: [batch_size]} 有效性掩码
        }
    """
    # 构建prompts
    prompts = [_build_prompt(sample["instruction"]) for sample in batch]

    # 提取目标桶索引
    target_buckets = {teacher: [] for teacher in required_teachers}
    masks = {teacher: [] for teacher in required_teachers}

    for sample in batch:
        for teacher in required_teachers:
            # 检查教师响应是否存在，且包含 nte_scores
            if (teacher in sample.get("responses", {}) and
                "nte_scores" in sample["responses"][teacher]):
                nte_score = sample["responses"][teacher]["nte_scores"]["nte_score"]

                # 将分数映射到桶
                bucket_idx = _score_to_bucket(nte_score, teacher_bucket_ranges[teacher])
                target_buckets[teacher].append(bucket_idx)
                masks[teacher].append(1.0)
            else:
                # 没有分数，用0占位，mask标记为0
                target_buckets[teacher].append(0)
                masks[teacher].append(0.0)

    # 转换为 tensor
    target_buckets = {
        teacher: torch.tensor(indices, dtype=torch.long, device=device)
        for teacher, indices in target_buckets.items()
    }
    masks = {
        teacher: torch.tensor(mask, dtype=torch.float32, device=device)
        for teacher, mask in masks.items()
    }

    return {
        "prompts": prompts,
        "target_buckets": target_buckets,
        "masks": masks
    }


def train_step(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """训练步骤（桶化分类模式）

    参数:
        batch: 批次数据（原始数据，list of samples）
        device: 设备（由trainer传入）
        info: 训练信息，包含:
            - models: 注册的模型字典（包含router, required_teachers, teacher_bucket_ranges）
            - config: 配置对象

    返回:
        losses: {
            "router": {
                "loss": 总损失（加权交叉熵）,
                "ce_loss": 交叉熵损失,
            },
            "metrics": {
                "top1_acc": Top-1准确率（桶预测准确率）
            }
        }
    """
    # 获取模型和配置
    router = info["models"]["router"]
    required_teachers = info["models"]["required_teachers"]
    teacher_bucket_ranges = info["models"]["teacher_bucket_ranges"]

    # 处理batch：构建prompts + 提取target bucket indices
    processed = _process_batch(batch, required_teachers, teacher_bucket_ranges, device)

    prompts = processed["prompts"]
    target_buckets = processed["target_buckets"]
    masks = processed["masks"]

    # 前向传播
    logits_dict = router(prompts)  # {teacher_name: [batch, num_buckets]}

    # 计算损失：对每个教师独立计算交叉熵，然后平均
    total_loss = 0.0
    valid_teachers = 0

    for teacher in required_teachers:
        logits = logits_dict[teacher]  # [batch, num_buckets]
        targets = target_buckets[teacher]  # [batch]
        mask = masks[teacher]  # [batch]

        # 计算交叉熵损失（reduction='none' 以便应用 mask）
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [batch]

        # 应用 mask：只计算有效样本的损失
        masked_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)
        total_loss = total_loss + masked_loss

        if mask.sum() > 0:
            valid_teachers += 1

    # 平均到有效的教师上
    if valid_teachers > 0:
        total_loss = total_loss / valid_teachers

    # 计算桶预测准确率
    with torch.no_grad():
        correct_count = 0
        total_count = 0

        for teacher in required_teachers:
            logits = logits_dict[teacher]
            targets = target_buckets[teacher]
            mask = masks[teacher]

            # 预测的桶
            pred_buckets = torch.argmax(logits, dim=1)  # [batch]

            # 计算准确率（只考虑有效样本）
            valid_mask = mask > 0
            if valid_mask.sum() > 0:
                correct = (pred_buckets[valid_mask] == targets[valid_mask]).sum().item()
                total = valid_mask.sum().item()
                correct_count += correct
                total_count += total

        bucket_acc = correct_count / total_count if total_count > 0 else 0.0

    # 返回损失字典和指标
    return {
        "router": {
            "loss": total_loss,
            "ce_loss": total_loss.item()
        },
        "metrics": {
            "bucket_acc": bucket_acc  # 桶预测准确率
        }
    }


def eval_step(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """评估步骤（桶化分类模式）

    参数:
        batch: 批次数据（原始数据，list of samples）
        device: 设备（由trainer传入）
        info: 评估信息

    返回:
        metrics: {
            "ce_loss": 交叉熵损失,
            "bucket_acc": 桶预测准确率（预测桶 == 真实桶）,
            "teacher_acc": 教师选择准确率（最重要！预测最佳教师 == 真实最佳教师）
        }
    """
    # 获取模型和配置
    router = info["models"]["router"]
    required_teachers = info["models"]["required_teachers"]
    teacher_bucket_ranges = info["models"]["teacher_bucket_ranges"]
    teacher_bucket_centers = info["models"]["teacher_bucket_centers"]

    # 处理batch：构建prompts + 提取target
    processed = _process_batch(batch, required_teachers, teacher_bucket_ranges, device)

    prompts = processed["prompts"]
    target_buckets = processed["target_buckets"]
    masks = processed["masks"]

    # 前向传播
    with torch.no_grad():
        logits_dict = router(prompts)  # {teacher_name: [batch, num_buckets]}

    # 计算交叉熵损失
    total_loss = 0.0
    valid_teachers = 0

    for teacher in required_teachers:
        logits = logits_dict[teacher]
        targets = target_buckets[teacher]
        mask = masks[teacher]

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        masked_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)
        total_loss = total_loss + masked_loss

        if mask.sum() > 0:
            valid_teachers += 1

    ce_loss_value = (total_loss / valid_teachers).item() if valid_teachers > 0 else 0.0

    # 计算桶预测准确率
    bucket_correct = 0
    bucket_total = 0

    for teacher in required_teachers:
        logits = logits_dict[teacher]
        targets = target_buckets[teacher]
        mask = masks[teacher]

        pred_buckets = torch.argmax(logits, dim=1)
        valid_mask = mask > 0

        if valid_mask.sum() > 0:
            bucket_correct += (pred_buckets[valid_mask] == targets[valid_mask]).sum().item()
            bucket_total += valid_mask.sum().item()

    bucket_acc = bucket_correct / bucket_total if bucket_total > 0 else 0.0

    # 计算教师选择准确率（基于桶中心值）⭐ 最重要的指标
    # 将桶索引转换为对应的桶中心分数，然后选择最佳教师
    batch_size = len(prompts)
    teacher_correct = 0
    teacher_total = 0

    for i in range(batch_size):
        # 检查所有教师是否都有有效分数
        all_valid = all(masks[teacher][i] > 0 for teacher in required_teachers)
        if not all_valid:
            continue

        # 预测的桶中心分数
        pred_scores = {}
        for teacher in required_teachers:
            pred_bucket = torch.argmax(logits_dict[teacher][i]).item()
            pred_scores[teacher] = teacher_bucket_centers[teacher][pred_bucket]

        # 真实的桶中心分数
        true_scores = {}
        for teacher in required_teachers:
            true_bucket = target_buckets[teacher][i].item()
            true_scores[teacher] = teacher_bucket_centers[teacher][true_bucket]

        # 选择最佳教师
        pred_best = max(pred_scores, key=pred_scores.get)
        true_best = max(true_scores, key=true_scores.get)

        if pred_best == true_best:
            teacher_correct += 1
        teacher_total += 1

    teacher_acc = teacher_correct / teacher_total if teacher_total > 0 else 0.0

    return {
        "ce_loss": ce_loss_value,
        "bucket_acc": bucket_acc,      # 桶预测准确率
        "teacher_acc": teacher_acc,    # 教师选择准确率 ⭐ 最重要！
    }
