"""训练和评估步骤函数（支持桶化分类模式和回归模式）"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List
import numpy as np
import random


class ExpandedDatasetWrapper:
    """数据集包装器：将原始样本展开为 [问题+教师] 对

    原始数据格式:
    {
        "instruction": str,
        "responses": {
            "teacher1": {"nte_scores": {"nte_score": float}},
            "teacher2": {"nte_scores": {"nte_score": float}}
        }
    }

    展开后格式:
    {
        "instruction": str,
        "teacher_name": str,
        "nte_score": float
    }

    每次训练时，每个样本随机选择一个教师，实现 "分batch：每个batch随机采样 [问题+随机教师]"
    """

    def __init__(self, original_dataset, required_teachers, seed=42):
        """
        参数:
            original_dataset: 原始数据集（未展开）
            required_teachers: 需要的教师列表
            seed: 随机种子
        """
        self.original_dataset = original_dataset
        self.required_teachers = required_teachers
        self.seed = seed
        self.rng = random.Random(seed)

        # 预先过滤：只保留包含所有教师NTE分数的样本
        self.valid_samples = []
        for sample in original_dataset:
            responses = sample.get("responses", {})
            has_all_scores = all(
                teacher in responses and "nte_scores" in responses[teacher]
                for teacher in required_teachers
            )
            if has_all_scores:
                self.valid_samples.append(sample)

    def __len__(self):
        # 长度等于有效样本数（每个样本每个epoch只采样一次，但随机选择教师）
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """获取展开后的样本（随机选择一个教师）"""
        sample = self.valid_samples[idx]
        instruction = sample["instruction"]
        responses = sample["responses"]

        # 随机选择一个教师（每次调用都重新随机，实现随机采样）
        teacher_name = self.rng.choice(self.required_teachers)

        # 提取该教师的NTE分数
        nte_score = responses[teacher_name]["nte_scores"]["nte_score"]

        return {
            "instruction": instruction,
            "teacher_name": teacher_name,
            "nte_score": nte_score
        }


def _build_prompt(instruction: str, teacher_name: str) -> str:
    """构建路由器的输入prompt（包含教师名称）

    参数:
        instruction: 问题文本
        teacher_name: 教师模型名称

    返回:
        prompt: 构建好的prompt
    """
    return f"""Question: {instruction}

Teacher: {teacher_name}"""


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


def _process_batch_regression(batch, required_teachers, device):
    """处理batch数据（回归模式）：每个样本展开为 [问题+教师] 对

    参数:
        batch: list of samples from dataset，每个样本已经包含 teacher_name 字段
        required_teachers: list of teacher names（用于验证）
        device: 设备

    返回:
        {
            "prompts": [batch_size] 文本列表（包含教师名称），
            "target_scores": [batch_size] 目标NTE分数，
            "teacher_names": [batch_size] 对应的教师名称列表
        }
    """
    prompts = []
    target_scores = []
    teacher_names_out = []

    for sample in batch:
        instruction = sample["instruction"]
        teacher_name = sample["teacher_name"]  # 从样本中获取教师名称
        nte_score = sample["nte_score"]  # 从样本中获取NTE分数

        # 构建prompt（包含教师名称）
        prompt = _build_prompt(instruction, teacher_name)
        prompts.append(prompt)
        target_scores.append(nte_score)
        teacher_names_out.append(teacher_name)

    # 转换为 tensor
    target_scores_tensor = torch.tensor(target_scores, dtype=torch.float32, device=device)

    return {
        "prompts": prompts,
        "target_scores": target_scores_tensor,  # [batch_size]
        "teacher_names": teacher_names_out
    }


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


def train_step_regression(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """训练步骤（回归模式 - 单教师NTE分数预测）

    参数:
        batch: 批次数据，每个样本格式为:
            {
                "instruction": str,
                "teacher_name": str,  # 单个教师名称
                "nte_score": float    # 该教师的NTE分数
            }
        device: 设备
        info: 训练信息，包含:
            - models: 注册的模型字典（包含router, required_teachers）
            - config: 配置对象

    返回:
        losses: {
            "router": {
                "loss": MSE损失,
                "mse_loss": MSE损失值
            },
            "metrics": {
                "mae": 平均绝对误差
            }
        }
    """
    # 获取模型和配置
    router = info["models"]["router"]
    required_teachers = info["models"]["required_teachers"]

    # 处理batch
    processed = _process_batch_regression(batch, required_teachers, device)

    prompts = processed["prompts"]
    target_scores = processed["target_scores"]  # [batch_size]

    # 获取模型的dtype
    model_dtype = next(router.parameters()).dtype

    # 前向传播
    pred_scores = router(prompts)  # [batch_size]

    # 转换target为模型的dtype
    target_scores = target_scores.to(dtype=model_dtype)
    pred_scores = pred_scores.to(target_scores.device)

    # 计算MSE损失
    mse_loss = F.mse_loss(pred_scores, target_scores)

    # 计算MAE（用于监控）
    with torch.no_grad():
        mae = F.l1_loss(pred_scores, target_scores).item()

    return {
        "router": {
            "loss": mse_loss,
            "mse_loss": mse_loss.item()
        },
        "metrics": {
            "mae": mae
        }
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

    # 计算桶预测准确率（分教师统计）
    with torch.no_grad():
        per_teacher_acc = {}
        total_correct = 0
        total_count = 0

        for teacher in required_teachers:
            logits = logits_dict[teacher]
            targets = target_buckets[teacher]
            mask = masks[teacher]

            # 预测的桶
            pred_buckets = torch.argmax(logits, dim=1)  # [batch]

            # 计算该教师的准确率
            valid_mask = mask > 0
            if valid_mask.sum() > 0:
                correct = (pred_buckets[valid_mask] == targets[valid_mask]).sum().item()
                total = valid_mask.sum().item()
                per_teacher_acc[teacher] = correct / total

                total_correct += correct
                total_count += total
            else:
                per_teacher_acc[teacher] = 0.0

        # 总体准确率
        bucket_acc = total_correct / total_count if total_count > 0 else 0.0

    # 返回损失字典和指标
    metrics = {
        "bucket_acc": bucket_acc  # 总体桶预测准确率
    }

    # 添加每个教师的准确率
    for teacher in required_teachers:
        # 将教师名称中的特殊字符替换为下划线（用于指标名称）
        teacher_key = teacher.replace(".", "_").replace("-", "_")
        metrics[f"bucket_acc_{teacher_key}"] = per_teacher_acc[teacher]

    return {
        "router": {
            "loss": total_loss,
            "ce_loss": total_loss.item()
        },
        "metrics": metrics
    }


def eval_step_regression(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """评估步骤（回归模式 - 批量推理获取教师路由准确率）

    参数:
        batch: 批次数据，原始格式（每个样本包含所有教师的响应）
        device: 设备
        info: 评估信息

    返回:
        metrics: {
            "mse_loss": MSE损失,
            "mae": 平均绝对误差,
            "teacher_acc": 教师选择准确率 ⭐ 最重要！
        }
    """
    # 获取模型和配置
    router = info["models"]["router"]
    required_teachers = info["models"]["required_teachers"]

    # 获取模型的dtype
    model_dtype = next(router.parameters()).dtype

    # 批量推理：为每个样本x每个教师构建prompt
    all_prompts = []
    all_targets = []
    sample_indices = []  # 记录每个prompt对应的样本索引
    teacher_indices = []  # 记录每个prompt对应的教师索引

    for sample_idx, sample in enumerate(batch):
        instruction = sample["instruction"]
        responses = sample.get("responses", {})

        for teacher_idx, teacher_name in enumerate(required_teachers):
            # 检查该教师是否有NTE分数
            if teacher_name in responses and "nte_scores" in responses[teacher_name]:
                nte_score = responses[teacher_name]["nte_scores"]["nte_score"]

                # 构建prompt
                prompt = _build_prompt(instruction, teacher_name)
                all_prompts.append(prompt)
                all_targets.append(nte_score)
                sample_indices.append(sample_idx)
                teacher_indices.append(teacher_idx)

    if len(all_prompts) == 0:
        return {
            "mse_loss": 0.0,
            "mae": 0.0,
            "teacher_acc": 0.0
        }

    # 转换为 tensor
    target_scores = torch.tensor(all_targets, dtype=model_dtype, device=device)

    # 前向传播（批量）
    with torch.no_grad():
        pred_scores = router(all_prompts)  # [num_prompts]

    # 计算MSE和MAE
    mse_loss = F.mse_loss(pred_scores, target_scores).item()
    mae = F.l1_loss(pred_scores, target_scores).item()

    # 计算教师选择准确率
    # 重组预测和真实分数：按样本分组
    num_samples = len(batch)
    sample_pred_scores = {}  # {sample_idx: {teacher_idx: score}}
    sample_true_scores = {}  # {sample_idx: {teacher_idx: score}}

    for i, (sample_idx, teacher_idx) in enumerate(zip(sample_indices, teacher_indices)):
        if sample_idx not in sample_pred_scores:
            sample_pred_scores[sample_idx] = {}
            sample_true_scores[sample_idx] = {}

        sample_pred_scores[sample_idx][teacher_idx] = pred_scores[i].item()
        sample_true_scores[sample_idx][teacher_idx] = target_scores[i].item()

    # 统计教师选择准确率
    teacher_correct = 0
    teacher_total = 0

    for sample_idx in range(num_samples):
        if sample_idx not in sample_pred_scores:
            continue

        pred_dict = sample_pred_scores[sample_idx]
        true_dict = sample_true_scores[sample_idx]

        # 只统计所有教师都有效的样本
        if len(pred_dict) == len(required_teachers):
            # 选择预测分数最高的教师
            pred_best_idx = max(pred_dict, key=pred_dict.get)
            true_best_idx = max(true_dict, key=true_dict.get)

            if pred_best_idx == true_best_idx:
                teacher_correct += 1
            teacher_total += 1

    teacher_acc = teacher_correct / teacher_total if teacher_total > 0 else 0.0

    return {
        "mse_loss": mse_loss,
        "mae": mae,
        "teacher_acc": teacher_acc
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
            "bucket_acc": 总体桶预测准确率,
            "bucket_acc_{teacher}": 各教师的桶预测准确率,
            "teacher_acc": 教师选择准确率 ⭐ 最重要！
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

    # 计算桶预测准确率（分教师统计）
    per_teacher_bucket_acc = {}
    total_bucket_correct = 0
    total_bucket_count = 0

    for teacher in required_teachers:
        logits = logits_dict[teacher]
        targets = target_buckets[teacher]
        mask = masks[teacher]

        pred_buckets = torch.argmax(logits, dim=1)
        valid_mask = mask > 0

        if valid_mask.sum() > 0:
            correct = (pred_buckets[valid_mask] == targets[valid_mask]).sum().item()
            total = valid_mask.sum().item()
            per_teacher_bucket_acc[teacher] = correct / total

            total_bucket_correct += correct
            total_bucket_count += total
        else:
            per_teacher_bucket_acc[teacher] = 0.0

    # 总体桶预测准确率
    bucket_acc = total_bucket_correct / total_bucket_count if total_bucket_count > 0 else 0.0

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

    # 组装指标
    metrics = {
        "ce_loss": ce_loss_value,
        "bucket_acc": bucket_acc,      # 总体桶预测准确率
        "teacher_acc": teacher_acc,    # 教师选择准确率 ⭐ 最重要！
    }

    # 添加每个教师的桶预测准确率
    for teacher in required_teachers:
        teacher_key = teacher.replace(".", "_").replace("-", "_")
        metrics[f"bucket_acc_{teacher_key}"] = per_teacher_bucket_acc[teacher]

    return metrics
