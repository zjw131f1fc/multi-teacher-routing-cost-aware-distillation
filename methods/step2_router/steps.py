"""训练和评估步骤函数"""

import torch
import torch.nn.functional as F
from typing import Dict, Any
import numpy as np


def _build_prompt(instruction: str) -> str:
    """构建路由器的输入prompt"""
    return f"""Analyze the following problem and determine which AI model would be most suitable to answer it.

Problem:
{instruction}

Analysis:"""


def _process_batch(batch, tokenizer, required_teachers, max_seq_length, device):
    """处理batch数据：tokenization + 提取target scores

    返回:
        {
            "input_ids": [batch_size, seq_len],
            "attention_mask": [batch_size, seq_len],
            "target_scores": [batch_size, num_teachers],
            "masks": [batch_size, num_teachers]
        }
    """
    # 为每个instruction添加prompt
    prompts = [_build_prompt(sample["instruction"]) for sample in batch]

    # Tokenize
    encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )

    # 提取目标 NTE 分数
    target_scores = []
    masks = []

    for sample in batch:
        scores = []
        mask = []

        for teacher in required_teachers:
            # 检查教师响应是否存在，且包含 nte_scores
            if (teacher in sample.get("responses", {}) and
                "nte_scores" in sample["responses"][teacher]):
                nte_score = sample["responses"][teacher]["nte_scores"]["nte_score"]
                scores.append(nte_score)
                mask.append(1.0)
            else:
                # 没有分数，用0占位，mask标记为0
                scores.append(0.0)
                mask.append(0.0)

        target_scores.append(scores)
        masks.append(mask)

    # 不对input_ids和attention_mask指定设备，让模型的device_map自动处理
    # 只将target和mask移到指定设备（用于loss计算）
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "target_scores": torch.tensor(target_scores, dtype=torch.float32, device=device),
        "masks": torch.tensor(masks, dtype=torch.float32, device=device)
    }


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
        losses: {
            "router": {
                "loss": 总损失,
                "mse": MSE损失,
                "mae": MAE (用于监控)
            }
        }
    """
    # 获取模型和配置
    router = info["models"]["router"]
    tokenizer = info["models"]["tokenizer"]
    required_teachers = info["models"]["required_teachers"]
    max_seq_length = info["models"]["max_seq_length"]

    # 处理batch：tokenization + 提取target
    processed = _process_batch(batch, tokenizer, required_teachers, max_seq_length, device)

    input_ids = processed["input_ids"]
    attention_mask = processed["attention_mask"]
    target_scores = processed["target_scores"]
    masks = processed["masks"]

    # 获取模型的dtype（用于target）
    model_dtype = next(router.parameters()).dtype

    # input_ids和attention_mask保持在CPU，让LLM的device_map自动处理
    # 只需将target转换为模型的dtype
    target_scores = target_scores.to(dtype=model_dtype)
    masks = masks.to(dtype=model_dtype)

    # 前向传播（device_map会自动处理输入的设备分配）
    pred_scores = router(input_ids, attention_mask)  # [batch, num_teachers]

    # 将预测结果移到target所在的设备
    pred_scores = pred_scores.to(target_scores.device)

    # 计算损失 (只对有效的教师计算)
    mse_loss = F.mse_loss(pred_scores, target_scores, reduction='none')  # [batch, num_teachers]
    mse_loss = (mse_loss * masks).sum() / (masks.sum() + 1e-8)  # 平均到有效的教师上

    # 计算 MAE 用于监控
    mae = torch.abs(pred_scores - target_scores)
    mae = (mae * masks).sum() / (masks.sum() + 1e-8)

    # 返回损失字典
    return {
        "router": {
            "loss": mse_loss,  # 总损失，用于反向传播
            "mse": mse_loss.detach(),  # 用于日志记录
            "mae": mae.detach(),  # 用于日志记录
        }
    }


def eval_step(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """评估步骤

    参数:
        batch: 批次数据（原始数据，list of samples）
        device: 设备（由trainer传入）
        info: 评估信息

    返回:
        metrics: {
            "mse": MSE损失,
            "mae": MAE,
            "per_teacher_mse": 每个教师的MSE (list),
            "per_teacher_mae": 每个教师的MAE (list),
        }
    """
    # 获取模型和配置
    router = info["models"]["router"]
    tokenizer = info["models"]["tokenizer"]
    required_teachers = info["models"]["required_teachers"]
    max_seq_length = info["models"]["max_seq_length"]

    # 处理batch：tokenization + 提取target
    processed = _process_batch(batch, tokenizer, required_teachers, max_seq_length, device)

    input_ids = processed["input_ids"]
    attention_mask = processed["attention_mask"]
    target_scores = processed["target_scores"]
    masks = processed["masks"]

    # 获取模型的dtype
    model_dtype = next(router.parameters()).dtype

    # input_ids和attention_mask保持在CPU，让LLM的device_map自动处理
    # 只需将target转换为模型的dtype
    target_scores = target_scores.to(dtype=model_dtype)
    masks = masks.to(dtype=model_dtype)

    # 前向传播（device_map会自动处理输入的设备分配）
    with torch.no_grad():
        pred_scores = router(input_ids, attention_mask)  # [batch, num_teachers]

    # 将预测结果移到target所在的设备
    pred_scores = pred_scores.to(target_scores.device)

    # 计算整体 MSE 和 MAE
    mse_loss = F.mse_loss(pred_scores, target_scores, reduction='none')  # [batch, num_teachers]
    mse = (mse_loss * masks).sum() / (masks.sum() + 1e-8)

    mae = torch.abs(pred_scores - target_scores)
    mae = (mae * masks).sum() / (masks.sum() + 1e-8)

    # 计算更实际的指标：Top-1 准确率
    # 对于每个样本，比较预测的最佳教师和真实的最佳教师是否一致
    pred_best_teacher = pred_scores.argmax(dim=1)  # [batch]
    true_best_teacher = target_scores.argmax(dim=1)  # [batch]
    top1_correct = (pred_best_teacher == true_best_teacher).float().mean()

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
