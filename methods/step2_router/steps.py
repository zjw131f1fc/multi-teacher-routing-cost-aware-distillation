"""训练和评估步骤函数"""

import torch
import torch.nn.functional as F
from typing import Dict, Any
import numpy as np


def train_step(batch: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """训练步骤

    参数:
        batch: 批次数据，包含:
            - input_ids: [batch_size, seq_len]
            - attention_mask: [batch_size, seq_len]
            - target_scores: [batch_size, num_teachers]
            - masks: [batch_size, num_teachers] (标记哪些教师有有效分数)
        info: 训练信息，包含:
            - models: 注册的模型字典
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
    # 获取模型
    router = info["models"]["router"]
    device = next(router.parameters()).device

    # 准备输入
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_scores = batch["target_scores"].to(device)
    masks = batch["masks"].to(device)

    # 前向传播
    pred_scores = router(input_ids, attention_mask)  # [batch, num_teachers]

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


def eval_step(batch: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """评估步骤

    参数:
        batch: 批次数据
        info: 评估信息

    返回:
        metrics: {
            "mse": MSE损失,
            "mae": MAE,
            "per_teacher_mse": 每个教师的MSE (list),
            "per_teacher_mae": 每个教师的MAE (list),
        }
    """
    # 获取模型
    router = info["models"]["router"]
    device = next(router.parameters()).device

    # 准备输入
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_scores = batch["target_scores"].to(device)
    masks = batch["masks"].to(device)

    # 前向传播
    with torch.no_grad():
        pred_scores = router(input_ids, attention_mask)  # [batch, num_teachers]

    # 计算整体 MSE 和 MAE
    mse_loss = F.mse_loss(pred_scores, target_scores, reduction='none')  # [batch, num_teachers]
    mse = (mse_loss * masks).sum() / (masks.sum() + 1e-8)

    mae = torch.abs(pred_scores - target_scores)
    mae = (mae * masks).sum() / (masks.sum() + 1e-8)

    # 计算每个教师的指标
    num_teachers = target_scores.shape[1]
    per_teacher_mse = []
    per_teacher_mae = []

    for t in range(num_teachers):
        teacher_mask = masks[:, t]  # [batch]
        if teacher_mask.sum() > 0:
            teacher_mse = (mse_loss[:, t] * teacher_mask).sum() / teacher_mask.sum()
            teacher_mae = (torch.abs(pred_scores[:, t] - target_scores[:, t]) * teacher_mask).sum() / teacher_mask.sum()
        else:
            teacher_mse = torch.tensor(0.0, device=device)
            teacher_mae = torch.tensor(0.0, device=device)

        per_teacher_mse.append(teacher_mse.item())
        per_teacher_mae.append(teacher_mae.item())

    return {
        "mse": mse.item(),
        "mae": mae.item(),
        "per_teacher_mse": per_teacher_mse,
        "per_teacher_mae": per_teacher_mae,
    }
