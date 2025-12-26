"""训练和评估步骤函数"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List


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


def _process_batch(batch, required_teachers, device):
    """处理batch数据：构建prompts + 提取target scores

    参数:
        batch: list of samples from dataset
        required_teachers: list of teacher names
        device: 设备

    返回:
        {
            "prompts": [batch_size] 文本列表,
            "target_scores": [batch_size, num_teachers],
            "masks": [batch_size, num_teachers]
        }
    """
    # 构建prompts
    prompts = [_build_prompt(sample["instruction"]) for sample in batch]

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

    return {
        "prompts": prompts,
        "target_scores": torch.tensor(target_scores, dtype=torch.float32, device=device),
        "masks": torch.tensor(masks, dtype=torch.float32, device=device)
    }


def train_step(batch: Dict[str, Any], device: str, info: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """训练步骤

    参数:
        batch: 批次数据（原始数据，list of samples）
        device: 设备（由trainer传入）
        info: 训练信息，包含:
            - models: 注册的模型字典（包含router, required_teachers）
            - config: 配置对象

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
    required_teachers = info["models"]["required_teachers"]

    # 处理batch：构建prompts + 提取target
    processed = _process_batch(batch, required_teachers, device)

    prompts = processed["prompts"]
    target_scores = processed["target_scores"]
    masks = processed["masks"]

    # 获取模型的dtype（用于target）
    model_dtype = next(router.parameters()).dtype
    target_scores = target_scores.to(dtype=model_dtype)
    masks = masks.to(dtype=model_dtype)

    # 前向传播
    pred_scores = router(prompts)  # [batch, num_teachers]

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
        metrics: {
            "mse": MSE损失,
            "mae": MAE,
            "top1_acc": Top-1准确率,
            "pearson": Pearson相关系数,
        }
    """
    # 获取模型和配置
    router = info["models"]["router"]
    required_teachers = info["models"]["required_teachers"]

    # 处理batch：构建prompts + 提取target
    processed = _process_batch(batch, required_teachers, device)

    prompts = processed["prompts"]
    target_scores = processed["target_scores"]
    masks = processed["masks"]

    # 获取模型的dtype
    model_dtype = next(router.parameters()).dtype
    target_scores = target_scores.to(dtype=model_dtype)
    masks = masks.to(dtype=model_dtype)

    # 前向传播
    with torch.no_grad():
        pred_scores = router(prompts)  # [batch, num_teachers]

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
