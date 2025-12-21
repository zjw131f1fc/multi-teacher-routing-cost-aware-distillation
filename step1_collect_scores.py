"""Step 1: 收集学生模型的 NTE 分数

计算学生模型对每个 teacher response 的似然概率，并计算 NTE 分数。
保存到 method_cache/student_scores.json

NTE Score 公式:
    NTE(x,y) = V(y) * M_prox(p)

    其中:
    - V(y): 教师响应的质量分数（来自 rewards）
    - M_prox(p): 近端可学习性，基于学生模型的似然概率 p
        M_prox = (p^α * (1-p)^β) / Z
        建议参数: α=2.0, β=0.5
"""

import os
import json
import torch
import logging
from tqdm import tqdm
from typing import Dict, Any, List
import numpy as np

from engine.configs.loader import load_config
from engine.backbones.loader import load_backbone
from engine.datas.loader import load_dataset


# ==================== NTE 分数计算 ====================

def compute_beta_normalization(alpha: float, beta: float) -> float:
    """计算 Beta 分布的归一化常数 Z，使得最大值为 1

    对于 f(p) = p^α * (1-p)^β，最大值出现在 p* = α/(α+β)
    因此 Z = (p*)^α * (1-p*)^β
    """
    p_star = alpha / (alpha + beta)
    Z = (p_star ** alpha) * ((1 - p_star) ** beta)
    return Z


def compute_proximity_score(p: float, alpha: float = 2.0, beta: float = 0.5, Z: float = None) -> float:
    """计算近端可学习性分数 M_prox

    参数:
        p: 学生模型的似然概率 [0, 1]
        alpha: Beta 分布参数，控制对高 p 的偏好
        beta: Beta 分布参数，控制对不确定性的考虑
        Z: 归一化常数，如果为 None 则自动计算

    返回:
        M_prox: [0, 1] 之间的分数
    """
    if Z is None:
        Z = compute_beta_normalization(alpha, beta)

    # 防止数值问题
    p = np.clip(p, 1e-8, 1 - 1e-8)

    score = (p ** alpha) * ((1 - p) ** beta) / Z
    return float(score)


def compute_nte_score(
    quality: float,
    proximity: float,
    quality_power: float = 0.3,
    proximity_power: float = 0.7,
    quality_floor: float = 0.1
) -> float:
    """计算 NTE (Normalized Teaching Efficacy) 分数

    使用 Soft Quality + 指数平滑组合:
        V_soft(y) = ε + (1-ε) × V(y)  (将 [0,1] 映射到 [ε, 1])
        NTE = V_soft(y)^α × M_prox(p)^β

    这样即使 quality=0，仍能通过 proximity 区分不同的响应。

    参数:
        quality: 教师响应的质量分数 V(y) [0, 1]（通常为离散的 0 或 1）
        proximity: 近端可学习性分数 M_prox [0, 1]
        quality_power: 质量分数的指数 α (default=0.3, 推荐值)
        proximity_power: 可学习性分数的指数 β (default=0.7, 推荐值)
        quality_floor: 质量分数的下限 ε (default=0.1)

    返回:
        NTE: [0, 1] 之间的分数（自动归一化）

    说明:
        - α<β (推荐 α=0.3, β=0.7): 更重视可学习性，适合离散质量分数
        - α=β=0.5: 两者平衡，类似几何平均
        - α>β: 更重视质量
        - quality_floor: 控制低质量响应的权重
          - ε=0.1: 低质量响应仍保留 10% 的权重（推荐）
          - ε=0.0: 退化为原始方案，quality=0 时 NTE=0
          - ε=0.5: 高低质量响应权重差异较小

    示例 (α=0.3, β=0.7, ε=0.1):
        - Quality=0, Proximity=0.9: V_soft=0.1, NTE = 0.1^0.3 × 0.9^0.7 ≈ 0.478
        - Quality=0, Proximity=0.5: V_soft=0.1, NTE = 0.1^0.3 × 0.5^0.7 ≈ 0.314
        - Quality=0, Proximity=0.1: V_soft=0.1, NTE = 0.1^0.3 × 0.1^0.7 ≈ 0.158
        - Quality=1, Proximity=0.9: V_soft=1.0, NTE = 1.0^0.3 × 0.9^0.7 ≈ 0.933
        - Quality=1, Proximity=0.1: V_soft=1.0, NTE = 1.0^0.3 × 0.1^0.7 ≈ 0.200
        → 即使 quality=0，proximity 的差异仍能显著体现 (0.478 vs 0.158)
        → proximity 主导: 低质量+高可学习性 (0.478) > 高质量+低可学习性 (0.200)
    """
    # 防止数值问题
    quality = np.clip(quality, 0.0, 1.0)
    proximity = np.clip(proximity, 0.0, 1.0)

    # Soft Quality: 将 [0, 1] 映射到 [quality_floor, 1]
    quality_soft = quality_floor + (1.0 - quality_floor) * quality

    # 指数平滑组合
    nte = (quality_soft ** quality_power) * (proximity ** proximity_power)

    # 最终保证在 [0, 1]
    # 注意: 因为 quality_soft ∈ [ε, 1], proximity ∈ [0, 1]
    # 所以 NTE ∈ [0, 1] (当 ε > 0 时最小值 > 0)
    return float(np.clip(nte, 0.0, 1.0))


# ==================== 似然概率计算 ====================

def compute_sequence_likelihood(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str = "cuda"
) -> float:
    """计算学生模型对 response 的长度归一化似然概率

    参数:
        model: 学生模型（LLM）
        tokenizer: 分词器
        prompt: 输入问题
        response: 教师响应
        device: 设备

    返回:
        p: Geometric Mean Likelihood，即 exp(mean_log_prob)
    """
    # 构造完整输入
    full_text = prompt + response

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)

    prompt_length = inputs["input_ids"].shape[1]
    full_length = full_inputs["input_ids"].shape[1]

    # 如果 response 为空或太短，返回很小的概率
    if full_length <= prompt_length:
        return 1e-8

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=full_inputs["input_ids"],
            attention_mask=full_inputs["attention_mask"]
        )
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # 计算 response 部分的 log likelihood
    # 注意：logits[i] 预测的是 token[i+1]
    response_logits = logits[:, prompt_length-1:full_length-1, :]  # [1, response_len, vocab]
    response_token_ids = full_inputs["input_ids"][:, prompt_length:full_length]  # [1, response_len]

    # 计算 log probabilities
    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)

    # 提取每个 token 的 log prob
    token_log_probs = log_probs.gather(
        dim=-1,
        index=response_token_ids.unsqueeze(-1)
    ).squeeze(-1)  # [1, response_len]

    # 计算平均 log prob (长度归一化)
    mean_log_prob = token_log_probs.mean().item()

    # 转换为概率 (Geometric Mean Likelihood)
    likelihood = np.exp(mean_log_prob)

    # Clip to [0, 1]
    likelihood = np.clip(likelihood, 0.0, 1.0)

    return likelihood


# ==================== 主流程 ====================

def collect_student_scores(config_path: str = "configs/step1_collect_scores.yaml"):
    """收集学生模型的 NTE 分数

    参数:
        config_path: 配置文件路径，包含所有必需的配置参数
    """
    # 加载配置
    print("加载配置...")
    config = load_config(override_file=config_path)
    logger = config.logger
    device = config.global_settings["device"]

    # 从配置中提取参数
    method_cfg = config.method_settings
    dataset_cfg = config.dataset_settings

    alpha = method_cfg.get("nte_alpha", 2.0)
    beta = method_cfg.get("nte_beta", 0.5)
    quality_power = method_cfg.get("nte_quality_power", 0.3)
    proximity_power = method_cfg.get("nte_proximity_power", 0.7)
    quality_floor = method_cfg.get("nte_quality_floor", 0.1)
    max_samples = method_cfg.get("max_samples", None)
    reward_key = method_cfg.get("reward_key", "math_verify")
    study_name = method_cfg.get("study_name", "step1_collect_scores")

    # 根据数据集名称自动生成文件名
    dataset_name = dataset_cfg["name"]  # e.g., "distill-openr1-math"

    # 文件路径：固定使用 "step1_collect_scores.json"
    # study_name 仅用于日志标识，不影响文件名
    output_file = f"method_cache/{dataset_name}/step1_collect_scores.json"
    supplement_file = f"method_cache/{dataset_name}/supplement.json"

    # 创建输出目录（包含 dataset_name 子目录）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logger.info(f"Study Name: {study_name}")
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"Reward Key: {reward_key}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"补充数据: {supplement_file}")

    # 加载学生模型（LLM Backbone）
    logger.info("加载学生模型...")
    student_backbone = load_backbone(config)
    student_model = student_backbone.model
    tokenizer = student_backbone.tokenizer
    student_model.eval()

    # 加载数据集
    logger.info("加载数据集...")
    dataset_bundle = load_dataset(config)

    # 合并补充数据（复用 main 中的逻辑）
    logger.info("合并补充数据...")
    if os.path.exists(supplement_file):
        with open(supplement_file, 'r') as f:
            supplement_data = json.loads(f.read())

        supplement_index = {item["instruction"]: item for item in supplement_data if "instruction" in item}

        for split_name, split_dataset in dataset_bundle["splits"].items():
            for i in range(len(split_dataset)):
                sample = split_dataset[i]
                instruction = sample.get("instruction", "")
                if instruction in supplement_index:
                    supplement_item = supplement_index[instruction]
                    if "responses" in supplement_item:
                        for teacher_name, teacher_data in supplement_item["responses"].items():
                            if teacher_name not in sample["responses"]:
                                sample["responses"][teacher_name] = teacher_data
    else:
        logger.warning(f"补充数据文件不存在: {supplement_file}")

    # 过滤数据（保留包含所有必需教师的样本）
    required_teachers = config.method_settings.get("required_teachers", None)
    if required_teachers:
        logger.info(f"过滤数据: 要求包含所有这些教师 {required_teachers}")
        for split_name, split_dataset in dataset_bundle["splits"].items():
            filtered_samples = []
            for sample in split_dataset:
                responses = sample.get("responses", {})
                if all(t in responses for t in required_teachers):
                    filtered_samples.append(sample)
            split_dataset.samples = filtered_samples

    # 计算归一化常数
    Z = compute_beta_normalization(alpha, beta)
    logger.info(f"Beta 归一化常数 Z = {Z:.6f} (α={alpha}, β={beta})")
    logger.info(f"NTE 指数参数: quality_power={quality_power}, proximity_power={proximity_power}")
    logger.info(f"NTE Quality Floor: {quality_floor} (低质量响应保留 {quality_floor*100:.0f}% 权重)")

    # 收集所有样本的分数
    all_scores = []

    # 只处理训练集
    train_dataset = dataset_bundle["splits"]["train"]
    dataset_total = len(train_dataset)
    total_samples = dataset_total if max_samples is None else min(max_samples, dataset_total)

    logger.info(f"数据集总样本数: {dataset_total}")
    logger.info(f"开始计算 NTE 分数，共 {total_samples} 个样本...")

    for idx in tqdm(range(total_samples), desc="计算 NTE 分数"):
        sample = train_dataset[idx]
        instruction = sample["instruction"]
        responses = sample["responses"]

        # 对每个教师的响应计算分数
        teacher_scores = {}

        for teacher_name, teacher_data in responses.items():
            # 提取教师响应文本
            messages = teacher_data.get("messages", [])
            if not messages:
                continue

            # messages 应该是列表格式: [{role: "user", content: "..."}, ...]
            response_text = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    response_text = msg.get("content", "")

            if not response_text:
                continue

            # 1. 计算似然概率
            likelihood = compute_sequence_likelihood(
                model=student_model,
                tokenizer=tokenizer,
                prompt=instruction,
                response=response_text,
                device=device
            )

            # 2. 计算近端可学习性分数
            proximity = compute_proximity_score(likelihood, alpha, beta, Z)

            # 3. 提取质量分数（从 rewards 中，使用配置的 reward_key）
            rewards = teacher_data.get("rewards", {})
            if reward_key in rewards:
                quality = rewards[reward_key]
                # 归一化到 [0, 1]（假设 reward 已经在 [0, 1] 范围内）
                quality = np.clip(quality, 0.0, 1.0)
            else:
                # 如果没有指定的 reward_key，跳过这个教师
                logger.warning(f"教师 {teacher_name} 缺少 {reward_key} 分数，跳过")
                continue

            # 4. 计算 NTE 分数
            nte_score = compute_nte_score(
                quality, proximity, quality_power, proximity_power, quality_floor
            )

            # 保存结果
            teacher_scores[teacher_name] = {
                "likelihood": float(likelihood),
                "proximity": float(proximity),
                "quality": float(quality),
                "nte_score": float(nte_score)
            }

        # 记录这个样本的所有分数
        all_scores.append({
            "instruction": instruction,
            "teacher_scores": teacher_scores
        })

    # 保存到文件
    logger.info(f"保存分数到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    logger.info(f"完成！共收集 {len(all_scores)} 个样本的分数")

    # 打印详细的统计信息
    all_nte_scores = []
    all_likelihoods = []
    all_proximities = []
    all_qualities = []
    teacher_stats = {}  # 每个教师的统计信息

    for item in all_scores:
        for teacher_name, scores in item["teacher_scores"].items():
            all_nte_scores.append(scores["nte_score"])
            all_likelihoods.append(scores["likelihood"])
            all_proximities.append(scores["proximity"])
            all_qualities.append(scores["quality"])

            # 按教师分类统计
            if teacher_name not in teacher_stats:
                teacher_stats[teacher_name] = {
                    "nte_scores": [],
                    "likelihoods": [],
                    "proximities": [],
                    "qualities": []
                }
            teacher_stats[teacher_name]["nte_scores"].append(scores["nte_score"])
            teacher_stats[teacher_name]["likelihoods"].append(scores["likelihood"])
            teacher_stats[teacher_name]["proximities"].append(scores["proximity"])
            teacher_stats[teacher_name]["qualities"].append(scores["quality"])

    if all_nte_scores:
        logger.info("\n" + "="*60)
        logger.info("整体统计 (所有教师)")
        logger.info("="*60)

        logger.info(f"\nNTE 分数统计 (n={len(all_nte_scores)}):")
        logger.info(f"  Mean: {np.mean(all_nte_scores):.4f}")
        logger.info(f"  Std:  {np.std(all_nte_scores):.4f}")
        logger.info(f"  Min:  {np.min(all_nte_scores):.4f}")
        logger.info(f"  Max:  {np.max(all_nte_scores):.4f}")
        logger.info(f"  Median: {np.median(all_nte_scores):.4f}")

        logger.info(f"\n似然概率 (Likelihood) 统计:")
        logger.info(f"  Mean: {np.mean(all_likelihoods):.4f}")
        logger.info(f"  Std:  {np.std(all_likelihoods):.4f}")
        logger.info(f"  Min:  {np.min(all_likelihoods):.4f}")
        logger.info(f"  Max:  {np.max(all_likelihoods):.4f}")
        logger.info(f"  Median: {np.median(all_likelihoods):.4f}")

        logger.info(f"\n近端可学习性 (Proximity) 统计:")
        logger.info(f"  Mean: {np.mean(all_proximities):.4f}")
        logger.info(f"  Std:  {np.std(all_proximities):.4f}")
        logger.info(f"  Min:  {np.min(all_proximities):.4f}")
        logger.info(f"  Max:  {np.max(all_proximities):.4f}")
        logger.info(f"  Median: {np.median(all_proximities):.4f}")

        logger.info(f"\n质量分数 (Quality) 统计:")
        logger.info(f"  Mean: {np.mean(all_qualities):.4f}")
        logger.info(f"  Std:  {np.std(all_qualities):.4f}")
        logger.info(f"  Min:  {np.min(all_qualities):.4f}")
        logger.info(f"  Max:  {np.max(all_qualities):.4f}")
        logger.info(f"  Median: {np.median(all_qualities):.4f}")

        # 按教师分别统计
        logger.info("\n" + "="*60)
        logger.info("各教师分别统计")
        logger.info("="*60)

        for teacher_name, stats in teacher_stats.items():
            logger.info(f"\n教师: {teacher_name} (n={len(stats['nte_scores'])})")
            logger.info(f"  NTE Mean: {np.mean(stats['nte_scores']):.4f}, Std: {np.std(stats['nte_scores']):.4f}")
            logger.info(f"  Likelihood Mean: {np.mean(stats['likelihoods']):.4f}, Std: {np.std(stats['likelihoods']):.4f}")
            logger.info(f"  Proximity Mean: {np.mean(stats['proximities']):.4f}, Std: {np.std(stats['proximities']):.4f}")
            logger.info(f"  Quality Mean: {np.mean(stats['qualities']):.4f}, Std: {np.std(stats['qualities']):.4f}")

        logger.info("\n" + "="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: 收集学生模型的 NTE 分数")
    parser.add_argument("--config", type=str, default="configs/step1_collect_scores.yaml",
                        help="配置文件路径")

    args = parser.parse_args()

    collect_student_scores(config_path=args.config)
