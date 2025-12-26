"""分析 Step2 训练数据的 prompt 长度

统计所有训练样本经过 prompt 模板处理后的 token 长度，给出最大长度建议。

使用方法:
    python analyze_prompt_length.py --config configs/step2_train_router.yaml
"""

import os
import json
from typing import Dict, Any
from tqdm import tqdm

from engine.configs.loader import load_config
from engine.backbones.loader import load_backbone
from engine.datas.loader import load_dataset


# ==================== Prompt 模板 ====================

ROUTER_PROMPT_TEMPLATE = """Analyze the following problem and determine which AI model would be most suitable to answer it.

Problem:
{instruction}

Analysis:"""


def build_router_prompt(instruction: str) -> str:
    """构建路由器的输入prompt"""
    return ROUTER_PROMPT_TEMPLATE.format(instruction=instruction)


# ==================== 预加载函数 ====================

def preload_fn(config: Dict) -> Dict[str, Any]:
    """预加载数据集和 NTE 分数"""
    logger = config["logger"]
    logger.info("预加载数据集...")

    # 加载数据集
    dataset_bundle = load_dataset(config)

    # 根据数据集名称加载文件
    dataset_name = config["dataset_settings"]["name"]
    supplement_file = f"method_cache/{dataset_name}/supplement.json"
    score_file = f"method_cache/{dataset_name}/step1_collect_scores.json"

    # ==================== 合并补充数据 ====================
    logger.info("合并补充数据...")

    if os.path.exists(supplement_file):
        logger.info(f"加载补充数据: {supplement_file}")

        with open(supplement_file, 'r', encoding='utf-8') as f:
            supplement_data = json.load(f)

        logger.info(f"加载了 {len(supplement_data)} 条补充数据")

        # 创建补充数据的索引（按 instruction 建立映射）
        supplement_index = {}
        for item in supplement_data:
            instruction = item.get("instruction", "")
            if instruction:
                supplement_index[instruction] = item

        logger.info(f"建立了 {len(supplement_index)} 条补充数据索引")

        # 合并数据：遍历 dataset_bundle 中的每个 split
        for split_name, split_dataset in dataset_bundle["splits"].items():
            merged_count = 0
            for i in range(len(split_dataset)):
                sample = split_dataset[i]
                instruction = sample.get("instruction", "")

                # 如果在补充数据中找到匹配的 instruction
                if instruction in supplement_index:
                    supplement_item = supplement_index[instruction]

                    # 合并 responses：将补充数据的 responses 添加到原有数据中
                    if "responses" in supplement_item:
                        for teacher_name, teacher_data in supplement_item["responses"].items():
                            # 如果原数据中没有这个教师的响应，则添加
                            if teacher_name not in sample.get("responses", {}):
                                sample["responses"][teacher_name] = teacher_data
                                merged_count += 1

            logger.info(f"Split '{split_name}': 合并了 {merged_count} 条教师响应")

        logger.info("补充数据合并完成!")
    else:
        logger.warning(f"补充数据文件不存在: {supplement_file}")

    # ==================== 加载 NTE 分数 ====================
    logger.info(f"加载 NTE 分数: {score_file}")

    if not os.path.exists(score_file):
        raise FileNotFoundError(
            f"NTE 分数文件不存在: {score_file}\n"
            f"请先运行 step1_collect_scores.py 收集分数"
        )

    with open(score_file, 'r', encoding='utf-8') as f:
        student_scores = json.load(f)

    logger.info(f"加载了 {len(student_scores)} 条 NTE 分数")

    # 合并 NTE 分数到数据集
    score_index = {}
    for item in student_scores:
        instruction = item.get("instruction", "")
        if instruction:
            score_index[instruction] = item.get("teacher_scores", {})

    logger.info(f"建立了 {len(score_index)} 条分数索引")

    # 合并分数到每个 split
    for split_name, split_dataset in dataset_bundle["splits"].items():
        merged_count = 0
        for i in range(len(split_dataset)):
            sample = split_dataset[i]
            instruction = sample.get("instruction", "")

            if instruction in score_index:
                teacher_scores = score_index[instruction]

                # 为每个教师添加 NTE 分数
                for teacher_name, scores_dict in teacher_scores.items():
                    if teacher_name in sample.get("responses", {}):
                        sample["responses"][teacher_name]["nte_scores"] = scores_dict
                        merged_count += 1

        logger.info(f"Split '{split_name}': 合并了 {merged_count} 条 NTE 分数")

    # 过滤数据：只保留包含所有必需教师且有 NTE 分数的样本
    required_teachers = config["method_settings"].get("required_teachers", None)

    if required_teachers:
        logger.info(f"过滤数据: 要求包含所有教师的 NTE 分数 {required_teachers}")

        for split_name, split_dataset in dataset_bundle["splits"].items():
            original_count = len(split_dataset)

            filtered_samples = []
            for sample in split_dataset:
                responses = sample.get("responses", {})

                # 检查是否包含所有教师且都有 NTE 分数
                has_all_scores = all(
                    teacher_name in responses and "nte_scores" in responses[teacher_name]
                    for teacher_name in required_teachers
                )

                if has_all_scores:
                    filtered_samples.append(sample)

            split_dataset.samples = filtered_samples
            filtered_count = len(filtered_samples)

            logger.info(
                f"Split '{split_name}': {original_count} -> {filtered_count} "
                f"(过滤掉 {original_count - filtered_count} 个样本)"
            )

    logger.info("数据准备完成!")

    # ==================== 手动划分训练集和验证集 ====================
    logger.info("手动划分训练集和验证集...")

    train_split = dataset_bundle["splits"]["train"]
    train_samples = train_split.samples

    # 划分比例: 80% train, 20% val
    total = len(train_samples)
    train_size = int(total * 0.8)

    # 随机打乱并划分
    import random
    random.seed(config["global_settings"]["seed"])
    shuffled_samples = train_samples.copy()
    random.shuffle(shuffled_samples)

    train_data = shuffled_samples[:train_size]
    val_data = shuffled_samples[train_size:]

    logger.info(f"原始训练数据: {total} 样本")
    logger.info(f"划分后 - 训练集: {len(train_data)} 样本")
    logger.info(f"划分后 - 验证集: {len(val_data)} 样本")

    # 重新打包为 dataset
    from engine.datas.base.distill import DistillDataset
    dataset_bundle["splits"]["train"] = DistillDataset(train_data)
    dataset_bundle["splits"]["val"] = DistillDataset(val_data)

    return {
        "dataset_bundle": dataset_bundle
    }


# ==================== 分析函数 ====================

def analyze_prompt_lengths(config: Dict, cache: Dict[str, Any]):
    """分析所有样本的 prompt 长度"""
    logger = config["logger"]
    dataset_bundle = cache["dataset_bundle"]

    logger.info("=" * 80)
    logger.info("开始分析 Prompt 长度")
    logger.info("=" * 80)

    # 加载 DeBERTa tokenizer
    logger.info("加载 DeBERTa Tokenizer...")
    encoder_backbone = load_backbone(config)
    tokenizer = encoder_backbone.tokenizer

    # 收集所有 prompts
    all_prompts = []
    for split_name in ["train", "val"]:
        if split_name not in dataset_bundle["splits"]:
            continue

        split_dataset = dataset_bundle["splits"][split_name]
        logger.info(f"\n处理 {split_name} 集: {len(split_dataset)} 个样本")

        for sample in tqdm(split_dataset, desc=f"构建 {split_name} prompts"):
            instruction = sample.get("instruction", "")
            prompt = build_router_prompt(instruction)
            all_prompts.append(prompt)

    logger.info(f"\n总共收集了 {len(all_prompts)} 个 prompts")

    # 统计 token 长度
    logger.info("\n开始 tokenization...")
    token_lengths = []

    for prompt in tqdm(all_prompts, desc="Tokenizing"):
        tokens = tokenizer(
            prompt,
            truncation=False,  # 不截断，统计真实长度
            return_tensors=None,
            add_special_tokens=True  # 包含特殊 token
        )
        token_lengths.append(len(tokens["input_ids"]))

    # 统计信息
    import numpy as np
    token_lengths = np.array(token_lengths)

    logger.info("\n" + "=" * 80)
    logger.info("Token 长度统计")
    logger.info("=" * 80)
    logger.info(f"总样本数: {len(token_lengths)}")
    logger.info(f"最小长度: {token_lengths.min()}")
    logger.info(f"最大长度: {token_lengths.max()}")
    logger.info(f"平均长度: {token_lengths.mean():.2f}")
    logger.info(f"中位数长度: {np.median(token_lengths):.2f}")
    logger.info(f"标准差: {token_lengths.std():.2f}")
    logger.info("")
    logger.info("百分位数:")
    for percentile in [50, 75, 90, 95, 99, 99.5, 100]:
        value = np.percentile(token_lengths, percentile)
        logger.info(f"  {percentile:5.1f}%: {value:.0f}")

    # 长度分布
    logger.info("\n长度分布:")
    bins = [0, 128, 256, 384, 512, 768, 1024, float('inf')]
    labels = ["0-128", "128-256", "256-384", "384-512", "512-768", "768-1024", "1024+"]

    for i in range(len(bins) - 1):
        count = ((token_lengths >= bins[i]) & (token_lengths < bins[i+1])).sum()
        percentage = count / len(token_lengths) * 100
        logger.info(f"  {labels[i]:12s}: {count:6d} ({percentage:5.2f}%)")

    # 建议
    logger.info("\n" + "=" * 80)
    logger.info("建议的 max_length 配置")
    logger.info("=" * 80)

    # 计算不同覆盖率下的长度
    for coverage in [95, 99, 99.5, 100]:
        max_len = np.percentile(token_lengths, coverage)
        # 向上取整到 128 的倍数
        suggested_len = int(np.ceil(max_len / 128) * 128)
        truncated = ((token_lengths > suggested_len).sum() / len(token_lengths) * 100)
        logger.info(f"覆盖 {coverage:5.1f}% 样本: max_length={suggested_len:4d} (截断 {truncated:.2f}% 样本)")

    # DeBERTa 最大长度
    logger.info(f"\nDeBERTa-v3-base 官方最大长度: 512")
    over_512 = (token_lengths > 512).sum()
    percentage_over_512 = over_512 / len(token_lengths) * 100
    logger.info(f"超过 512 的样本数: {over_512} ({percentage_over_512:.2f}%)")

    logger.info("\n" + "=" * 80)


# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="分析 Step2 训练数据的 prompt 长度")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/step2_train_router.yaml",
        help="配置文件路径"
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(override_file=args.config)
    logger = config["logger"]

    logger.info("=" * 60)
    logger.info("分析 Step2 Prompt 长度")
    logger.info("=" * 60)

    # 预加载数据
    cache = preload_fn(config)

    # 分析长度
    analyze_prompt_lengths(config, cache)

    logger.info("\n分析完成!")


if __name__ == "__main__":
    main()
