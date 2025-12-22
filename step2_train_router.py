"""Step 2: 训练路由模型

训练一个 Qwen-1.5B-Instruct 模型预测每个教师的 NTE 分数。

使用方法:
    python step2_train_router.py --config configs/step2_train_router.yaml
"""

import os
import json
import torch
from typing import Dict, Any
from torch.utils.data import DataLoader

from engine.configs.loader import load_config
from engine.backbones.loader import load_backbone
from engine.datas.loader import load_dataset
from engine.trainers.loader import load_trainer

from methods.step2_router import RouterRegressor, train_step, eval_step


# ==================== 数据准备 ====================

def collate_fn(batch, tokenizer, required_teachers, max_seq_length=512):
    """批次数据处理函数

    参数:
        batch: list of samples from dataset
        tokenizer: LLM tokenizer
        required_teachers: list of teacher names
        max_seq_length: 最大序列长度

    返回:
        {
            "input_ids": [batch, seq_len],
            "attention_mask": [batch, seq_len],
            "target_scores": [batch, num_teachers],
            "masks": [batch, num_teachers]
        }
    """
    instructions = [sample["instruction"] for sample in batch]

    # Tokenize
    encodings = tokenizer(
        instructions,
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

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "target_scores": torch.tensor(target_scores, dtype=torch.float32),
        "masks": torch.tensor(masks, dtype=torch.float32)
    }


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

    return {
        "dataset_bundle": dataset_bundle
    }


# ==================== 训练函数 ====================

def run_fn(config: Dict, cache: Dict[str, Any]) -> Dict[str, Any]:
    """执行训练"""
    logger = config["logger"]
    dataset_bundle = cache["dataset_bundle"]
    device = config.get("global_settings", {}).get("device")

    # 从配置中提取参数
    method_cfg = config["method_settings"]
    required_teachers = method_cfg["required_teachers"]
    num_teachers = len(required_teachers)
    max_seq_length = method_cfg.get("max_seq_length", 512)

    logger.info(f"教师列表: {required_teachers}")
    logger.info(f"教师数量: {num_teachers}")

    # 加载 LLM Backbone
    logger.info("加载 LLM Backbone...")
    llm_backbone = load_backbone(config)

    # 获取 hidden_dim
    hidden_dim = llm_backbone.model.config.hidden_size
    logger.info(f"Hidden dimension: {hidden_dim}")

    # 创建 RouterRegressor
    logger.info("创建 RouterRegressor...")
    router = RouterRegressor(
        llm_backbone=llm_backbone,
        num_teachers=num_teachers,
        hidden_dim=hidden_dim,
        dropout=method_cfg.get("dropout", 0.1)
    ).to(device)

    # 创建 Trainer
    logger.info("创建 Trainer...")
    trainer = load_trainer(config, dataset_bundle)

    # 注册模型
    trainer.register_model("router", router)

    # 添加参数组（全量微调）
    trainer.add_param_group("router", list(router.parameters()))

    # 创建优化器
    trainer.setup_optimizers()

    # 自定义 collate_fn
    def custom_collate(batch):
        return collate_fn(batch, llm_backbone.tokenizer, required_teachers, max_seq_length)

    # 设置 collate_fn (通过修改 trainer 的 dataloaders)
    # 注意: 这里需要重新创建 DataLoader
    for split_name, split_dataset in dataset_bundle["splits"].items():
        batch_size = config["trainer_settings"]["dl_settings"]["batch_size"]
        shuffle = (split_name == "train")

        dataloader = DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=custom_collate,
            num_workers=0  # 避免多进程问题
        )

        trainer.dataloaders[split_name] = dataloader

    # 注册训练和评估步骤
    trainer.register_train_step(train_step)
    trainer.register_eval_step(eval_step)

    # 执行训练
    logger.info("开始训练...")
    result = trainer.run()

    return result


# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 2: 训练路由模型")
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
    logger.info("Step 2: 训练路由模型")
    logger.info("=" * 60)

    # 不使用 Manager，直接执行
    cache = preload_fn(config)
    result = run_fn(config, cache)

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
