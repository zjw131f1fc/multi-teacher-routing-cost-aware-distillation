"""Step 2: 训练路由模型

训练一个 DeBERTa-v3-base 模型（全量微调）预测每个教师的 NTE 桶（分类模式）。

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

from methods.step2_router import RouterClassifier, train_step, eval_step


# ==================== Prompt 模板 ====================

ROUTER_PROMPT_TEMPLATE = """Analyze the following problem and determine which AI model would be most suitable to answer it.

Problem:
{instruction}

Analysis:"""


def build_router_prompt(instruction: str) -> str:
    """构建路由器的输入prompt"""
    return ROUTER_PROMPT_TEMPLATE.format(instruction=instruction)


# ==================== 数据准备 ====================

# collate_fn 已移除，不再需要手动 tokenization


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

    # ==================== 自适应桶划分（教师特定）====================
    required_teachers = config["method_settings"].get("required_teachers", [])
    use_bucketing = config["method_settings"].get("use_bucketing", False)
    adaptive_bucketing = config["method_settings"].get("adaptive_bucketing", False)

    if use_bucketing and adaptive_bucketing:
        logger.info("="*80)
        logger.info("自适应桶划分（教师特定）: 为每个教师独立分桶")
        logger.info("="*80)

        # 收集每个教师的NTE分数（分别存储）
        teacher_scores = {teacher: [] for teacher in required_teachers}
        train_split = dataset_bundle["splits"]["train"]

        for sample in train_split:
            responses = sample.get("responses", {})
            for teacher_name in required_teachers:
                if (teacher_name in responses and
                    "nte_scores" in responses[teacher_name]):
                    nte_score = responses[teacher_name]["nte_scores"]["nte_score"]
                    teacher_scores[teacher_name].append(nte_score)

        # 检查是否有足够的数据
        valid_teachers = {k: v for k, v in teacher_scores.items() if len(v) > 0}

        if len(valid_teachers) > 0:
            import numpy as np
            from methods.step2_router.adaptive_bucketing import adaptive_bucketing_per_teacher

            # 转换为 numpy arrays
            teacher_scores_np = {k: np.array(v) for k, v in valid_teachers.items()}

            # 获取分桶配置
            num_buckets = config["method_settings"].get("num_buckets", 5)
            bucketing_method = config["method_settings"].get("bucketing_method", "kmeans")
            balance_weight = config["method_settings"].get("bucketing_balance_weight", 0.0)

            logger.info(f"桶数量: {num_buckets}")
            logger.info(f"分桶方法: {bucketing_method}")
            logger.info(f"均匀性权重: {balance_weight}")

            # 为每个教师独立计算分桶
            bucketing_results = adaptive_bucketing_per_teacher(
                teacher_scores=teacher_scores_np,
                num_buckets=num_buckets,
                method=bucketing_method,
                balance_weight=balance_weight,
                logger=logger
            )

            # 提取每个教师的桶配置
            teacher_bucket_ranges = {}
            teacher_bucket_centers = {}

            for teacher_name, result in bucketing_results.items():
                teacher_bucket_ranges[teacher_name] = result["bucket_ranges"]
                teacher_bucket_centers[teacher_name] = result["bucket_centers"]

            # 更新配置（保存教师特定的桶配置）
            config["method_settings"]["num_buckets"] = num_buckets
            config["method_settings"]["teacher_bucket_ranges"] = teacher_bucket_ranges
            config["method_settings"]["teacher_bucket_centers"] = teacher_bucket_centers

            logger.info(f"\n已更新配置为教师特定分桶:")
            logger.info(f"  num_buckets: {num_buckets}")
            for teacher_name in required_teachers:
                if teacher_name in teacher_bucket_ranges:
                    logger.info(f"  {teacher_name}:")
                    logger.info(f"    bucket_ranges: {[round(x, 2) for x in teacher_bucket_ranges[teacher_name]]}")
                    logger.info(f"    bucket_centers: {[round(x, 2) for x in teacher_bucket_centers[teacher_name]]}")
        else:
            logger.warning("未找到任何教师的NTE分数，无法计算自适应桶边界")
            logger.warning("将使用配置文件中的固定桶边界（如果有）")
    elif use_bucketing:
        # 如果没有开启自适应分桶，检查是否已配置教师特定桶边界
        if "teacher_bucket_ranges" in config["method_settings"]:
            logger.info("使用配置文件中的教师特定桶边界")
        else:
            logger.info("使用配置文件中的固定桶边界（所有教师共享）")
            logger.info(f"  bucket_ranges: {config['method_settings'].get('bucket_ranges')}")

    logger.info("="*80)

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

    logger.info(f"教师列表: {required_teachers}")
    logger.info(f"教师数量: {num_teachers}")

    # 加载 DeBERTa Encoder Backbone
    logger.info("加载 DeBERTa Encoder Backbone...")
    encoder_backbone = load_backbone(config)

    # 获取 hidden_dim
    hidden_dim = config["backbone_settings"]["encoder_settings"]["hidden_dim"]
    logger.info(f"Hidden dimension: {hidden_dim}")

    # 检查是否启用桶化模式
    use_bucketing = method_cfg.get("use_bucketing", False)

    if use_bucketing:
        # 分类模式（桶化）
        logger.info("="*80)
        logger.info("创建 RouterClassifier (分类模式 - 桶化)...")
        logger.info("="*80)

        num_buckets = method_cfg.get("num_buckets", 5)
        teacher_bucket_ranges = method_cfg.get("teacher_bucket_ranges", {})
        teacher_bucket_centers = method_cfg.get("teacher_bucket_centers", {})

        logger.info(f"桶数量: {num_buckets}")

        # 验证桶配置
        if not teacher_bucket_ranges or not teacher_bucket_centers:
            raise ValueError(
                "桶化模式需要 teacher_bucket_ranges 和 teacher_bucket_centers 配置！\n"
                "这些配置应该在 preload_fn 中通过自适应分桶生成。"
            )

        # 打印每个教师的桶配置
        for teacher_name in required_teachers:
            if teacher_name in teacher_bucket_ranges:
                logger.info(f"  {teacher_name}:")
                logger.info(f"    bucket_ranges: {[round(x, 4) for x in teacher_bucket_ranges[teacher_name]]}")
                logger.info(f"    bucket_centers: {[round(x, 4) for x in teacher_bucket_centers[teacher_name]]}")

        model_kwargs = {
            "encoder_backbone": encoder_backbone,
            "num_teachers": num_teachers,
            "teacher_names": required_teachers,
            "num_buckets": num_buckets,
            "hidden_dim": hidden_dim,
            "dropout": method_cfg.get("dropout", 0.1)
        }

        router = RouterClassifier(**model_kwargs)

    else:
        # 回归模式（向后兼容）
        logger.info("创建 RouterRegressor (回归模式)...")

        from methods.step2_router import RouterRegressor  # 导入回归模型

        model_kwargs = {
            "encoder_backbone": encoder_backbone,
            "num_teachers": num_teachers,
            "hidden_dim": hidden_dim,
            "dropout": method_cfg.get("dropout", 0.1),
            "score_scale": 1.0
        }

        router = RouterRegressor(**model_kwargs)

    # 重新组织数据集: 将 val 作为 test 传给 trainer
    # Trainer 的 evaluate() 方法会使用 splits["test"]
    logger.info("重新组织数据集: 使用 val 作为评估集...")
    reorganized_bundle = {
        "splits": {
            "train": dataset_bundle["splits"]["train"],
            "test": dataset_bundle["splits"]["val"]  # 使用 val 作为评估集
        },
        "meta": dataset_bundle["meta"],
        "judge": dataset_bundle.get("judge")
    }

    logger.info(f"训练集大小: {len(reorganized_bundle['splits']['train'])}")
    logger.info(f"验证集大小: {len(reorganized_bundle['splits']['test'])}")

    # 创建 Trainer
    logger.info("创建 Trainer...")
    trainer = load_trainer(config, reorganized_bundle)

    # 注册模型
    trainer.register_model("router", router)

    # 添加参数组（全量微调）
    trainer.add_param_group("router", list(router.parameters()))

    # 创建优化器
    trainer.setup_optimizers()

    # 注册额外信息（传递给train_step和eval_step）
    trainer.register_model("required_teachers", required_teachers)

    if use_bucketing:
        # 桶化模式：传递桶配置
        trainer.register_model("teacher_bucket_ranges", teacher_bucket_ranges)
        trainer.register_model("teacher_bucket_centers", teacher_bucket_centers)

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

    # 使用 Manager 的 direct 模式
    from engine.managers.loader import load_manager

    manager = load_manager(
        config=config,
        preload_fn=preload_fn,
        run_fn=run_fn,
        task_generator_fn=None,  # 单任务模式（direct模式）
        result_handler_fn=None
    )

    # 启动训练
    manager.start()
    manager.wait()

    # 获取结果摘要
    summary = manager.get_summary()
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"总任务数: {summary['total_tasks']}")
    logger.info(f"已完成: {summary['completed']}")
    logger.info(f"失败: {summary['failed']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
