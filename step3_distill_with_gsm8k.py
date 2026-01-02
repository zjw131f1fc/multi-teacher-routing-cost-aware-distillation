"""Step 3: 知识蒸馏（使用多数据集模式：蒸馏训练 + GSM8K 测试）

使用训练好的路由器选择最优教师，蒸馏到学生模型（Qwen2.5-0.5B-Instruct）。
- 训练数据来自蒸馏数据集（OpenR1-Math）
- 测试数据来自 QA 数据集（GSM8K）

使用新的多数据集加载模式，dataset_settings 配置为列表，返回多个 bundle。

使用方法:
    python step3_distill_with_gsm8k.py --config configs/step3_distill_with_gsm8k.yaml
"""

import os
import json
import torch
from typing import Dict, Any

from engine.configs.loader import load_config
from engine.backbones.loader import load_backbone
from engine.datas.loader import load_dataset
from engine.trainers.loader import load_trainer

from methods.step3_distill import route_to_teacher
from methods.step3_distill.steps_with_qa import train_step, eval_step


# ==================== 预加载函数 ====================

def preload_fn(config: Dict) -> Dict[str, Any]:
    """预加载数据集、路由器和教师模型"""
    logger = config["logger"]
    logger.info("预加载数据集...")

    # 使用新的多数据集加载模式
    # 返回列表: [distill_bundle, qa_bundle]
    dataset_bundles = load_dataset(config)

    logger.info(f"加载了 {len(dataset_bundles)} 个数据集")

    # 分离两个数据集
    distill_bundle = dataset_bundles[0]  # 蒸馏数据集（OpenR1-Math）
    qa_bundle = dataset_bundles[1]       # QA 数据集（GSM8K）

    logger.info(f"蒸馏数据集: {distill_bundle['meta'].get('dataset_name', 'N/A')}")
    logger.info(f"QA 数据集: {qa_bundle['meta'].get('dataset_name', 'N/A')}")

    # 根据数据集名称加载文件
    dataset_name = "distill-openr1-math"  # 使用 OpenR1-Math 的缓存
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

        # 合并数据：只对蒸馏数据集的训练集进行合并
        train_split = distill_bundle["splits"].get("train")
        if train_split:
            merged_count = 0
            for i in range(len(train_split)):
                sample = train_split[i]
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

            logger.info(f"训练集: 合并了 {merged_count} 条教师响应")

        logger.info("补充数据合并完成!")
    else:
        logger.warning(f"补充数据文件不存在: {supplement_file}")

    # ==================== 检查是否需要 NTE 分数 ====================
    # 定义需要 NTE 分数的路由策略
    routing_strategy = config["method_settings"].get("routing_strategy", "random")

    # 需要 NTE 分数的策略列表
    # - nte_best: 根据 NTE 分数选择最佳教师
    # - nte_weighted: 根据 NTE 分数加权采样教师
    # - knn_stats: 基于 KNN 和统计信息的路由（需要历史分数）
    strategies_need_scores = ["nte_best", "nte_weighted", "knn_stats"]

    # 不需要 NTE 分数的策略：
    # - best_teacher: 直接使用配置指定的最佳教师
    # - random: 随机选择教师
    # - round_robin: 轮流选择教师
    # 如需添加新策略，请更新上述 strategies_need_scores 列表

    need_nte_scores = routing_strategy in strategies_need_scores
    logger.info(f"路由策略: {routing_strategy}")
    logger.info(f"是否需要 NTE 分数: {need_nte_scores}")

    if need_nte_scores:
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

        # 合并 NTE 分数到训练集
        score_index = {}
        for item in student_scores:
            instruction = item.get("instruction", "")
            if instruction:
                score_index[instruction] = item.get("teacher_scores", {})

        logger.info(f"建立了 {len(score_index)} 条分数索引")

        # 只对蒸馏数据集的训练集合并分数
        train_split = distill_bundle["splits"].get("train")
        if train_split:
            merged_count = 0
            for i in range(len(train_split)):
                sample = train_split[i]
                instruction = sample.get("instruction", "")

                if instruction in score_index:
                    teacher_scores = score_index[instruction]

                    # 为每个教师添加 NTE 分数
                    for teacher_name, scores_dict in teacher_scores.items():
                        if teacher_name in sample.get("responses", {}):
                            sample["responses"][teacher_name]["nte_scores"] = scores_dict
                            merged_count += 1

            logger.info(f"训练集: 合并了 {merged_count} 条 NTE 分数")

        # 过滤训练集：只保留包含所有必需教师且有 NTE 分数的样本
        required_teachers = config["method_settings"].get("required_teachers", None)

        if required_teachers and train_split:
            logger.info(f"过滤训练集: 要求包含所有教师的 NTE 分数 {required_teachers}")

            original_count = len(train_split)

            filtered_samples = []
            for sample in train_split:
                responses = sample.get("responses", {})

                # 检查是否包含所有教师且都有 NTE 分数
                has_all_scores = all(
                    teacher_name in responses and "nte_scores" in responses[teacher_name]
                    for teacher_name in required_teachers
                )

                if has_all_scores:
                    filtered_samples.append(sample)

            train_split.samples = filtered_samples
            filtered_count = len(filtered_samples)

            logger.info(
                f"训练集: {original_count} -> {filtered_count} "
                f"(过滤掉 {original_count - filtered_count} 个样本)"
            )
    else:
        # ==================== 不需要 NTE 分数的路由策略 ====================
        logger.info("当前路由策略不需要 NTE 分数，跳过加载和过滤")

        # 但仍然需要过滤训练集：保留包含所有必需教师响应的样本
        required_teachers = config["method_settings"].get("required_teachers", None)
        train_split = distill_bundle["splits"].get("train")

        if required_teachers and train_split:
            logger.info(f"过滤训练集: 要求包含所有教师的响应 {required_teachers}")

            original_count = len(train_split)

            filtered_samples = []
            for sample in train_split:
                responses = sample.get("responses", {})

                # 检查是否包含所有教师的响应
                has_all_teachers = all(
                    teacher_name in responses
                    for teacher_name in required_teachers
                )

                if has_all_teachers:
                    filtered_samples.append(sample)

            train_split.samples = filtered_samples
            filtered_count = len(filtered_samples)

            logger.info(
                f"训练集: {original_count} -> {filtered_count} "
                f"(过滤掉 {original_count - filtered_count} 个样本)"
            )

    logger.info("数据准备完成!")

    # ==================== 加载路由器（可选）====================
    router_checkpoint = config["method_settings"].get("router_checkpoint", None)
    router_model = None

    if router_checkpoint and os.path.exists(router_checkpoint):
        logger.info(f"加载路由器模型: {router_checkpoint}")
        # TODO: 实际加载路由器模型的代码
        logger.info("路由器加载完成!")
    else:
        logger.warning(f"路由器检查点不存在或未配置: {router_checkpoint}")
        logger.warning("将使用配置的路由策略")

    # ==================== 手动划分训练集和验证集 ====================
    logger.info("手动划分训练集和验证集...")

    train_split = distill_bundle["splits"]["train"]
    train_samples = train_split.samples

    # 划分比例: 80% train, 20% val
    total = len(train_samples)
    train_size = int(total * 0.8)

    # 按顺序划分（不打乱）
    train_data = train_samples[:train_size]
    val_data = train_samples[train_size:]

    logger.info(f"原始训练数据: {total} 样本")
    logger.info(f"划分后 - 训练集: {len(train_data)} 样本")
    logger.info(f"划分后 - 验证集: {len(val_data)} 样本")

    # 重新打包为 dataset
    from engine.datas.base.distill import DistillDataset
    distill_bundle["splits"]["train"] = DistillDataset(train_data)
    distill_bundle["splits"]["val"] = DistillDataset(val_data)

    # 合并两个数据集的 splits：训练集来自蒸馏，测试集来自 QA
    merged_bundle = {
        "splits": {
            "train": distill_bundle["splits"]["train"],
            "test": qa_bundle["splits"]["test"]
        },
        "meta": {
            "train_source": distill_bundle["meta"],
            "test_source": qa_bundle["meta"]
        },
        "judge": qa_bundle.get("judge")  # 使用 QA 的 judge 函数
    }

    logger.info("数据集合并完成:")
    logger.info(f"  训练集: {len(merged_bundle['splits']['train'])} 样本（来自蒸馏数据集）")
    logger.info(f"  测试集: {len(merged_bundle['splits']['test'])} 样本（来自 GSM8K）")

    return {
        "dataset_bundle": merged_bundle,
        "router_model": router_model
    }


# ==================== 训练函数 ====================

def run_fn(config: Dict, cache: Dict[str, Any]) -> Dict[str, Any]:
    """执行蒸馏训练

    使用 preload_fn 中合并好的 dataset_bundle:
    - splits["train"]: 来自蒸馏数据集（OpenR1-Math）
    - splits["test"]: 来自 QA 数据集（GSM8K）
    - judge: 来自 QA 数据集（用于测试集评估）
    """
    logger = config["logger"]
    dataset_bundle = cache["dataset_bundle"]
    router_model = cache.get("router_model", None)
    device = config.get("global_settings", {}).get("device")

    # 从配置中提取参数
    method_cfg = config["method_settings"]
    required_teachers = method_cfg["required_teachers"]

    logger.info(f"教师列表: {required_teachers}")
    logger.info(f"教师数量: {len(required_teachers)}")

    # 加载学生模型（LLM Backbone）
    logger.info("加载学生模型（Qwen2.5-0.5B-Instruct）...")
    student_backbone = load_backbone(config)
    student_model = student_backbone.model
    tokenizer = student_backbone.tokenizer

    logger.info(f"学生模型参数量: {sum(p.numel() for p in student_model.parameters()) / 1e9:.2f}B")

    # 数据集已在 preload_fn 中完成合并和组织
    logger.info(f"训练集大小: {len(dataset_bundle['splits']['train'])}")
    logger.info(f"测试集大小 (GSM8K): {len(dataset_bundle['splits']['test'])}")

    # 创建 Trainer
    logger.info("创建 Trainer...")
    trainer = load_trainer(config, dataset_bundle)

    # 注册模型
    trainer.register_model("student", student_model)
    trainer.add_param_group("student", list(student_model.parameters()))
    trainer.setup_optimizers()

    # 注册额外信息
    trainer.register_model("tokenizer", tokenizer)
    trainer.register_model("required_teachers", required_teachers)
    trainer.register_model("router_model", router_model)
    trainer.register_model("dataset_bundle", dataset_bundle)  # 传递 judge 函数

    # 注册KNN和统计信息（可选，用于knn_stats策略）
    trainer.register_model("knn_index", None)
    trainer.register_model("teacher_statistics", None)

    # 注册训练和评估步骤
    trainer.register_train_step(train_step)
    trainer.register_eval_step(eval_step)

    # 执行训练
    logger.info("开始蒸馏训练...")
    result = trainer.run()

    return result


# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: 知识蒸馏（使用 GSM8K 测试集）")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/step3_distill_with_gsm8k.yaml",
        help="配置文件路径"
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(override_file=args.config)
    logger = config["logger"]

    logger.info("=" * 60)
    logger.info("Step 3: 知识蒸馏（使用 GSM8K 测试集）")
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
    logger.info("蒸馏训练完成!")
    logger.info(f"总任务数: {summary['total_tasks']}")
    logger.info(f"已完成: {summary['completed']}")
    logger.info(f"失败: {summary['failed']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
