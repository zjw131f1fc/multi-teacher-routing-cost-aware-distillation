"""Multi-Teacher Routing with Cost-Aware Distillation - 主训练脚本

使用新的Engine架构实现多教师路由感知蒸馏。
"""

import torch
from typing import Dict, Any

from engine.configs.loader import load_config

# ===================== Manager函数 =====================

def preload_fn(config: Dict) -> Dict[str, Any]:
    """预加载重量级资源"""
    import json
    import os
    from engine.datas.loader import load_dataset

    logger = config["logger"]
    logger.info("预加载Dataset...")

    # 只加载蒸馏数据集
    dataset_bundle = load_dataset(config)

    # 根据数据集名称自动确定文件路径
    dataset_name = config["dataset_settings"]["name"]
    if dataset_name.startswith("distill-"):
        dataset_id = dataset_name[len("distill-"):]  # e.g., "openr1-math"
    else:
        dataset_id = dataset_name

    # ==================== 合并补充数据 ====================
    logger.info("合并 method_cache 中的补充数据...")

    cache_file = f"method_cache/{dataset_id}.json"
    if not os.path.exists(cache_file):
        logger.warning(f"补充数据文件不存在: {cache_file}")
    else:
        logger.info(f"加载补充数据: {cache_file}")

        # 逐行读取 JSON（避免一次性加载整个文件）
        with open(cache_file, 'r') as f:
            content = f.read()
            supplement_data = json.loads(content)

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
                            if teacher_name not in sample["responses"]:
                                sample["responses"][teacher_name] = teacher_data
                                merged_count += 1

            logger.info(f"Split '{split_name}': 合并了 {merged_count} 条教师响应")

    logger.info("数据合并完成!")

    # ==================== 合并学生分数 ====================
    logger.info("合并学生模型的 NTE 分数...")

    # 使用统一的命名规则：step1_collect_scores_{dataset_id}.json
    score_file = f"method_cache/step1_collect_scores_{dataset_id}.json"
    if not os.path.exists(score_file):
        logger.warning(f"学生分数文件不存在: {score_file}")
        logger.warning("请先运行 step1_collect_scores.py 收集学生分数")
    else:
        logger.info(f"加载学生分数: {score_file}")

        with open(score_file, 'r', encoding='utf-8') as f:
            student_scores = json.load(f)

        logger.info(f"加载了 {len(student_scores)} 条学生分数")

        # 创建分数索引（按 instruction 建立映射）
        score_index = {}
        for item in student_scores:
            instruction = item.get("instruction", "")
            if instruction:
                score_index[instruction] = item.get("teacher_scores", {})

        logger.info(f"建立了 {len(score_index)} 条分数索引")

        # 合并分数：遍历 dataset_bundle 中的每个 split
        for split_name, split_dataset in dataset_bundle["splits"].items():
            merged_count = 0
            for i in range(len(split_dataset)):
                sample = split_dataset[i]
                instruction = sample.get("instruction", "")

                # 如果在分数数据中找到匹配的 instruction
                if instruction in score_index:
                    teacher_scores = score_index[instruction]

                    # 为每个教师添加 NTE 分数
                    for teacher_name, scores_dict in teacher_scores.items():
                        if teacher_name in sample["responses"]:
                            # 将分数信息添加到教师响应中
                            sample["responses"][teacher_name]["nte_scores"] = scores_dict
                            merged_count += 1

            logger.info(f"Split '{split_name}': 合并了 {merged_count} 条 NTE 分数")

    logger.info("学生分数合并完成!")

    # ==================== 过滤数据 ====================
    # 只保留包含所有必需教师响应的样本
    required_teachers = config.get("method_settings", {}).get("required_teachers", None)

    if required_teachers:
        logger.info(f"过滤数据: 要求包含所有这些教师 {required_teachers}")

        for split_name, split_dataset in dataset_bundle["splits"].items():
            original_count = len(split_dataset)

            # 过滤样本：只保留包含所有必需教师的样本
            filtered_samples = []
            for sample in split_dataset:
                responses = sample.get("responses", {})
                # 检查是否包含所有必需的教师
                has_all_teachers = all(
                    teacher_name in responses
                    for teacher_name in required_teachers
                )
                if has_all_teachers:
                    filtered_samples.append(sample)

            # 替换数据集
            split_dataset.samples = filtered_samples
            filtered_count = len(filtered_samples)

            logger.info(
                f"Split '{split_name}': {original_count} -> {filtered_count} "
                f"(过滤掉 {original_count - filtered_count} 个样本)"
            )
    else:
        logger.info("未配置 required_teachers，跳过过滤")

    return {
        "dataset_bundle": dataset_bundle
    }


def run_fn(config: Dict, cache: Dict[str, Any]) -> Dict[str, Any]:
    """执行训练"""
    from engine.trainers.loader import load_trainer
    from methods.distill_router import (
        Router,
        Student,
        train_step,
        eval_step
    )

    logger = config["logger"]
    dataset_bundle = cache["dataset_bundle"]
    device = config.get("global_settings", {}).get("device")

    # 将dataset_bundle放入config供eval_step使用
    config["_dataset_bundle"] = dataset_bundle

    logger.info("创建Trainer...")
    trainer = load_trainer(config, dataset_bundle)

    # ==================== 创建模型 ====================

    # 1. Router
    logger.info("创建Router...")
    router = Router(config=config).to(device=device)

    # 2. Student (暂时不需要backbone，先创建一个简单的模型)
    logger.info("创建Student...")
    student = Student(
        config=config,
        backbone=None  # TODO: 根据需要添加backbone
    ).to(device=device)

    # ==================== 注册模型 ====================

    trainer.register_model("router", router)
    trainer.register_model("student", student)

    # ==================== 添加参数组 ====================

    trainer.add_param_group("router", list(router.parameters()))
    trainer.add_param_group("student", list(student.parameters()))

    # ==================== 创建优化器 ====================

    trainer.setup_optimizers()

    # ==================== 注册训练和评估函数 ====================

    trainer.register_train_step(train_step)
    trainer.register_eval_step(eval_step)

    # ==================== 执行训练 ====================

    logger.info("开始训练...")
    logger.info(f"Number of teachers: {config['method_settings']['num_teachers']}")
    logger.info(f"Router hidden dim: {config['method_settings']['router_d_hidden']}")

    result = trainer.run()

    return result


# ===================== 主函数 =====================

def main():
    """主函数"""
    # 加载配置
    config = load_config(override_file="configs/distill_router.yaml")
    logger = config["logger"]

    from engine.managers.loader import load_manager

    logger.info("=" * 60)
    logger.info("Multi-Teacher Routing with Cost-Aware Distillation")
    logger.info("=" * 60)

    # 创建Manager
    manager = load_manager(
        config=config,
        preload_fn=preload_fn,
        run_fn=run_fn,
        task_generator_fn=None,  # 单任务模式
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
