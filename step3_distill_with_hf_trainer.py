"""Step 3: 知识蒸馏（使用 HuggingFace Trainer）

使用 HuggingFace Transformers Trainer 进行知识蒸馏训练：
- 训练数据来自蒸馏数据集（OpenR1-Math）
- 测试数据来自 QA 数据集（GSM8K）
- 使用路由策略选择最优教师
- 支持 DeepSpeed、FSDP、混合精度等高级特性

使用方法:
    python step3_distill_with_hf_trainer.py --config configs/step3_distill_hf_trainer.yaml
"""

import os
import json
from typing import Dict, Any

from engine.configs.loader import load_config
from engine.backbones.loader import load_backbone
from engine.datas.loader import load_dataset
from engine.trainers.loader import load_trainer
from methods.step3_distill.distillation_trainer import DistillationTrainer


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

    # ==================== 过滤训练集：要求包含所有必需教师 ====================
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

    # ==================== 根据长度过滤训练数据 ====================
    max_train_length = config["method_settings"].get("max_train_length", None)

    if max_train_length is not None and max_train_length > 0:
        logger.info(f"开始根据长度过滤训练数据（最大长度: {max_train_length}）...")

        train_split = distill_bundle["splits"].get("train")
        if train_split:
            original_count = len(train_split)
            filtered_samples = []

            for sample in train_split:
                instruction = sample.get("instruction", "")
                responses = sample.get("responses", {})

                # 找到最短的教师响应长度
                min_response_length = float('inf')
                has_valid_response = False

                for teacher_name, teacher_data in responses.items():
                    messages = teacher_data.get("messages", [])

                    # 提取教师的响应文本
                    teacher_response = ""
                    for msg in messages:
                        if msg.get("role") == "assistant":
                            teacher_response = msg.get("content", "")
                            break

                    if teacher_response:
                        has_valid_response = True
                        # 构建完整的训练文本
                        full_text = instruction + teacher_response
                        # 粗略估算：1 token ≈ 1 个字符
                        estimated_length = len(full_text) + 200  # 加上 template 开销
                        min_response_length = min(min_response_length, estimated_length)

                # 只保留长度在限制内且有有效响应的样本
                if has_valid_response and min_response_length <= max_train_length:
                    filtered_samples.append(sample)

            train_split.samples = filtered_samples
            filtered_count = len(filtered_samples)

            logger.info(
                f"训练集长度过滤: {original_count} -> {filtered_count} "
                f"(过滤掉 {original_count - filtered_count} 个超长样本)"
            )
        else:
            logger.warning("训练集为空，跳过长度过滤")
    else:
        logger.info("未设置 max_train_length，跳过训练数据长度过滤")

    # ==================== 手动划分训练集和验证集 ====================
    logger.info("手动划分训练集和验证集...")

    train_split = distill_bundle["splits"]["train"]
    train_samples = train_split.samples

    # 划分比例: 80% train, 20% val (可选，HF Trainer 支持验证集)
    # 这里我们直接使用全部训练数据，不划分验证集
    # 如果需要验证集，可以在这里添加划分逻辑

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
    }


# ==================== 训练函数 ====================

def run_fn(config: Dict, cache: Dict[str, Any]) -> Dict[str, Any]:
    """执行蒸馏训练（使用 HuggingFace Trainer）"""
    logger = config["logger"]
    dataset_bundle = cache["dataset_bundle"]
    device = config.get("global_settings", {}).get("device")

    # 从配置中提取参数
    method_cfg = config["method_settings"]
    required_teachers = method_cfg["required_teachers"]
    routing_strategy = method_cfg.get("routing_strategy", "random")
    best_teacher_name = method_cfg.get("best_teacher_name", None)
    max_seq_length = method_cfg.get("max_seq_length", 2048)

    logger.info(f"教师列表: {required_teachers}")
    logger.info(f"路由策略: {routing_strategy}")

    # 加载学生模型（LLM Backbone）
    logger.info("加载学生模型（Qwen2.5-0.5B-Instruct）...")
    student_backbone = load_backbone(config)
    student_model = student_backbone.model
    tokenizer = student_backbone.tokenizer

    logger.info(f"学生模型参数量: {sum(p.numel() for p in student_model.parameters()) / 1e9:.2f}B")

    # 数据集已在 preload_fn 中完成合并和组织
    logger.info(f"训练集大小: {len(dataset_bundle['splits']['train'])}")
    logger.info(f"测试集大小 (GSM8K): {len(dataset_bundle['splits']['test'])}")

    # 创建 HF Trainer Wrapper
    logger.info("创建 HuggingFace Trainer...")
    trainer_wrapper = load_trainer(config, dataset_bundle)

    # 构建 Trainer（使用新接口）
    trainer_wrapper.build_trainer(
        model=student_model,
        tokenizer=tokenizer,
        trainer_class=DistillationTrainer,  # 使用自定义 Trainer
        trainer_kwargs={
            # DistillationTrainer 的额外参数
            "required_teachers": required_teachers,
            "routing_strategy": routing_strategy,
            "best_teacher_name": best_teacher_name,
            "max_seq_length": max_seq_length,
            "router_model": None,
            "knn_index": None,
            "teacher_statistics": None,
            "dataset_bundle": dataset_bundle,
        }
    )

    # 执行训练
    logger.info("开始蒸馏训练...")
    result = trainer_wrapper.run()

    return result


# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: 知识蒸馏（使用 HuggingFace Trainer）")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/step3_distill_hf_trainer.yaml",
        help="配置文件路径"
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(override_file=args.config)
    logger = config["logger"]

    logger.info("=" * 60)
    logger.info("Step 3: 知识蒸馏（使用 HuggingFace Trainer）")
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
