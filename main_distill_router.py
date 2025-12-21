"""Multi-Teacher Routing with Cost-Aware Distillation - 主训练脚本

使用新的Engine架构实现多教师路由感知蒸馏。
"""

import torch
from typing import Dict, Any

from engine.configs.loader import load_config

# ===================== Manager函数 =====================

def preload_fn(config: Dict) -> Dict[str, Any]:
    """预加载重量级资源"""
    from engine.datas.loader import load_dataset

    logger = config["logger"]
    logger.info("预加载Dataset...")

    # 只加载蒸馏数据集
    dataset_bundle = load_dataset(config)

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
