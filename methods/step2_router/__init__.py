"""Step2: 路由模型训练

训练一个 DeBERTa-v3-base 模型作为路由器，预测每个教师的 NTE 桶（分类模式）或 NTE 分数（回归模式）。
"""

from .models.router_regressor import RouterClassifier, RouterRegressor, RouterRegressorSingle
from .steps import (
    train_step,
    eval_step,
    train_step_regression,
    eval_step_regression,
    ExpandedDatasetWrapper
)
from .adaptive_bucketing import (
    adaptive_bucketing_per_teacher,
    quantile_bucketing,
    kmeans_bucketing,
    uniform_bucketing,
    compute_quantization_error,
    compare_bucketing_methods,
)

__all__ = [
    "RouterClassifier",  # 分类模式（桶化，推荐）
    "RouterRegressor",   # 回归模式（向后兼容，多头）
    "RouterRegressorSingle",  # 回归模式（单头，新架构）
    "train_step",
    "eval_step",
    "train_step_regression",  # 回归模式训练步骤
    "eval_step_regression",   # 回归模式评估步骤
    "ExpandedDatasetWrapper",  # 数据展开包装器
    "adaptive_bucketing_per_teacher",
    "quantile_bucketing",
    "kmeans_bucketing",
    "uniform_bucketing",
    "compute_quantization_error",
    "compare_bucketing_methods",
]
