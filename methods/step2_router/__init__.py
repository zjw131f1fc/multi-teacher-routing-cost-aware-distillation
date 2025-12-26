"""Step2: 路由模型训练

训练一个 DeBERTa-v3-base 模型作为路由器，预测每个教师的 NTE 桶（分类模式）或 NTE 分数（回归模式）。
"""

from .models.router_regressor import RouterClassifier, RouterRegressor
from .steps import train_step, eval_step
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
    "RouterRegressor",   # 回归模式（向后兼容）
    "train_step",
    "eval_step",
    "adaptive_bucketing_per_teacher",
    "quantile_bucketing",
    "kmeans_bucketing",
    "uniform_bucketing",
    "compute_quantization_error",
    "compare_bucketing_methods",
]
