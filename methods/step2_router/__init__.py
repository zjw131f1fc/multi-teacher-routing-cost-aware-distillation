"""Step2: 路由模型训练

训练一个 Qwen-1.5B-Instruct 模型作为路由器，预测每个教师的 NTE 分数。
"""

from .models.router_regressor import RouterRegressor
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
    "RouterRegressor",
    "train_step",
    "eval_step",
    "adaptive_bucketing_per_teacher",
    "quantile_bucketing",
    "kmeans_bucketing",
    "uniform_bucketing",
    "compute_quantization_error",
    "compare_bucketing_methods",
]
