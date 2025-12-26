"""自适应分桶策略（教师特定版本）

核心思想：
1. **每个教师独立分桶**：不同教师的 NTE 分数分布可能差异很大
2. **固定桶数量**：用户指定桶数量（例如 5 个桶）
3. **自适应分割点**：使用 K-Means 最小化量化误差，自动选择最优分割点

分桶策略：
1. quantile: 分位数法（样本均衡，但量化误差不一定最小）
2. kmeans: K-Means聚类（量化误差最小，但桶大小不均衡）
3. uniform: 均匀分割（等宽桶）
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.cluster import KMeans
import warnings


def compute_uniformity_score(bucket_distribution: List[int]) -> float:
    """计算样本分布均匀性分数（变异系数的倒数）

    变异系数（CV）= 标准差 / 均值
    均匀性分数 = 1 / (1 + CV)，范围 [0, 1]，越接近1越均匀

    参数:
        bucket_distribution: 每个桶的样本数列表

    返回:
        uniformity_score: 均匀性分数（0到1之间，1表示完全均匀）
    """
    counts = np.array(bucket_distribution)
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0

    mean_count = counts.mean()
    if mean_count == 0:
        return 0.0

    std_count = counts.std()
    cv = std_count / mean_count  # 变异系数

    # 转换为均匀性分数（越均匀，CV越小，分数越高）
    uniformity_score = 1.0 / (1.0 + cv)

    return float(uniformity_score)


def compute_quantization_error(scores: np.ndarray, bucket_ranges: List[float]) -> Dict[str, float]:
    """计算量化误差

    参数:
        scores: NTE分数数组
        bucket_ranges: 桶边界列表

    返回:
        {
            "mse": 均方误差,
            "mae": 平均绝对误差,
            "max_error": 最大误差
        }
    """
    # 计算桶中心
    centers = []
    all_bounds = [0.0] + list(bucket_ranges) + [1.0]
    for i in range(len(all_bounds) - 1):
        centers.append((all_bounds[i] + all_bounds[i + 1]) / 2)

    # 将每个分数映射到对应的桶中心
    quantized = np.zeros_like(scores)
    for i, score in enumerate(scores):
        for j, threshold in enumerate(bucket_ranges):
            if score < threshold:
                quantized[i] = centers[j]
                break
        else:
            quantized[i] = centers[-1]

    errors = scores - quantized
    return {
        "mse": float(np.mean(errors ** 2)),
        "mae": float(np.mean(np.abs(errors))),
        "max_error": float(np.max(np.abs(errors)))
    }


def uniform_bucketing(num_buckets: int) -> Tuple[List[float], List[float]]:
    """均匀分割（等宽桶）

    参数:
        num_buckets: 桶数量

    返回:
        (bucket_ranges, bucket_centers)
    """
    step = 1.0 / num_buckets
    bucket_ranges = [step * (i + 1) for i in range(num_buckets - 1)]

    # 计算桶中心
    all_bounds = [0.0] + bucket_ranges + [1.0]
    centers = [(all_bounds[i] + all_bounds[i + 1]) / 2 for i in range(len(all_bounds) - 1)]

    return bucket_ranges, centers


def quantile_bucketing(scores: np.ndarray, num_buckets: int) -> Tuple[List[float], List[float]]:
    """分位数分桶（样本均衡）

    参数:
        scores: NTE分数数组
        num_buckets: 桶数量

    返回:
        (bucket_ranges, bucket_centers)
    """
    percentiles = np.linspace(0, 100, num_buckets + 1)[1:-1]
    bucket_ranges = np.percentile(scores, percentiles).tolist()

    # 计算桶中心
    all_bounds = [0.0] + bucket_ranges + [1.0]
    centers = [(all_bounds[i] + all_bounds[i + 1]) / 2 for i in range(len(all_bounds) - 1)]

    return bucket_ranges, centers


def kmeans_bucketing(scores: np.ndarray, num_buckets: int) -> Tuple[List[float], List[float]]:
    """K-Means聚类分桶（最小化量化误差）

    参数:
        scores: NTE分数数组
        num_buckets: 桶数量

    返回:
        (bucket_ranges, bucket_centers)
    """
    # K-Means 聚类
    kmeans = KMeans(n_clusters=num_buckets, random_state=42, n_init=10)
    kmeans.fit(scores.reshape(-1, 1))

    # 获取聚类中心并排序
    centers = sorted(kmeans.cluster_centers_.flatten().tolist())

    # 计算桶边界（相邻中心的中点）
    bucket_ranges = []
    for i in range(len(centers) - 1):
        boundary = (centers[i] + centers[i + 1]) / 2
        bucket_ranges.append(boundary)

    return bucket_ranges, centers


def adaptive_bucketing_per_teacher(
    teacher_scores: Dict[str, np.ndarray],
    num_buckets: int,
    method: str = "kmeans",
    balance_weight: float = 0.0,
    logger=None
) -> Dict[str, Dict[str, Any]]:
    """为每个教师独立进行自适应分桶

    参数:
        teacher_scores: {teacher_name: scores_array} 每个教师的分数数组
        num_buckets: 桶数量（固定值）
        method: 分桶方法 ("kmeans", "quantile", "uniform")
        balance_weight: 均匀性权重（0到1之间）
            - 0: 只考虑量化误差（默认）
            - 1: 只考虑样本分布均匀性
            - 0.5: 两者权重相等
        logger: 日志记录器

    返回:
        {
            teacher_name: {
                "bucket_ranges": [...],
                "bucket_centers": [...],
                "quantization_errors": {...},
                "bucket_distribution": [...],  # 每个桶的样本数
                "uniformity_score": float,      # 均匀性分数
                "combined_score": float         # 综合得分（如果balance_weight > 0）
            }
        }
    """
    log = logger.info if logger else print

    log("="*80)
    log(f"自适应分桶（教师特定）- 桶数量={num_buckets}, 方法={method}, 均匀性权重={balance_weight}")
    log("="*80)

    results = {}

    for teacher_name, scores in teacher_scores.items():
        log(f"\n处理教师: {teacher_name}")
        log(f"  分数数量: {len(scores)}")
        log(f"  分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
        log(f"  分数均值: {scores.mean():.4f}, 标准差: {scores.std():.4f}")

        # 根据方法选择分桶策略
        if method == "kmeans":
            bucket_ranges, bucket_centers = kmeans_bucketing(scores, num_buckets)
        elif method == "quantile":
            bucket_ranges, bucket_centers = quantile_bucketing(scores, num_buckets)
        elif method == "uniform":
            bucket_ranges, bucket_centers = uniform_bucketing(num_buckets)
        else:
            raise ValueError(f"未知的分桶方法: {method}")

        # 计算量化误差
        errors = compute_quantization_error(scores, bucket_ranges)

        # 统计每个桶的样本分布
        all_bounds = [0.0] + bucket_ranges + [1.0]
        bucket_distribution = []
        log(f"\n  桶配置:")
        log(f"    桶边界: {[round(x, 4) for x in bucket_ranges]}")
        log(f"    桶中心: {[round(x, 4) for x in bucket_centers]}")
        log(f"    量化误差 - MSE: {errors['mse']:.4f}, MAE: {errors['mae']:.4f}")

        log(f"\n  桶分布:")
        for i in range(len(bucket_centers)):
            lower = all_bounds[i]
            upper = all_bounds[i + 1]
            mask = (scores >= lower) & (scores < upper if i < len(bucket_centers) - 1 else scores <= upper)
            count = mask.sum()
            percentage = count / len(scores) * 100
            bucket_distribution.append(int(count))

            bucket_scores = scores[mask]
            if len(bucket_scores) > 0:
                actual_mean = bucket_scores.mean()
                log(f"    Bucket {i}: [{lower:.2f}, {upper:.2f}) 中心={bucket_centers[i]:.2f}, "
                    f"实际均值={actual_mean:.2f}, 样本数={count:4d} ({percentage:5.1f}%)")
            else:
                log(f"    Bucket {i}: [{lower:.2f}, {upper:.2f}) 中心={bucket_centers[i]:.2f}, "
                    f"样本数=   0 (  0.0%)")

        # 计算均匀性分数
        uniformity_score = compute_uniformity_score(bucket_distribution)
        log(f"\n  均匀性分数: {uniformity_score:.4f} (1.0=完全均匀)")

        # 计算综合得分（归一化处理）
        # 归一化MSE: 使用MSE/(分数范围^2)作为归一化误差，值越小越好
        max_possible_mse = (1.0 / 2) ** 2  # 最大可能的MSE（所有分数都离中心最远）
        normalized_error = errors['mse'] / max_possible_mse  # 0到1之间，越小越好

        # 综合得分 = (1-balance_weight) * (1-normalized_error) + balance_weight * uniformity_score
        # 这样两者都是"越大越好"的指标
        combined_score = (1 - balance_weight) * (1 - normalized_error) + balance_weight * uniformity_score

        if balance_weight > 0:
            log(f"  综合得分: {combined_score:.4f} (balance_weight={balance_weight})")
            log(f"    - 量化准确度: {(1-normalized_error):.4f} (权重={1-balance_weight:.2f})")
            log(f"    - 均匀性: {uniformity_score:.4f} (权重={balance_weight:.2f})")

        results[teacher_name] = {
            "bucket_ranges": bucket_ranges,
            "bucket_centers": bucket_centers,
            "quantization_errors": errors,
            "bucket_distribution": bucket_distribution,
            "uniformity_score": uniformity_score,
            "combined_score": combined_score
        }

    log("="*80)

    return results


def compare_bucketing_methods(
    scores: np.ndarray,
    num_buckets: int,
    balance_weight: float = 0.0,
    logger=None
) -> Dict[str, Any]:
    """比较不同分桶方法的量化误差和均匀性

    参数:
        scores: NTE分数数组
        num_buckets: 桶数量
        balance_weight: 均匀性权重（0到1之间）
        logger: 日志记录器

    返回:
        各方法的比较结果
    """
    log = logger.info if logger else print

    log(f"\n分桶方法比较 (桶数量={num_buckets}, 均匀性权重={balance_weight}):")
    log("-" * 80)

    results = {}
    max_possible_mse = (1.0 / 2) ** 2

    # 辅助函数：计算桶分布
    def get_bucket_distribution(scores, bucket_ranges):
        all_bounds = [0.0] + bucket_ranges + [1.0]
        distribution = []
        for i in range(len(all_bounds) - 1):
            lower = all_bounds[i]
            upper = all_bounds[i + 1]
            mask = (scores >= lower) & (scores < upper if i < len(all_bounds) - 2 else scores <= upper)
            distribution.append(int(mask.sum()))
        return distribution

    # 1. K-Means
    k_ranges, k_centers = kmeans_bucketing(scores, num_buckets)
    k_errors = compute_quantization_error(scores, k_ranges)
    k_dist = get_bucket_distribution(scores, k_ranges)
    k_uniformity = compute_uniformity_score(k_dist)
    k_normalized_error = k_errors['mse'] / max_possible_mse
    k_combined = (1 - balance_weight) * (1 - k_normalized_error) + balance_weight * k_uniformity
    results["kmeans"] = {
        "ranges": k_ranges,
        "centers": k_centers,
        "errors": k_errors,
        "distribution": k_dist,
        "uniformity": k_uniformity,
        "combined_score": k_combined
    }
    log(f"K-Means  - MSE: {k_errors['mse']:.4f}, 均匀性: {k_uniformity:.4f}, 综合得分: {k_combined:.4f}")

    # 2. 分位数法
    q_ranges, q_centers = quantile_bucketing(scores, num_buckets)
    q_errors = compute_quantization_error(scores, q_ranges)
    q_dist = get_bucket_distribution(scores, q_ranges)
    q_uniformity = compute_uniformity_score(q_dist)
    q_normalized_error = q_errors['mse'] / max_possible_mse
    q_combined = (1 - balance_weight) * (1 - q_normalized_error) + balance_weight * q_uniformity
    results["quantile"] = {
        "ranges": q_ranges,
        "centers": q_centers,
        "errors": q_errors,
        "distribution": q_dist,
        "uniformity": q_uniformity,
        "combined_score": q_combined
    }
    log(f"Quantile - MSE: {q_errors['mse']:.4f}, 均匀性: {q_uniformity:.4f}, 综合得分: {q_combined:.4f}")

    # 3. 均匀分割
    u_ranges, u_centers = uniform_bucketing(num_buckets)
    u_errors = compute_quantization_error(scores, u_ranges)
    u_dist = get_bucket_distribution(scores, u_ranges)
    u_uniformity = compute_uniformity_score(u_dist)
    u_normalized_error = u_errors['mse'] / max_possible_mse
    u_combined = (1 - balance_weight) * (1 - u_normalized_error) + balance_weight * u_uniformity
    results["uniform"] = {
        "ranges": u_ranges,
        "centers": u_centers,
        "errors": u_errors,
        "distribution": u_dist,
        "uniformity": u_uniformity,
        "combined_score": u_combined
    }
    log(f"Uniform  - MSE: {u_errors['mse']:.4f}, 均匀性: {u_uniformity:.4f}, 综合得分: {u_combined:.4f}")

    # 推荐
    if balance_weight > 0:
        best_method = max(results.keys(), key=lambda x: results[x]["combined_score"])
        log(f"\n推荐使用: {best_method} (综合得分最高)")
    else:
        best_method = min(results.keys(), key=lambda x: results[x]["errors"]["mse"])
        log(f"\n推荐使用: {best_method} (MSE最小)")

    return results




if __name__ == "__main__":
    # 测试教师特定分桶
    np.random.seed(42)

    # 模拟两个教师的NTE分数分布（分布差异很大）
    teacher_scores = {
        "deepseek-r1": np.clip(np.random.beta(5, 2, 1000), 0, 1),  # 偏高分，[0, 1]范围
        "qwen2.5-math-7b-instruct": np.clip(np.random.beta(2, 5, 1000), 0, 1),  # 偏低分，[0, 1]范围
    }

    print("=" * 80)
    print("测试教师特定自适应分桶")
    print("=" * 80)

    # 测试1: K-Means 方法
    print("\n方法1: K-Means（量化误差最小）")
    results = adaptive_bucketing_per_teacher(teacher_scores, num_buckets=5, method="kmeans")

    # 测试2: 分位数方法
    print("\n\n方法2: Quantile（样本均衡）")
    results = adaptive_bucketing_per_teacher(teacher_scores, num_buckets=5, method="quantile")

    # 测试3: 均匀分割
    print("\n\n方法3: Uniform（等宽桶）")
    results = adaptive_bucketing_per_teacher(teacher_scores, num_buckets=5, method="uniform")

    # 测试4: 比较单个教师的不同方法
    print("\n\n" + "=" * 80)
    print("比较单个教师的不同分桶方法")
    print("=" * 80)
    compare_bucketing_methods(teacher_scores["deepseek-r1"], num_buckets=5)

