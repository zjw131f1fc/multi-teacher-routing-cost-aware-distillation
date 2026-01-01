"""路由函数：选择最优教师

支持多种路由策略：
1. 基于路由模型（Router Model）
2. 基于KNN+统计学（KNN + Statistics）
3. 固定使用最强教师（Best Teacher Baseline）
"""

import random
from typing import Dict, Any, Optional


def route_by_model(
    instruction: str,
    required_teachers: list,
    router_model: Any,
    epoch: int,
    sample_data: Optional[Dict] = None
) -> str:
    """基于路由模型的路由策略

    使用训练好的路由器模型预测每个教师的NTE分数，选择最优教师。

    参数:
        instruction: 问题文本
        required_teachers: 可用的教师列表
        router_model: 训练好的路由器模型
        epoch: 当前训练轮次（可用于动态调整策略）
        sample_data: 完整的样本数据（可选，包含NTE分数等信息）

    返回:
        teacher_name: 选择的教师名称

    TODO:
        - [ ] 1. 为每个教师构建路由器输入 prompt
        - [ ] 2. 使用路由器模型预测每个教师的NTE分数
        - [ ] 3. 根据预测分数选择最优教师
        - [ ] 4. 支持多种选择策略：
        -       - 贪心策略：始终选择分数最高的教师
        -       - 采样策略：根据分数分布进行采样（探索性更强）
        -       - 温度采样：使用softmax温度控制探索程度
        - [ ] 5. 支持早期探索策略：前几个epoch使用更高的探索率

    示例实现：
        ```python
        # 1. 为每个教师预测NTE分数
        teacher_scores = {}
        for teacher_name in required_teachers:
            prompt = build_router_prompt(instruction, teacher_name)
            score = router_model.predict(prompt)  # 预测NTE分数
            teacher_scores[teacher_name] = score

        # 2. 贪心策略：选择分数最高的教师
        best_teacher = max(teacher_scores, key=teacher_scores.get)
        return best_teacher

        # 或者：温度采样策略（可选）
        # import torch
        # import torch.nn.functional as F
        # scores_tensor = torch.tensor(list(teacher_scores.values()))
        # temperature = 1.0 if epoch < 1 else 0.5  # 早期高温度，后期低温度
        # probs = F.softmax(scores_tensor / temperature, dim=0)
        # selected_idx = torch.multinomial(probs, 1).item()
        # return required_teachers[selected_idx]
        ```
    """
    # TODO: 实现基于路由模型的路由策略
    # 当前回退：随机选择
    return random.choice(required_teachers)


def route_by_best_teacher(
    instruction: str,
    required_teachers: list,
    best_teacher_name: str,
    epoch: int,
    sample_data: Optional[Dict] = None
) -> str:
    """固定使用最强教师的路由策略（基线策略）

    始终选择预先指定的最强教师，不考虑问题内容。
    这是一个简单但强大的基线策略，用于对比其他复杂路由策略的效果。

    参数:
        instruction: 问题文本（该策略不使用）
        required_teachers: 可用的教师列表
        best_teacher_name: 预先指定的最强教师名称
        epoch: 当前训练轮次（该策略不使用）
        sample_data: 完整的样本数据（可选，该策略不使用）

    返回:
        teacher_name: 固定返回最强教师名称

    使用场景:
        - 作为基线策略，评估复杂路由策略的必要性
        - 当某个教师在全局上明显优于其他教师时
        - 简单快速，无需额外计算

    示例配置:
        ```yaml
        method_settings:
          routing_strategy: "best_teacher"
          best_teacher_name: "deepseek-r1"  # 指定最强教师
        ```
    """
    # 验证最强教师是否在可用教师列表中
    if best_teacher_name not in required_teachers:
        raise ValueError(
            f"指定的最强教师 '{best_teacher_name}' 不在可用教师列表中: {required_teachers}"
        )

    # 始终返回最强教师
    return best_teacher_name


def route_by_knn_stats(
    instruction: str,
    required_teachers: list,
    epoch: int,
    sample_data: Optional[Dict] = None,
    knn_index: Optional[Any] = None,
    teacher_statistics: Optional[Dict] = None
) -> str:
    """基于KNN+统计学的路由策略

    使用KNN查找相似问题，结合历史统计信息选择最优教师。

    参数:
        instruction: 问题文本
        required_teachers: 可用的教师列表
        epoch: 当前训练轮次（可用于动态调整策略）
        sample_data: 完整的样本数据（可选，包含NTE分数等信息）
        knn_index: KNN索引（用于快速检索相似样本）
        teacher_statistics: 教师统计信息（历史性能数据）

    返回:
        teacher_name: 选择的教师名称

    TODO:
        - [ ] 1. 将问题编码为向量（使用预训练模型如BERT、Sentence-BERT等）
        - [ ] 2. 使用KNN查找K个最相似的历史问题
        - [ ] 3. 统计这K个问题中每个教师的表现（NTE分数）
        - [ ] 4. 结合全局统计信息（如教师整体正确率、平均NTE分数等）
        - [ ] 5. 综合局部（KNN）和全局（统计学）信息选择教师
        - [ ] 6. 支持动态权重：随着训练进行，逐渐增加KNN权重，减少全局统计权重
        - [ ] 7. 冷启动处理：训练初期优先使用全局统计，后期优先使用KNN

    示例实现：
        ```python
        # 1. 编码问题为向量
        # from sentence_transformers import SentenceTransformer
        # encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # query_embedding = encoder.encode(instruction)

        # 2. KNN检索（如果有索引）
        # if knn_index is not None:
        #     k = 10  # 检索10个最近邻
        #     distances, indices = knn_index.search(query_embedding, k)
        #
        #     # 3. 统计KNN结果中每个教师的表现
        #     teacher_knn_scores = {t: [] for t in required_teachers}
        #     for idx in indices[0]:
        #         neighbor = historical_data[idx]
        #         for teacher in required_teachers:
        #             if teacher in neighbor['teacher_scores']:
        #                 teacher_knn_scores[teacher].append(
        #                     neighbor['teacher_scores'][teacher]
        #                 )
        #
        #     # 计算每个教师的平均KNN分数
        #     teacher_knn_avg = {
        #         t: np.mean(scores) if scores else 0.0
        #         for t, scores in teacher_knn_scores.items()
        #     }
        # else:
        #     teacher_knn_avg = {t: 0.0 for t in required_teachers}

        # 4. 结合全局统计信息
        # if teacher_statistics is not None:
        #     # 动态权重：早期更依赖全局统计，后期更依赖KNN
        #     knn_weight = min(0.8, epoch * 0.2)  # 逐渐增加到0.8
        #     global_weight = 1.0 - knn_weight
        #
        #     final_scores = {}
        #     for teacher in required_teachers:
        #         knn_score = teacher_knn_avg[teacher]
        #         global_score = teacher_statistics.get(teacher, {}).get('avg_nte', 0.5)
        #         final_scores[teacher] = (
        #             knn_weight * knn_score +
        #             global_weight * global_score
        #         )
        #
        #     # 选择分数最高的教师
        #     best_teacher = max(final_scores, key=final_scores.get)
        #     return best_teacher

        # 5. 冷启动回退
        # return random.choice(required_teachers)
        ```
    """
    # TODO: 实现基于KNN+统计学的路由策略
    # 当前回退：随机选择
    return random.choice(required_teachers)


def route_to_teacher(
    instruction: str,
    required_teachers: list,
    epoch: int,
    strategy: str = "random",
    router_model: Optional[Any] = None,
    sample_data: Optional[Dict] = None,
    knn_index: Optional[Any] = None,
    teacher_statistics: Optional[Dict] = None,
    best_teacher_name: Optional[str] = None
) -> str:
    """路由函数：为给定问题选择最优教师

    参数:
        instruction: 问题文本
        required_teachers: 可用的教师列表
        epoch: 当前训练轮次
        strategy: 路由策略，可选值:
            - "random": 随机选择（默认）
            - "model": 基于路由模型
            - "knn_stats": 基于KNN+统计学
            - "best_teacher": 固定使用最强教师（基线策略）
        router_model: 训练好的路由器模型（strategy="model"时需要）
        sample_data: 完整的样本数据（可选）
        knn_index: KNN索引（strategy="knn_stats"时可选）
        teacher_statistics: 教师统计信息（strategy="knn_stats"时可选）
        best_teacher_name: 最强教师名称（strategy="best_teacher"时需要）

    返回:
        teacher_name: 选择的教师名称
    """
    if strategy == "model":
        if router_model is None:
            raise ValueError("strategy='model' 需要提供 router_model 参数")
        return route_by_model(
            instruction=instruction,
            required_teachers=required_teachers,
            router_model=router_model,
            epoch=epoch,
            sample_data=sample_data
        )
    elif strategy == "knn_stats":
        return route_by_knn_stats(
            instruction=instruction,
            required_teachers=required_teachers,
            epoch=epoch,
            sample_data=sample_data,
            knn_index=knn_index,
            teacher_statistics=teacher_statistics
        )
    elif strategy == "best_teacher":
        if best_teacher_name is None:
            raise ValueError("strategy='best_teacher' 需要提供 best_teacher_name 参数")
        return route_by_best_teacher(
            instruction=instruction,
            required_teachers=required_teachers,
            best_teacher_name=best_teacher_name,
            epoch=epoch,
            sample_data=sample_data
        )
    else:  # "random" or fallback
        return random.choice(required_teachers)


def build_router_prompt(instruction: str, teacher_name: str) -> str:
    """构建路由器的输入prompt（与step2保持一致）

    参数:
        instruction: 问题文本
        teacher_name: 教师模型名称

    返回:
        prompt: 构建好的prompt
    """
    return f"""Question: {instruction}

Teacher: {teacher_name}"""
