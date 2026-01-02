# 分数差预测模式 (Score Difference Prediction Mode)

## 概述

分数差预测模式是一种专为**双教师场景**（一强一弱）设计的训练模式。通过直接指定强弱教师名字，模型学习预测：

1. **强教师的绝对分数**（单值预测）
2. **弱教师与强教师的分数差**（相对预测）

## 核心思想

### 传统回归模式的挑战
- 需要同时预测多个教师的绝对分数
- 每个教师的分数范围可能差异很大
- 模型难以学习不同教师之间的相对关系

### 分数差模式的优势
- **降低任务难度**：将双目标预测转化为 1 个绝对预测 + 1 个相对预测
- **显式建模强弱关系**：直接学习强弱教师之间的差异
- **自动保证排序**：通过差值约束确保强教师分数 ≥ 弱教师分数
- **直观配置**：通过教师名字直接指定强弱，无需记忆索引

## 架构设计

```
输入: instruction (问题文本)
    ↓
LLM Backbone (Qwen-1.5B, 全量微调)
    ↓
Pooling (mean/last token)
    ↓
┌─────────────────────────────────┐
│  两个独立的 MLP Head:            │
│                                 │
│  1. Strong Teacher Head         │
│     hidden_dim → hidden_dim//4  │
│     → 1 (强教师的绝对分数)        │
│                                 │
│  2. Diff Head                   │
│     hidden_dim → hidden_dim//4  │
│     → 1 (弱教师与强教师的分数差)   │
└─────────────────────────────────┘
    ↓
重构两个教师的分数:
  - strong_score = sigmoid(strong_logit) * 10
  - diff = -sigmoid(diff_logit) * 10  # 范围 [-10, 0]
  - weak_score = strong_score + diff
    ↓
输出: [score_strong, score_weak] (根据教师列表顺序)
```

## 配置说明

### 在配置文件中启用分数差模式

编辑 `configs/step2_train_router.yaml`:

```yaml
method_settings:
  # 必需的教师列表（必须恰好2个教师）
  required_teachers:
    - "deepseek-r1"
    - "qwen2.5-math-7b-instruct"

  # 模式选择（三选一）
  use_bucketing: false       # 分类模式（桶化）
  use_score_diff: true       # 分数差预测模式 ← 启用此选项

  # 分数差模式参数：直接指定强教师名字
  strong_teacher_name: "deepseek-r1"  # 强教师的名字
                                       # 必须是 required_teachers 中的一个
                                       # 另一个教师自动成为弱教师
```

### 重要说明

- **双教师限制**：此模式**仅支持 2 个教师**，否则会报错
- **名字匹配**：`strong_teacher_name` 必须在 `required_teachers` 列表中
- **自动识别弱教师**：不需要指定弱教师，会自动推断

## 训练流程

### 1. 数据准备

数据格式与其他模式相同，要求每个样本包含**恰好2个教师**的 NTE 分数：

```python
{
    "instruction": "问题描述",
    "responses": {
        "deepseek-r1": {
            "nte_scores": {"nte_score": 8.5}  # 强教师
        },
        "qwen2.5-math-7b-instruct": {
            "nte_scores": {"nte_score": 6.2}  # 弱教师
        }
    }
}
```

### 2. 强弱教师识别

训练脚本会自动根据 `strong_teacher_name` 识别：

```python
# 例如：required_teachers = ["deepseek-r1", "qwen2.5-math-7b-instruct"]
# strong_teacher_name = "deepseek-r1"

# 自动计算：
# strong_teacher_idx = 0
# weak_teacher_idx = 1
# weak_teacher_name = "qwen2.5-math-7b-instruct"
```

### 3. 损失函数

分数差模式使用 **MSE 损失**（与回归模式相同）：

```python
# 模型输出已经重构为完整分数（按教师列表顺序）
pred_scores = [8.3, 6.1]      # [deepseek-r1, qwen2.5-math-7b-instruct]
target_scores = [8.5, 6.2]    # 真实分数

# MSE损失
loss = MSE(pred_scores, target_scores)
```

优势：
- 模型内部学习的是分数差，但损失基于完整分数计算
- 自动约束预测的相对关系（弱教师分数 ≤ 强教师分数）
- 训练稳定，收敛快速

### 4. 评估指标

评估指标与回归模式相同：

- **MSE** (Mean Squared Error): 平均平方误差
- **MAE** (Mean Absolute Error): 平均绝对误差
- **Top-1 Accuracy**: 预测的最佳教师是否正确（最重要！）
- **Pearson Correlation**: 预测分数与真实分数的相关系数

## 实现细节

### 模型前向传播（双教师版本）

```python
# 1. 预测强教师的分数
strong_logit = self.strong_head(pooled)  # [batch, 1]
strong_score = torch.sigmoid(strong_logit) * 10  # [batch, 1], 范围 [0, 10]

# 2. 预测差值（使用负sigmoid确保非正）
diff_logit = self.diff_head(pooled)  # [batch, 1]
diff = -(torch.sigmoid(diff_logit) * 10)  # [batch, 1], 范围 [-10, 0]

# 3. 重构弱教师的分数
weak_score = strong_score + diff  # [batch, 1]

# 4. 根据 strong_teacher_idx 拼接（按教师列表顺序）
if strong_teacher_idx == 0:
    # 强教师在第0位：[deepseek-r1, qwen2.5-math-7b-instruct]
    scores = torch.cat([strong_score, weak_score], dim=-1)  # [batch, 2]
else:
    # 强教师在第1位：[qwen2.5-math-7b-instruct, deepseek-r1]
    scores = torch.cat([weak_score, strong_score], dim=-1)  # [batch, 2]
```

### 差值约束机制

- 使用 `sigmoid` 将 diff_logit 约束到 [0, 1]
- 乘以 `score_scale` (默认 10) 得到 [0, 10]
- 取负得到 [-10, 0]
- 这确保了 `weak_score = strong_score + diff ≤ strong_score`

## 使用场景

### 适合的场景

1. **双教师知识蒸馏**：一个强教师 + 一个弱教师的场景
2. **教师能力明确**：两个教师之间存在明显的强弱关系
3. **相对关系重要**：需要准确捕捉强弱教师之间的差异
4. **排序稳定性**：希望保证强教师始终被识别为最优

### 不适合的场景

1. **多教师场景**：超过2个教师（此模式有硬性限制）
2. **教师能力接近**：两个教师表现相似，没有明显强弱
3. **动态排序**：最强教师在不同问题上会变化
4. **需要独立预测**：需要每个教师完全独立的评分

## 与其他模式的比较

| 特性 | 回归模式 | 分类模式 (桶化) | 分数差模式 |
|------|---------|----------------|-----------|
| 输出类型 | 连续分数 | 离散桶概率 | 连续分数 |
| 预测目标 | N 个绝对分数 | N × K 个桶概率 | 1 个绝对分数 + 1 个差值 |
| 教师数量 | 任意 | 任意 | **仅2个** |
| 任务难度 | 中等 | 高（需要分桶） | **最低**（分解任务） |
| 排序保证 | 无 | 无 | **有**（自动保证） |
| 相对关系 | 隐式学习 | 隐式学习 | **显式建模** |
| 配置方式 | 索引 | 索引 | **教师名字** |
| 参数量 | 中等 | 高（独立头） | 中等 |
| 适用场景 | 通用 | 分布不均 | **双教师强弱** |

## 训练命令

```bash
# 1. 确保配置文件正确设置
#    - use_score_diff: true
#    - strong_teacher_name: "deepseek-r1"  (或你的强教师名字)
#    - required_teachers 必须恰好包含2个教师

# 2. 运行训练
python step2_train_router.py
```

训练日志示例：
```
使用分数差预测模式（仅支持2个教师）
强教师: deepseek-r1 (索引 0)
弱教师: qwen2.5-math-7b-instruct (索引 1)
Pooling 策略: mean
```

## 总结

分数差预测模式是专为**双教师场景**优化的训练方案：

**核心优势**：
- ✅ **最低任务难度**（1个绝对 + 1个相对）
- ✅ **显式建模强弱关系**
- ✅ **自动排序约束**（强 ≥ 弱）
- ✅ **直观配置**（通过教师名字）
- ✅ **专为双教师优化**

**使用建议**：
- 明确知道哪个教师更强时使用
- 配合 Top-1 准确率作为主要评估指标
- 可以与回归模式对比实验，选择更优方案
- 适合一强一弱的典型双教师蒸馏场景
