# Step 1: 收集学生模型的 NTE 分数

## 功能

收集学生模型（LLM Backbone）对每个教师响应的似然概率，并计算 NTE (Normalized Teaching Efficacy) 分数。

## NTE 分数公式

```
V_soft(y) = ε + (1-ε) × V(y)  (将 [0,1] 映射到 [ε, 1])
NTE(x,y) = V_soft(y)^α × M_prox(p)^β
```

使用 **Soft Quality + 指数平滑组合**，解决离散质量分数（0/1）导致的区分度问题。

其中：
- `V(y)`: 教师响应的质量分数（从 `rewards[reward_key]` 中提取，通常为 0 或 1）
- `V_soft(y)`: 平滑后的质量分数（将 [0,1] 映射到 [ε, 1]）
- `M_prox(p)`: 近端可学习性，基于学生模型的似然概率 p
  - `M_prox = (p^α_prox * (1-p)^β_prox) / Z`
  - 建议参数: `α_prox=2.0, β_prox=0.5`
- `α`: 质量分数的指数（默认 0.5）
- `β`: 可学习性分数的指数（默认 0.5）
- `ε`: 质量分数下限（默认 0.1）

**参数说明**：
- `α<β` (推荐 α=0.3, β=0.7): 更重视可学习性，适合离散质量分数
- `α=β=0.5`: 两者平衡，类似几何平均
- `α>β`: 更重视质量
- `ε=0.1`: 低质量响应保留 10% 权重，仍能通过 proximity 区分
- `ε=0.0`: 退化为原始方案，quality=0 时 NTE=0

**示例** (α=0.3, β=0.7, ε=0.1):
- Quality=0, Proximity=0.9 → V_soft=0.1 → NTE ≈ 0.478
- Quality=0, Proximity=0.5 → V_soft=0.1 → NTE ≈ 0.314
- Quality=0, Proximity=0.1 → V_soft=0.1 → NTE ≈ 0.158
- Quality=1, Proximity=0.9 → V_soft=1.0 → NTE ≈ 0.933
- Quality=1, Proximity=0.1 → V_soft=1.0 → NTE ≈ 0.200

**关键特性**：
- ✅ 即使 quality=0，proximity 差异仍能显著体现 (0.478 vs 0.158)
- ✅ Proximity 主导：低质量+高可学习性 (0.478) > 高质量+低可学习性 (0.200)

## 使用方法

### 1. 配置文件

编辑 `configs/step1_collect_scores.yaml`：

```yaml
dataset_settings:
  type: "distill"
  name: "distill-openr1-math"  # 数据集名称

method_settings:
  # Study name (用于日志标识，不影响输出文件路径)
  study_name: "step1_collect_scores"

  # 必需的教师列表
  required_teachers:
    - "deepseek-r1"
    - "qwen2.5-math-7b-instruct"

  # NTE Score 参数（M_prox 的 Beta 分布参数）
  nte_alpha: 2.0  # Beta 分布参数 α_prox
  nte_beta: 0.5   # Beta 分布参数 β_prox

  # NTE 指数平滑参数: V_soft = ε + (1-ε)×V, NTE = V_soft^α × M_prox^β
  nte_quality_power: 0.3     # 质量分数的指数 α (推荐 0.3, 更重视可学习性)
  nte_proximity_power: 0.7   # 可学习性分数的指数 β (推荐 0.7, 主导作用)
  nte_quality_floor: 0.1     # 质量分数下限 ε (低质量响应保留 10% 权重)

  # 质量分数的 reward key
  reward_key: "math_verify"  # 从 rewards 中提取哪个 key 作为质量分数

  # 处理控制
  max_samples: null  # null=全部, 设置数字可限制处理数量（调试用）
```

### 2. 运行脚本

```bash
# 使用默认配置
python step1_collect_scores.py

# 或指定配置文件
python step1_collect_scores.py --config configs/step1_collect_scores.yaml
```

### 3. 输出文件

根据数据集名称自动生成文件路径：
- 数据集: `distill-openr1-math`
- **输出文件**: `method_cache/distill-openr1-math/step1_collect_scores.json` (固定名称)
- **补充数据**: `method_cache/distill-openr1-math/supplement.json`

**注意**: `study_name` 仅用于日志标识，不影响文件路径。

文件结构：
```
method_cache/
└── distill-openr1-math/              # 数据集目录
    ├── step1_collect_scores.json     # NTE 分数 (固定名称)
    └── supplement.json                # 补充数据（可选）
```

## 输出格式

```json
[
  {
    "instruction": "问题描述...",
    "teacher_scores": {
      "teacher-name": {
        "likelihood": 0.45,      // 学生模型的似然概率
        "proximity": 0.78,       // 近端可学习性分数
        "quality": 0.85,         // 教师响应质量（从 reward_key 提取）
        "nte_score": 0.663       // 最终 NTE 分数
      }
    }
  }
]
```

## 调试模式

如果要快速测试（只处理前 10 个样本）：

```yaml
method_settings:
  max_samples: 10
```

## 后续步骤

收集完分数后，运行主训练脚本时会自动加载这些分数：

```bash
python main_distill_router.py
```

preload_fn 会自动根据数据集名称找到对应的分数文件并合并到数据集中：
- 文件路径: `method_cache/{dataset_name}/step1_collect_scores.json`
- 例如: `method_cache/distill-openr1-math/step1_collect_scores.json`
