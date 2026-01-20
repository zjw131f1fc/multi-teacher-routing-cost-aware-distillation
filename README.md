# RL-Based Cost-Aware Teacher Selection for LLM Distillation

基于强化学习的成本感知教师选择系统，用于LLM知识蒸馏中的合成数据生成。

## 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate sb3-env
```

### 2. 使用Mock数据测试

```python
from methods.rl_teacher_selection import (
    get_mock_data,
    Teacher,
    TeacherPool,
    TeacherSelectionEnv,
)
from stable_baselines3 import PPO

# 加载模拟数据
data = get_mock_data(num_instructions=1000, num_val_samples=500, device="cpu")

# 创建教师池
mock_teachers = data["teachers"]
teachers = [Teacher(t.name, t, cost=t.cost) for t in mock_teachers]
teacher_pool = TeacherPool(teachers)

# 创建环境
env = TeacherSelectionEnv(
    instructions=data["instructions"],
    student_model=data["student_model"],
    teacher_pool=teacher_pool,
    val_dataloader=data["val_loader"],
    config={
        "val_samples": 200,           # 验证集梯度采样数
        "lambda_cost": 0.05,          # 成本权重系数
        "student_update_freq": 100,   # 每N步训练student
        "student_train_steps": 10,    # 每次训练student的步数
    }
)

# 训练policy（使用SB3的PPO）
model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1)
model.learn(total_timesteps=10000)

# 保存policy
model.save("teacher_selection_policy")

# 获取生成的数据集
dataset_D = env.get_dataset()
print(f"Generated {len(dataset_D)} samples")
```

### 3. 多Epoch训练

系统支持多个epoch训练，第二个epoch开始，已经生成过的数据cost为0：

```python
# 第一个epoch：生成数据 + 训练policy
model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1)
model.learn(total_timesteps=len(data["instructions"]))  # 遍历一次所有instructions

# 第二个epoch：继续训练（已生成的数据cost=0）
env.reset()  # 重置环境
model.learn(total_timesteps=len(data["instructions"]))  # 再遍历一次

# 第三个epoch...
env.reset()
model.learn(total_timesteps=len(data["instructions"]))
```

### 4. 使用预加载的Cache

如果有之前实验生成的数据，可以预加载到cache中避免重复推理：

```python
# 加载预先生成的数据
preloaded_data = {
    "instruction_1": (input_ids_1, labels_1),
    "instruction_2": (input_ids_2, labels_2),
    # ...
}

# 加载到teacher pool
teacher_pool.load_preloaded_cache(teacher_idx=0, cache_data=preloaded_data)

# 注意：预加载的数据在实验逻辑上仍算"调用教师"，cost = teacher.cost
# 只有在本次运行中第二次访问时，cost才为0
```

### 5. 查看Cache统计

```python
# 查看所有教师的cache统计
stats = teacher_pool.get_cache_stats()
for teacher_name, teacher_stats in stats.items():
    print(f"{teacher_name}:")
    print(f"  Preloaded cache: {teacher_stats['preloaded_cache_size']}")
    print(f"  Runtime cache: {teacher_stats['runtime_cache_size']}")
    print(f"  Hit rate: {teacher_stats['hit_rate']:.1%}")
```

## 核心设计

### Cache机制

系统维护两个cache：

1. **preloaded_cache**：从文件加载的数据
   - 避免实际推理（节省计算）
   - 但在实验逻辑上仍算"调用教师"，cost = teacher.cost
   - 第二次访问时移动到runtime_cache，cost变为0

2. **runtime_cache**：本次运行中生成的数据
   - 第一次生成：cost = teacher.cost
   - 第二次访问：cost = 0（真正的复用）

### Reward函数

```
R = Cosine(g_sample, g_val) × ||g_sample|| - λ × Cost
```

- `g_sample`：合成样本的梯度
- `g_val`：验证集平均梯度方向（单位向量）
- `Cosine(·,·)`：梯度对齐度 ∈ [-1, 1]
- `||g_sample||`：梯度强度
- `Cost`：教师调用成本（runtime cache命中时为0）
- `λ`：成本权重系数

### Student训练

在RL训练过程中，每`student_update_freq`步：
1. 用累积的dataset_D训练student
2. 重新计算验证集梯度方向g_val

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `val_samples` | 200 | 验证集梯度采样数 |
| `lambda_cost` | 0.05 | 成本权重系数 |
| `student_update_freq` | 100 | 每N步训练student |
| `student_train_steps` | 10 | 每次训练student的步数 |

## 测试

详细的测试说明见 [TODO.md](TODO.md)

## 项目结构

```
methods/rl_teacher_selection/
├── __init__.py           # 模块导出
├── policy.py             # Policy network
├── mock_data.py          # 模拟数据生成
├── gradient.py           # 梯度计算工具
├── reward.py             # 奖励计算
├── teacher_pool.py       # 教师池管理（含cache）
└── sb3_env.py            # Gymnasium环境
```

## 依赖

- Python 3.11
- PyTorch 2.9.1+cu126
- Stable-Baselines3 2.7.1
- Gymnasium 1.2.3
- Transformers

## 参考文档

- 详细设计：[CLAUDE.md](CLAUDE.md)
- 实现细节：[doc.md](doc.md)
