# 测试待办事项

## 需要测试的模块

### 1. PolicyNetwork (`methods/rl_teacher_selection/policy.py`)

**使用示例**:
```python
from transformers import AutoModel, AutoTokenizer
from methods.rl_teacher_selection import PolicyNetwork

# 加载 encoder 和 tokenizer
encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# 创建 policy network
num_actions = 4  # 3 teachers + 1 reuse
policy = PolicyNetwork(
    encoder=encoder,
    tokenizer=tokenizer,
    num_actions=num_actions,
    hidden_dims=[512, 256],
    freeze_encoder=True,
)

# 前向传播
instruction = "Solve: 2x + 5 = 13"
probs = policy(instruction)  # shape: (1, num_actions)

# 采样动作
actions, log_probs = policy.sample_action(instruction)

# 批量处理
instructions = ["Q1", "Q2", "Q3"]
probs = policy(instructions)  # shape: (3, num_actions)
```

**注意**:
- 第一次运行会下载 DeBERTa-v3-base 模型（~500MB）
- 确保在 `sb3-env` 环境中运行

---

### 2. Gradient Computation (`methods/rl_teacher_selection/gradient.py`)

**使用示例**:
```python
from methods.rl_teacher_selection import (
    get_mock_data,
    compute_gradient,
    compute_validation_gradient,
)

# 加载模拟数据
data = get_mock_data(num_instructions=100, num_val_samples=50, device="cpu")
student = data["student_model"]
val_loader = data["val_loader"]

# 测试单样本梯度计算
batch = next(iter(val_loader))
input_ids, labels = batch
g = compute_gradient(student, input_ids, labels)
print(f"Gradient shape: {g.shape}")
print(f"Gradient norm: {torch.norm(g).item():.4f}")

# 测试验证集梯度计算
g_val = compute_validation_gradient(student, val_loader, num_samples=10)
print(f"Validation gradient shape: {g_val.shape}")
print(f"Validation gradient norm: {torch.norm(g_val).item():.4f}")  # 应该是1.0（单位向量）

# 测试余弦相似度
import torch.nn.functional as F
cos_sim = F.cosine_similarity(g, g_val, dim=0)
print(f"Cosine similarity: {cos_sim.item():.4f}")
```

**测试要点**:
- 梯度向量应该是1D tensor
- 验证集梯度应该是单位向量（norm=1.0）
- 可以用MockStudentModel快速测试
- 检查梯度计算后model.zero_grad()是否正确清空

---

### 3. Reward Computation (`methods/rl_teacher_selection/reward.py`)

**使用示例**:
```python
from methods.rl_teacher_selection import (
    get_mock_data,
    compute_gradient,
    compute_validation_gradient,
    compute_reward,
)

# 加载模拟数据
data = get_mock_data(num_instructions=100, num_val_samples=50, device="cpu")
student = data["student_model"]
val_loader = data["val_loader"]
teachers = data["teachers"]

# 计算验证集梯度方向
g_val = compute_validation_gradient(student, val_loader, num_samples=10)

# 模拟一个teacher生成样本
teacher = teachers[0]  # gpt-4, cost=1.0
instruction = data["instructions"][0]
input_ids, labels = teacher.generate(instruction)

# 计算样本梯度
g_sample = compute_gradient(student, input_ids, labels)

# 计算奖励
reward = compute_reward(
    g_sample=g_sample,
    g_val=g_val,
    cost=teacher.cost,
    lambda_cost=0.05,
)
print(f"Reward: {reward:.4f}")

# 对比不同教师的奖励
for teacher in teachers:
    input_ids, labels = teacher.generate(instruction)
    g = compute_gradient(student, input_ids, labels)
    r = compute_reward(g, g_val, teacher.cost, lambda_cost=0.05)
    print(f"{teacher.name}: reward={r:.4f}, cost={teacher.cost}")

# 测试Reuse（cost=0）
r_reuse = compute_reward(g_sample, g_val, cost=0.0, lambda_cost=0.05)
print(f"Reuse: reward={r_reuse:.4f}, cost=0.0")
```

**测试要点**:
- 奖励值应该是标量（float）
- 高质量教师应该有更高的梯度对齐度
- Reuse动作（cost=0）应该有更高的奖励（无成本惩罚）
- 调整lambda_cost会改变成本权重

---

## 测试脚本

可以创建一个统一的测试脚本：

```bash
# 激活环境
conda activate sb3-env

# 测试所有模块
python -c "
from methods.rl_teacher_selection import (
    get_mock_data,
    PolicyNetwork,
    compute_gradient,
    compute_validation_gradient,
)
from transformers import AutoModel, AutoTokenizer
import torch

print('=== Testing Mock Data ===')
data = get_mock_data(num_instructions=10, num_val_samples=5, device='cpu')
print(f'✓ Mock data loaded')

print('\n=== Testing Gradient Computation ===')
student = data['student_model']
val_loader = data['val_loader']
batch = next(iter(val_loader))
input_ids, labels = batch

g = compute_gradient(student, input_ids, labels)
print(f'✓ Single gradient: shape={g.shape}, norm={torch.norm(g).item():.4f}')

g_val = compute_validation_gradient(student, val_loader, num_samples=2)
print(f'✓ Validation gradient: shape={g_val.shape}, norm={torch.norm(g_val).item():.4f}')

print('\n=== Testing Policy Network ===')
encoder = AutoModel.from_pretrained('microsoft/deberta-v3-base')
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
policy = PolicyNetwork(encoder, tokenizer, num_actions=4)
probs = policy('Test instruction')
print(f'✓ Policy forward: shape={probs.shape}, sum={probs.sum().item():.4f}')

print('\n✓ All tests passed!')
"
```

---

## 已完成模块

- ✅ `policy.py` - PolicyNetwork
- ✅ `mock_data.py` - 模拟数据（含cache机制）
- ✅ `gradient.py` - 梯度计算工具
- ✅ `reward.py` - 奖励计算工具
- ✅ `teacher_pool.py` - 教师池管理（统一cache）
- ✅ `sb3_env.py` - SB3环境（核心集成）

## 完整使用示例

```python
from methods.rl_teacher_selection import (
    get_mock_data,
    Teacher,
    TeacherPool,
    TeacherSelectionEnv,
)
from stable_baselines3 import PPO

# 1. 加载模拟数据
data = get_mock_data(num_instructions=1000, num_val_samples=500, device="cpu")

# 2. 创建教师池（使用mock teachers）
mock_teachers = data["teachers"]
teachers = [
    Teacher(t.name, t, cost=t.cost) for t in mock_teachers
]
teacher_pool = TeacherPool(teachers)

# 3. 创建环境
env = TeacherSelectionEnv(
    instructions=data["instructions"],
    student_model=data["student_model"],
    teacher_pool=teacher_pool,
    val_dataloader=data["val_loader"],
    config={
        "val_samples": 200,
        "lambda_cost": 0.05,
        "max_dataset_size": 50000,
        "recompute_val_grad_freq": 100,
    }
)

# 4. 训练policy（使用SB3的PPO）
model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1)
model.learn(total_timesteps=10000)

# 5. 保存policy
model.save("teacher_selection_policy")

# 6. 使用训练好的policy
obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

# 7. 查看cache统计
stats = teacher_pool.get_cache_stats()
for teacher_name, teacher_stats in stats.items():
    print(f"{teacher_name}: hit_rate={teacher_stats['hit_rate']:.1%}")
```

## 待实现模块

所有核心模块已完成！可选的扩展：
- ⬜ 真实数据集加载
- ⬜ 真实教师模型集成
- ⬜ Cache持久化（保存/加载）
- ⬜ 更复杂的student训练策略
- ⬜ 实验跟踪和可视化
