# RL-Based Cost-Aware Teacher Selection 实现计划

**创建时间**: 2026-01-16
**状态**: 设计阶段

---

## 0. 交流偏好

- 回复时说"好好"而不是"好的"

---

## 1. 项目概述

实现一个基于强化学习的成本感知教师选择系统，用于LLM知识蒸馏中的合成数据生成。

**核心目标**:
- 从教师池中动态选择最合适的教师模型
- 平衡数据质量和调用成本
- 支持历史数据复用以降低成本

---

## 2. 系统架构

### 2.1 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop                            │
│  ┌────────────┐    ┌──────────────┐    ┌─────────────┐     │
│  │ Instruction│ -> │ Policy Network│ -> │   Action    │     │
│  │    Text    │    │  (BERT + MLP) │    │ (Teacher/   │     │
│  └────────────┘    └──────────────┘    │  Reuse)     │     │
│                                         └─────────────┘     │
│                           ↓                                  │
│  ┌────────────┐    ┌──────────────┐    ┌─────────────┐     │
│  │  Reward    │ <- │   Gradient   │ <- │  Generate/  │     │
│  │ Computation│    │  Computation │    │   Sample    │     │
│  └────────────┘    └──────────────┘    └─────────────┘     │
│         ↓                                                    │
│  ┌────────────┐                                             │
│  │ PPO Update │                                             │
│  └────────────┘                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Policy Network

**架构**:
```python
Input: instruction (text)
  ↓
[Frozen BERT Encoder]  # 使用预训练BERT，参数冻结
  ↓
[Trainable MLP Head]   # 2-3层MLP，输出动作概率
  ↓
Output: [P(T₁), P(T₂), ..., P(Tₙ), P(Reuse)]
```

**设计要点**:
- BERT encoder冻结，利用预训练知识
- 只训练MLP head，样本效率高
- 输出包含所有教师 + Reuse动作的概率分布

### 2.3 Reward Function

```
R = Cosine(g_sample, ḡ_val) · ||g_sample|| - λ · Cost
```

**组成部分**:
- `g_sample`: 合成样本在student模型上的梯度（全参数，flatten）
- `ḡ_val`: 验证集平均梯度方向（归一化为单位向量）
- `Cosine(·,·)`: 梯度方向对齐度 ∈ [-1, 1]
- `||g_sample||`: 梯度强度（学习信号强度）
- `Cost`: 教师调用成本（Reuse为0）
- `λ`: 成本权重系数（超参数）

**直觉**:
- 质量项：既要方向对齐，又要信号强
- 成本项：惩罚高成本选择，鼓励复用
- Policy学习在质量-成本间找平衡

---

## 3. 训练流程

### 3.1 Batch-Based训练

```python
# 初始化
instructions = load_and_shuffle_instructions()
batches = split_into_batches(instructions, batch_size=b)
dataset_D = []  # 累积的合成数据集
policy = PolicyNetwork()
student = StudentModel(0.5B)

# 训练循环
for batch_id, batch in enumerate(batches):
    # 1. 计算验证集梯度方向
    ḡ_val = compute_validation_gradient(student, val_set, K_samples)

    # 2. 处理batch中的每个instruction
    for instruction in batch:
        # 2.1 Policy选择action
        probs = policy(instruction)
        action = sample(probs)  # 从分布中采样

        # 2.2 执行action
        if action == "Reuse" and len(dataset_D) > 0:
            x, y = random_sample(dataset_D)
            cost = 0
        else:  # action is teacher Tⱼ
            x, y = teacher_j.generate(instruction)
            cost = Cost[teacher_j]

        # 2.3 计算reward
        g_sample = compute_gradient(student, x, y)
        reward = cosine(g_sample, ḡ_val) * norm(g_sample) - λ * cost

        # 2.4 存储经验和数据
        experience_buffer.add(instruction, action, reward)
        dataset_D.append((x, y))

        # 2.5 更新policy（每N个样本）
        if len(experience_buffer) >= N:
            update_policy_with_ppo(policy, experience_buffer)

    # 3. 更新student（用当前batch的数据）
    train_student(student, dataset_D[-len(batch):])

    # 4. 限制数据集大小
    if len(dataset_D) > max_size:
        dataset_D = dataset_D[-max_size:]

    # 5. 评估
    if (batch_id + 1) % eval_freq == 0:
        evaluate(student, test_set)
```

### 3.2 关键特性

**两个更新频率**:
- Policy更新：每N个样本（100-500），快速适应
- Student更新：每个batch结束，用batch内数据训练

**数据复用机制**:
- 维护数据集D（最大50k-100k样本）
- Reuse动作从D中随机采样，成本为0
- Policy自动学习何时复用有价值

**验证集梯度**:
- 每个batch开始时重新计算ḡ_val
- 从验证集随机采样K个样本（100-500）
- 归一化为单位向量作为"正确方向"

---

## 4. 实现计划

### 4.1 Phase 1: 核心组件实现

**1. Policy Network** (`methods/rl_teacher_selection/policy.py`)
```python
class PolicyNetwork(nn.Module):
    def __init__(self, bert_model_name, num_teachers, mlp_hidden_dims):
        # Frozen BERT encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        # Trainable MLP head
        self.mlp = MLP(bert_hidden_size, mlp_hidden_dims, num_teachers + 1)

    def forward(self, instruction_text):
        # Encode instruction
        with torch.no_grad():
            bert_output = self.bert(instruction_text)

        # Get action probabilities
        logits = self.mlp(bert_output)
        probs = F.softmax(logits, dim=-1)
        return probs
```

**2. Gradient Computation** (`methods/rl_teacher_selection/gradient.py`)
```python
def compute_gradient(model, x, y):
    """计算单个样本的梯度向量"""
    model.zero_grad()
    loss = model(x, y)
    loss.backward()

    # 收集所有参数梯度
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.flatten())

    g = torch.cat(grads)
    model.zero_grad()
    return g

def compute_validation_gradient(model, val_loader, num_samples):
    """计算验证集平均梯度方向"""
    grad_sum = None
    samples = random_sample(val_loader, num_samples)

    for x, y in samples:
        g = compute_gradient(model, x, y)
        grad_sum = g if grad_sum is None else grad_sum + g

    g_val = grad_sum / num_samples
    g_val_normalized = g_val / torch.norm(g_val)
    return g_val_normalized
```

**3. Reward Computation** (`methods/rl_teacher_selection/reward.py`)
```python
def compute_reward(g_sample, g_val, cost, lambda_cost):
    """计算奖励"""
    # 方向对齐度
    cosine_sim = F.cosine_similarity(g_sample, g_val, dim=0)

    # 梯度强度
    grad_norm = torch.norm(g_sample)

    # 总奖励
    reward = cosine_sim * grad_norm - lambda_cost * cost
    return reward.item()
```

**4. Training Loop** (`methods/rl_teacher_selection/trainer.py`)
```python
class RLTeacherSelectionTrainer:
    def __init__(self, config, policy, student, teachers, val_loader):
        self.policy = policy
        self.student = student
        self.teachers = teachers
        self.val_loader = val_loader

        # PPO components (using Stable-Baselines3)
        self.ppo_policy = ...

        # Data management
        self.dataset_D = []
        self.experience_buffer = []

    def train(self, instructions):
        batches = self.split_batches(instructions)

        for batch_id, batch in enumerate(batches):
            # 计算验证集梯度
            g_val = compute_validation_gradient(
                self.student, self.val_loader, self.config.val_samples
            )

            # 处理batch
            for instruction in batch:
                self.process_instruction(instruction, g_val)

            # 更新student
            self.update_student(batch)

            # 评估
            if (batch_id + 1) % self.config.eval_freq == 0:
                self.evaluate()
```

### 4.2 Phase 2: 集成到Engine框架

**目录结构**:
```
methods/rl_teacher_selection/
├── __init__.py
├── policy.py           # Policy network
├── gradient.py         # 梯度计算
├── reward.py           # 奖励计算
├── trainer.py          # 训练循环
├── teacher_pool.py     # 教师池管理
└── main.py             # 入口函数
```

**配置文件** (`configs/rl_teacher_selection.yaml`):
```yaml
method: rl_teacher_selection

# Policy network
policy_settings:
  bert_model: "bert-base-uncased"
  mlp_hidden_dims: [512, 256]
  freeze_bert: true

# Teachers
teacher_pool:
  - name: "gpt-4"
    cost: 1.0
  - name: "gpt-3.5"
    cost: 0.1
  - name: "llama-70b"
    cost: 0.5

# Training
training_settings:
  batch_size: 1000
  policy_update_freq: 200
  eval_freq: 5
  max_dataset_size: 50000

# Reward
reward_settings:
  lambda_cost: 0.05
  val_samples: 200

# PPO
ppo_settings:
  learning_rate: 1e-4
  clip_range: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
```

### 4.3 Phase 3: 实验和优化

**Baseline对比**:
- Random selection
- Fixed teacher (最贵/最便宜)
- Round-robin
- Cost-weighted random

**消融实验**:
- 不同λ值的影响
- 是否使用Reuse
- 部分层梯度 vs 全参数梯度
- 不同policy更新频率

**指标监控**:
- Student性能（accuracy/loss）
- 总成本
- 各教师选择频率
- Reuse频率
- 平均reward
- Policy loss/entropy

---

## 5. 技术栈

**环境**:
- Conda环境: `sb3-env`
- Python: 3.11
- PyTorch: 2.9.1+cu126
- CUDA: 12.6

**核心库**:
- Stable-Baselines3 2.7.1 (PPO实现)
- Transformers (BERT encoder)
- Gymnasium 1.2.3 (RL环境接口)

**现有框架**:
- `engine/configs`: 配置加载
- `engine/datas`: 数据集加载
- `engine/trainers`: 训练器框架
- `engine/backbones`: 模型加载

---

## 6. 关键挑战和解决方案

### 6.1 梯度计算效率

**挑战**: 0.5B参数模型，每个样本需要1次backward，可能成为瓶颈

**解决方案**:
1. 先用全参数梯度实现，测试是否是瓶颈
2. 如果慢，尝试只用最后2-3层梯度
3. 使用gradient checkpointing节省内存
4. 批处理多个样本（但改变语义）

### 6.2 Reuse过度依赖

**挑战**: 随着D增大，Reuse越来越有吸引力，可能导致policy过度依赖

**解决方案**:
1. 限制D大小（50k-100k）
2. Entropy bonus鼓励探索
3. 可选：给新生成action加exploration bonus
4. 监控Reuse频率，必要时调整

### 6.3 验证集梯度稳定性

**挑战**: ḡ_val需要足够稳定，但采样数不能太多

**解决方案**:
1. 超参数调优（100-500样本）
2. 每个batch重新采样（增加多样性）
3. 监控ḡ_val的变化幅度

### 6.4 PPO训练稳定性

**挑战**: Reward分布可能变化大，影响PPO稳定性

**解决方案**:
1. Reward normalization
2. 合适的clip range（0.2）
3. Value function帮助稳定
4. 监控policy loss和KL divergence

---

## 7. 下一步行动

### 立即开始
1. ✅ 环境配置（sb3-env已完成）
2. ⬜ 实现Policy Network
3. ⬜ 实现梯度计算模块
4. ⬜ 实现奖励计算

### 短期目标
5. ⬜ 实现训练循环框架
6. ⬜ 集成到engine框架
7. ⬜ 准备小规模测试数据
8. ⬜ 运行第一个实验

### 中期目标
9. ⬜ Baseline对比实验
10. ⬜ 超参数调优
11. ⬜ 消融实验
12. ⬜ 性能优化

---

## 8. 参考文档

- **详细设计**: `doc.md`
- **项目架构**: `.history2/CLAUDE.md`
- **配置示例**: `configs/`
- **Engine框架**: `engine/`

---

**最后更新**: 2026-01-16
**负责人**: Claude + User
