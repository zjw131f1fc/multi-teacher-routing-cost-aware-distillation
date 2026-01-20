# RL Teacher Selection 模块化设计

## 设计原则

1. **单一职责**: 每个模块只负责一个明确的功能
2. **依赖注入**: 通过参数传递依赖，避免内部创建
3. **函数式优先**: 核心逻辑使用纯函数，便于测试和组合
4. **Jupyter友好**: 每个模块可独立导入和测试

---

## 模块架构

```
methods/rl_teacher_selection/
├── __init__.py              # 统一导出接口
├── policy.py                # ✅ PolicyNetwork (已完成)
├── gradient.py              # 梯度计算工具
├── reward.py                # 奖励计算工具
├── teacher_pool.py          # 教师池管理
├── experience.py            # 经验缓冲区
└── trainer.py               # 训练循环（可选）
```

---

## 1. PolicyNetwork (已完成)

**文件**: `policy.py`

**职责**: 策略网络，输入instruction，输出动作概率

**接口**:
```python
class PolicyNetwork(nn.Module):
    def __init__(self, encoder, tokenizer, num_actions, hidden_dims, freeze_encoder)
    def forward(self, instruction_text) -> torch.Tensor  # 返回概率
    def sample_action(self, instruction_text) -> (actions, log_probs)
    def get_action_probs(self, instruction_text, actions) -> torch.Tensor
```

**特点**:
- 接受外部加载的encoder和tokenizer
- 冻结encoder，只训练MLP head
- 无副作用，纯粹的神经网络模块

---

## 2. Gradient Module

**文件**: `gradient.py`

**职责**: 梯度计算相关的工具函数

**接口**:
```python
def compute_sample_gradient(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: callable = None
) -> torch.Tensor:
    """计算单个样本的梯度向量

    Args:
        model: 学生模型
        input_ids: 输入token ids
        labels: 标签
        loss_fn: 损失函数（可选，默认用model的forward）

    Returns:
        grad: 梯度向量 (flatten后的一维tensor)
    """
    pass


def compute_validation_gradient(
    model: nn.Module,
    val_dataloader: DataLoader,
    num_samples: int,
    loss_fn: callable = None
) -> torch.Tensor:
    """计算验证集平均梯度方向

    Args:
        model: 学生模型
        val_dataloader: 验证集dataloader
        num_samples: 采样数量
        loss_fn: 损失函数

    Returns:
        g_val: 归一化的平均梯度方向 (单位向量)
    """
    pass


def compute_gradient_batch(
    model: nn.Module,
    batch_data: list[tuple],
    loss_fn: callable = None
) -> list[torch.Tensor]:
    """批量计算多个样本的梯度（并行优化版本，可选）

    Args:
        model: 学生模型
        batch_data: [(input_ids, labels), ...]
        loss_fn: 损失函数

    Returns:
        grads: 梯度列表
    """
    pass
```

**特点**:
- 纯函数，无状态
- 支持自定义loss函数
- 可选的批量优化版本
- 在notebook中可以单独测试每个函数

---

## 3. Reward Module

**文件**: `reward.py`

**职责**: 奖励计算相关的工具函数

**接口**:
```python
def compute_reward(
    g_sample: torch.Tensor,
    g_val: torch.Tensor,
    cost: float,
    lambda_cost: float
) -> float:
    """计算奖励: R = Cosine(g_sample, g_val) * ||g_sample|| - λ * Cost

    Args:
        g_sample: 样本梯度向量
        g_val: 验证集梯度方向（归一化）
        cost: 动作成本
        lambda_cost: 成本权重系数

    Returns:
        reward: 标量奖励值
    """
    pass


def compute_cosine_similarity(
    vec1: torch.Tensor,
    vec2: torch.Tensor
) -> float:
    """计算两个向量的余弦相似度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        cosine_sim: 余弦相似度 ∈ [-1, 1]
    """
    pass


def compute_gradient_norm(
    grad: torch.Tensor
) -> float:
    """计算梯度范数

    Args:
        grad: 梯度向量

    Returns:
        norm: L2范数
    """
    pass
```

**特点**:
- 纯函数，数学计算
- 每个函数都可以独立测试
- 清晰的输入输出

---

## 4. Teacher Pool Module

**文件**: `teacher_pool.py`

**职责**: 管理教师模型池，提供统一的调用接口

**接口**:
```python
class Teacher:
    """单个教师的抽象"""
    def __init__(self, name: str, model, tokenizer, cost: float):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.cost = cost

    def generate(self, instruction: str, **kwargs) -> str:
        """生成响应"""
        pass


class TeacherPool:
    """教师池管理"""
    def __init__(self, teachers: list[Teacher]):
        self.teachers = teachers
        self.num_teachers = len(teachers)

    def get_teacher(self, action_id: int) -> Teacher:
        """根据action id获取教师"""
        pass

    def get_cost(self, action_id: int) -> float:
        """获取动作成本（Reuse返回0）"""
        pass

    @property
    def num_actions(self) -> int:
        """动作数量（教师数 + 1 for Reuse）"""
        return self.num_teachers + 1
```

**特点**:
- 封装教师模型的调用细节
- 统一的接口，支持不同类型的教师（API、本地模型等）
- 成本管理集中在这里

---

## 5. Experience Module

**文件**: `experience.py`

**职责**: 管理RL经验缓冲区

**接口**:
```python
class Experience:
    """单条经验"""
    def __init__(
        self,
        instruction: str,
        action: int,
        reward: float,
        log_prob: float = None
    ):
        self.instruction = instruction
        self.action = action
        self.reward = reward
        self.log_prob = log_prob


class ExperienceBuffer:
    """经验缓冲区"""
    def __init__(self, max_size: int = 10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience: Experience):
        """添加经验"""
        pass

    def sample(self, batch_size: int) -> list[Experience]:
        """随机采样"""
        pass

    def get_all(self) -> list[Experience]:
        """获取所有经验"""
        pass

    def clear(self):
        """清空缓冲区"""
        pass

    def __len__(self) -> int:
        return len(self.buffer)
```

**特点**:
- 简单的数据结构
- 支持采样和批量获取
- 可以轻松扩展（如优先级采样）

---

## 6. SB3 Integration Module

**文件**: `sb3_env.py`

**职责**: 将我们的问题包装成Gymnasium环境，与SB3集成

**接口**:
```python
import gymnasium as gym
from gymnasium import spaces

class TeacherSelectionEnv(gym.Env):
    """Gymnasium环境，用于SB3训练

    Observation: instruction的embedding (固定维度向量)
    Action: 离散动作空间 [0, num_actions-1]
    Reward: 梯度对齐奖励 - 成本
    """

    def __init__(
        self,
        instructions: list[str],
        student_model: nn.Module,
        teacher_pool: TeacherPool,
        val_dataloader: DataLoader,
        policy_encoder,  # 用于编码instruction
        policy_tokenizer,
        config: dict
    ):
        super().__init__()

        self.instructions = instructions
        self.student = student_model
        self.teachers = teacher_pool
        self.val_loader = val_dataloader
        self.encoder = policy_encoder
        self.tokenizer = policy_tokenizer
        self.config = config

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(768,),  # DeBERTa hidden size
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(teacher_pool.num_actions)

        # 内部状态
        self.current_idx = 0
        self.dataset_D = []
        self.g_val = None

    def reset(self, seed=None, options=None):
        """重置环境，返回第一个observation"""
        super().reset(seed=seed)
        self.current_idx = 0

        # 重新计算验证集梯度
        self.g_val = compute_validation_gradient(
            self.student,
            self.val_loader,
            self.config["val_samples"]
        )

        # 返回第一个instruction的embedding
        obs = self._get_observation(self.current_idx)
        info = {}
        return obs, info

    def step(self, action: int):
        """执行action，返回(obs, reward, terminated, truncated, info)"""
        instruction = self.instructions[self.current_idx]

        # 执行action
        if action < self.teachers.num_teachers:
            # 调用教师生成
            teacher = self.teachers.get_teacher(action)
            response = teacher.generate(instruction)
            cost = teacher.cost
        else:
            # Reuse
            if len(self.dataset_D) > 0:
                response = random.choice(self.dataset_D)
            else:
                # 如果D为空，fallback到第一个教师
                teacher = self.teachers.get_teacher(0)
                response = teacher.generate(instruction)
            cost = 0.0

        # 计算reward
        input_ids, labels = self._prepare_data(instruction, response)
        g_sample = compute_sample_gradient(self.student, input_ids, labels)
        reward = compute_reward(g_sample, self.g_val, cost, self.config["lambda_cost"])

        # 存储数据
        self.dataset_D.append((instruction, response))
        if len(self.dataset_D) > self.config["max_dataset_size"]:
            self.dataset_D.pop(0)

        # 移动到下一个instruction
        self.current_idx += 1
        terminated = (self.current_idx >= len(self.instructions))
        truncated = False

        # 下一个observation
        if not terminated:
            obs = self._get_observation(self.current_idx)
        else:
            obs = np.zeros(768, dtype=np.float32)

        info = {
            "instruction": instruction,
            "action": action,
            "cost": cost,
            "response": response
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self, idx: int) -> np.ndarray:
        """将instruction编码为observation"""
        instruction = self.instructions[idx]

        # 使用policy的encoder编码
        inputs = self.tokenizer(
            instruction,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

        return embedding.cpu().numpy()

    def _prepare_data(self, instruction: str, response: str):
        """准备训练数据（需要根据具体任务实现）"""
        # 这里需要根据具体的student模型格式来实现
        pass
```

**特点**:
- 标准Gymnasium接口，完全兼容SB3
- 封装了所有环境逻辑
- 自动管理数据集D和验证集梯度

---

## 7. SB3 Training Wrapper

**文件**: `sb3_trainer.py`

**职责**: 使用SB3的PPO训练policy

**接口**:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class SB3Trainer:
    """使用SB3训练policy的包装器"""

    def __init__(
        self,
        env: TeacherSelectionEnv,
        policy_network: PolicyNetwork,
        config: dict
    ):
        self.env = env
        self.policy_net = policy_network
        self.config = config

        # 创建SB3的PPO模型
        # 注意：这里需要将我们的PolicyNetwork适配为SB3的policy
        self.model = PPO(
            policy="MlpPolicy",  # 或自定义policy
            env=env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            verbose=1
        )

    def train(self, total_timesteps: int):
        """训练policy"""
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str):
        """保存模型"""
        self.model.save(path)

    def load(self, path: str):
        """加载模型"""
        self.model = PPO.load(path, env=self.env)
```

**问题**: SB3的policy和我们的PolicyNetwork不兼容

**解决方案**: 两种方式

### 方案A: 使用SB3的policy（推荐）

不使用我们自己的PolicyNetwork，直接用SB3的MlpPolicy：

```python
# 创建环境
env = TeacherSelectionEnv(...)

# 使用SB3的policy
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    ...
)

# 训练
model.learn(total_timesteps=100000)
```

**优点**: 简单，充分利用SB3的优化
**缺点**: 不能使用预训练的DeBERTa encoder

### 方案B: 自定义SB3 Policy（复杂但灵活）

将我们的PolicyNetwork包装成SB3兼容的policy：

```python
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 encoder, tokenizer, **kwargs):
        # 使用我们的PolicyNetwork作为feature extractor
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # 替换默认的feature extractor
        self.encoder = encoder
        # 冻结encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, obs, deterministic=False):
        # 自定义forward逻辑
        pass
```

**优点**: 可以使用预训练encoder
**缺点**: 实现复杂，需要深入理解SB3的policy接口

---

## 推荐方案：混合架构

考虑到灵活性和简单性，推荐以下架构：

### 核心模块（独立，不依赖SB3）
1. `gradient.py` - 梯度计算工具
2. `reward.py` - 奖励计算工具
3. `teacher_pool.py` - 教师池管理
4. `experience.py` - 经验缓冲区

### SB3集成模块（可选）
5. `sb3_env.py` - Gymnasium环境包装
6. `sb3_trainer.py` - SB3训练包装器

### 自定义训练模块（可选）
7. `policy.py` - 自定义PolicyNetwork（如果不用SB3的policy）
8. `custom_trainer.py` - 自定义训练循环（如果不用SB3）

### 使用场景

**场景1: 使用SB3（简单，推荐）**
```python
# 只需要核心模块 + SB3集成
from methods.rl_teacher_selection import (
    TeacherSelectionEnv,
    TeacherPool,
    compute_sample_gradient,
    compute_validation_gradient,
    compute_reward
)
from stable_baselines3 import PPO

# 创建环境
env = TeacherSelectionEnv(...)

# 使用SB3训练
model = PPO("MlpPolicy", env, ...)
model.learn(total_timesteps=100000)
```

**场景2: 自定义训练（灵活）**
```python
# 使用核心模块 + 自定义policy
from methods.rl_teacher_selection import (
    PolicyNetwork,
    TeacherPool,
    compute_sample_gradient,
    compute_reward,
    ExperienceBuffer
)

# 手动实现训练循环
policy = PolicyNetwork(...)
for instruction in instructions:
    action = policy.sample_action(instruction)
    # ... 自己实现PPO更新
```

**特点**:
- 高层封装，适合快速实验
- 但用户也可以不用，自己组合底层模块
- 在notebook中可以逐步调用各个方法

---

## 使用示例（Jupyter Notebook）

### 方案A: 使用SB3（推荐，简单）

```python
# Cell 1: 导入
from transformers import AutoModel, AutoTokenizer
from stable_baselines3 import PPO
from methods.rl_teacher_selection import (
    TeacherSelectionEnv,
    TeacherPool,
    Teacher,
    compute_sample_gradient,
    compute_validation_gradient,
    compute_reward
)

# Cell 2: 加载模型
encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
student_model = load_student_model()  # 你的student模型

# Cell 3: 创建教师池
teachers = [
    Teacher("gpt-4", gpt4_model, gpt4_tokenizer, cost=1.0),
    Teacher("gpt-3.5", gpt35_model, gpt35_tokenizer, cost=0.1),
    Teacher("llama", llama_model, llama_tokenizer, cost=0.5),
]
teacher_pool = TeacherPool(teachers)

# Cell 4: 创建Gymnasium环境
env = TeacherSelectionEnv(
    instructions=train_instructions,
    student_model=student_model,
    teacher_pool=teacher_pool,
    val_dataloader=val_loader,
    policy_encoder=encoder,
    policy_tokenizer=tokenizer,
    config={
        "val_samples": 200,
        "lambda_cost": 0.05,
        "max_dataset_size": 50000,
    }
)

# Cell 5: 创建PPO模型（使用SB3的MlpPolicy）
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/"
)

# Cell 6: 训练
model.learn(total_timesteps=100000)

# Cell 7: 保存模型
model.save("teacher_selection_policy")

# Cell 8: 测试
obs, info = env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: action={action}, reward={reward:.4f}")
    if terminated:
        break
```

### 方案B: 使用自定义PolicyNetwork + 手动训练循环

```python
# Cell 1: 导入
from transformers import AutoModel, AutoTokenizer
from methods.rl_teacher_selection import (
    PolicyNetwork,
    TeacherPool,
    Teacher,
    ExperienceBuffer,
    Experience,
    compute_sample_gradient,
    compute_validation_gradient,
    compute_reward
)

# Cell 2: 加载模型
encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
student_model = load_student_model()

# Cell 3: 创建教师池
teachers = [
    Teacher("gpt-4", gpt4_model, gpt4_tokenizer, cost=1.0),
    Teacher("gpt-3.5", gpt35_model, gpt35_tokenizer, cost=0.1),
    Teacher("llama", llama_model, llama_tokenizer, cost=0.5),
]
teacher_pool = TeacherPool(teachers)

# Cell 4: 创建policy
policy = PolicyNetwork(
    encoder=encoder,
    tokenizer=tokenizer,
    num_actions=teacher_pool.num_actions,
    hidden_dims=[512, 256],
    freeze_encoder=True
)

# Cell 5: 初始化
dataset_D = []
experience_buffer = ExperienceBuffer(max_size=10000)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

# Cell 6: 计算验证集梯度
g_val = compute_validation_gradient(
    model=student_model,
    val_dataloader=val_loader,
    num_samples=200
)

# Cell 7: 训练循环（单个batch）
for instruction in train_instructions[:1000]:
    # 7.1 Policy选择action
    action, log_prob = policy.sample_action(instruction)
    action = action.item()

    # 7.2 执行action
    if action < teacher_pool.num_teachers:
        teacher = teacher_pool.get_teacher(action)
        response = teacher.generate(instruction)
        cost = teacher.cost
    else:  # Reuse
        if len(dataset_D) > 0:
            response = random.choice(dataset_D)
        else:
            teacher = teacher_pool.get_teacher(0)
            response = teacher.generate(instruction)
        cost = 0.0

    # 7.3 计算梯度和奖励
    input_ids, labels = prepare_data(instruction, response)
    g_sample = compute_sample_gradient(student_model, input_ids, labels)
    reward = compute_reward(g_sample, g_val, cost, lambda_cost=0.05)

    # 7.4 存储经验
    exp = Experience(instruction, action, reward, log_prob.item())
    experience_buffer.add(exp)
    dataset_D.append(response)

    # 7.5 更新policy（每N步）
    if len(experience_buffer) >= 200:
        # 这里需要实现PPO更新逻辑
        # 或者使用SB3的PPO算法
        update_policy_with_ppo(policy, experience_buffer, optimizer)
        experience_buffer.clear()

# Cell 8: 评估
# ...
```

### 方案C: 混合方式（推荐用于实验）

使用核心模块进行灵活实验，然后切换到SB3进行大规模训练：

```python
# 阶段1: 在notebook中快速原型和调试
# 使用核心模块手动测试各个组件

# 测试梯度计算
g_sample = compute_sample_gradient(student_model, input_ids, labels)
print(f"Gradient shape: {g_sample.shape}")
print(f"Gradient norm: {torch.norm(g_sample)}")

# 测试奖励计算
reward = compute_reward(g_sample, g_val, cost=1.0, lambda_cost=0.05)
print(f"Reward: {reward}")

# 测试policy
probs = policy(instruction)
print(f"Action probs: {probs}")

# 阶段2: 确认逻辑正确后，切换到SB3进行训练
env = TeacherSelectionEnv(...)
model = PPO("MlpPolicy", env, ...)
model.learn(total_timesteps=100000)
```

---

## 设计优势

1. **高度模块化**: 每个模块可以独立导入和测试
2. **灵活组合**: 用户可以选择使用高层Trainer或手动组合底层模块
3. **Jupyter友好**: 可以在notebook中逐步构建和测试
4. **易于扩展**: 每个模块职责清晰，容易修改和扩展
5. **无隐藏依赖**: 所有依赖通过参数显式传递
6. **纯函数优先**: 核心计算逻辑使用纯函数，易于测试

---

## 实现顺序建议

1. ✅ `policy.py` - 已完成
2. ⬜ `gradient.py` - 梯度计算（核心）
3. ⬜ `reward.py` - 奖励计算（简单）
4. ⬜ `teacher_pool.py` - 教师池管理
5. ⬜ `experience.py` - 经验缓冲区（简单）
6. ⬜ `trainer.py` - 训练循环（可选，最后实现）

每个模块实现后都可以在notebook中单独测试，确保正确性后再继续下一个。
