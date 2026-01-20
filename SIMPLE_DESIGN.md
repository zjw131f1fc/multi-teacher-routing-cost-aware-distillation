# 简化设计说明

## 核心原则

**模块化 + 高层封装** = 在notebook中方便使用

不需要在notebook中逐步构建所有细节，只需要能方便地导入和使用大模块。

---

## 模块分类

### 1. 工具函数（底层，可选直接使用）
- `gradient.py` - 梯度计算工具
- `reward.py` - 奖励计算工具

**特点**: 纯函数，可以单独测试，但通常被高层模块调用

### 2. 核心模块（高层，notebook中主要使用）
- `teacher_pool.py` - 教师池管理
- `sb3_env.py` - SB3环境（包含所有训练逻辑）

**特点**: 封装完整功能，在notebook中直接导入使用

---

## Notebook使用示例（简化版）

```python
# Cell 1: 导入
from transformers import AutoModel, AutoTokenizer
from stable_baselines3 import PPO
from methods.rl_teacher_selection import TeacherSelectionEnv, TeacherPool, Teacher

# Cell 2: 准备模型（你已有的代码）
encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
student_model = your_student_model
val_loader = your_val_loader

# Cell 3: 创建教师池
teachers = [
    Teacher("gpt-4", model1, tok1, cost=1.0),
    Teacher("gpt-3.5", model2, tok2, cost=0.1),
]
teacher_pool = TeacherPool(teachers)

# Cell 4: 创建环境（一行搞定）
env = TeacherSelectionEnv(
    instructions=train_instructions,
    student_model=student_model,
    teacher_pool=teacher_pool,
    val_dataloader=val_loader,
    policy_encoder=encoder,
    policy_tokenizer=tokenizer,
    config={"val_samples": 200, "lambda_cost": 0.05}
)

# Cell 5: 训练（SB3自动处理）
model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1)
model.learn(total_timesteps=100000)

# Cell 6: 保存
model.save("policy")
```

**就这么简单！** 不需要手动写训练循环，不需要逐步构建。

---

## 实现优先级

### 必须实现（核心功能）
1. ✅ `policy.py` - PolicyNetwork（已完成，但可能不用）
2. ⬜ `gradient.py` - 梯度计算工具
3. ⬜ `reward.py` - 奖励计算工具
4. ⬜ `teacher_pool.py` - 教师池管理
5. ⬜ `sb3_env.py` - SB3环境（最重要）

### 可选实现
- `experience.py` - 如果不用SB3才需要
- `custom_trainer.py` - 如果不用SB3才需要

---

## 关键点

1. **TeacherSelectionEnv** 是核心
   - 封装所有训练逻辑（梯度计算、奖励计算、数据管理）
   - 提供标准Gym接口
   - 在notebook中一行创建

2. **TeacherPool** 管理教师
   - 封装教师调用
   - 统一成本管理

3. **工具函数** 被环境内部调用
   - 用户通常不需要直接使用
   - 但可以单独测试

---

## 总结

**用户视角**: 在notebook中只需要关心3个东西
1. TeacherPool - 管理教师
2. TeacherSelectionEnv - 创建环境
3. SB3的PPO - 训练

**实现视角**: 需要实现5个模块
1. gradient.py - 工具
2. reward.py - 工具
3. teacher_pool.py - 高层模块
4. sb3_env.py - 高层模块（最重要）
5. policy.py - 已完成（但可能用SB3的policy代替）
