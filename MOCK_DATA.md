# Mock Data 使用说明

## 概述

提供了完整的模拟数据和模型，用于测试RL教师选择框架，无需真实数据集。

## 快速开始

```python
from methods.rl_teacher_selection import get_mock_data

# 一行加载所有模拟数据
data = get_mock_data(
    num_instructions=1000,    # 训练instructions数量
    num_val_samples=500,      # 验证集样本数量
    device="cpu"              # 或 "cuda"
)

# 包含：
# - data["instructions"]: List[str] - 训练instructions
# - data["student_model"]: MockStudentModel - 学生模型（~45M参数）
# - data["teachers"]: List[MockTeacher] - 3个教师（不同质量/成本）
# - data["val_loader"]: DataLoader - 验证集
```

## 组件说明

### 1. Instructions

模拟的数学问题：
```python
from methods.rl_teacher_selection import get_mock_instructions

instructions = get_mock_instructions(num_samples=1000)
# ["Solve the equation: 16x + 3 = 16", "Calculate: 15 * 19 + 18", ...]
```

### 2. Student Model

简单的Transformer模型（~45M参数）：
```python
from methods.rl_teacher_selection import get_mock_student_model

student = get_mock_student_model(device="cpu")

# Forward pass
loss = student(input_ids, labels)  # 返回loss
logits = student(input_ids)        # 返回logits
```

**特点**:
- 4层Transformer
- 512 hidden size
- 足够大以测试梯度计算，但足够小以快速运行

### 3. Teachers

3个不同质量/成本的教师：
```python
from methods.rl_teacher_selection import get_mock_teachers

teachers = get_mock_teachers()
# [
#   MockTeacher("gpt-4", quality=0.95, cost=1.0),
#   MockTeacher("gpt-3.5", quality=0.80, cost=0.1),
#   MockTeacher("llama-70b", quality=0.85, cost=0.5),
# ]

# 生成响应
teacher = teachers[0]
input_ids, labels = teacher.generate("Solve: 2x + 5 = 13")
```

**质量差异**:
- 高质量教师：生成更一致的tokens
- 低质量教师：生成更随机的tokens

**Cache机制**:
- 每个教师维护一个cache，存储已生成的响应
- 重复调用相同instruction会直接返回cache结果（模拟真实场景）
- 可以预填充cache来模拟之前实验的结果

```python
# Cache自动工作
teacher = teachers[0]
result1 = teacher.generate("Solve: x + 1 = 2")  # Cache miss
result2 = teacher.generate("Solve: x + 1 = 2")  # Cache hit - 返回相同结果

# 查看cache统计
stats = teacher.get_cache_stats()
print(f"Cache size: {stats['cache_size']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")

# 预填充cache（模拟之前的实验）
teacher.prefill_cache(instructions[:100])

# 清空cache
teacher.clear_cache()
```

### 4. Validation Loader

标准PyTorch DataLoader：
```python
from methods.rl_teacher_selection import get_mock_val_loader

val_loader = get_mock_val_loader(
    num_samples=500,
    batch_size=32,
    seq_len=50
)

# 使用
for input_ids, labels in val_loader:
    loss = student(input_ids, labels)
```

## 测试

```bash
# 运行测试
python methods/rl_teacher_selection/test_mock_data.py
```

## 替换为真实数据

当你准备好真实数据时，只需要提供相同接口的对象：

```python
# 替换前（mock）
data = get_mock_data()

# 替换后（真实）
data = {
    "instructions": your_real_instructions,      # List[str]
    "student_model": your_real_student_model,    # nn.Module with forward(input_ids, labels)
    "teachers": your_real_teachers,              # List of objects with generate() method
    "val_loader": your_real_val_loader,          # DataLoader
}

# 其余代码无需修改！
```

## 注意事项

1. **模拟数据不代表真实性能** - 仅用于测试框架
2. **Student模型很小** - 真实模型可能是0.5B-1B参数
3. **Teacher生成是随机的** - 真实教师会生成有意义的响应
4. **Instructions是简单模板** - 真实数据会更复杂

## 下一步

有了模拟数据，可以开始实现：
1. 梯度计算模块（使用mock student model测试）
2. 奖励计算模块（使用mock数据测试）
3. SB3环境（完整集成测试）
