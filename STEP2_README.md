# Step 2: 训练路由模型

## 功能

训练一个 Qwen-1.5B-Instruct 模型作为路由器，根据问题（instruction）预测每个教师模型的 NTE 分数（0-10），用于后续的教师选择。

## 模型架构

```
输入: instruction (问题文本)
    ↓
Qwen-1.5B (全量微调)
    ↓
Last Token Pooling (提取最后一个非padding token的hidden state)
    ↓
Linear Head: hidden_dim → num_teachers
    ↓
输出: [score_teacher1, score_teacher2, ...] (每个教师的预测 NTE 分数, 0-10)
```

### 关键设计

1. **全量微调**: 对整个 Qwen-1.5B 模型进行微调，学习问题特征到教师适配性的映射
2. **Last Token Pooling**: 取最后一个有效 token 的 hidden state 作为句子表示
3. **回归任务**: 直接预测 NTE 分数（MSE Loss）
4. **掩码机制**: 处理部分教师缺失分数的情况

## 训练目标

- **任务**: 回归任务，预测每个教师的 NTE 分数
- **损失函数**: MSE Loss (Mean Squared Error)
- **输入**: instruction (问题文本)
- **标签**: 从 Step1 收集的 `teacher_scores[teacher_name]["nte_score"]`

## 使用方法

### 1. 前置条件

确保已经完成 Step1，生成了 NTE 分数文件：
```
method_cache/distill-openr1-math/step1_collect_scores.json
```

### 2. 配置文件

编辑 `configs/step2_train_router.yaml`：

```yaml
dataset_settings:
  type: "distill"
  name: "distill-openr1-math"  # 数据集名称（需与 Step1 一致）
  distill_settings:
    split:
      train: 10000   # 训练集样本数
      test: 1000     # 测试集样本数

backbone_settings:
  type: "llm"
  name: "qwen2.5-1.5b-instruct"
  model_id: "Qwen/Qwen2.5-1.5B-Instruct"
  llm_settings:
    device_map: "auto"

trainer_settings:
  type: "dl"
  name: "pytorch"
  dl_settings:
    batch_size: 8              # 批次大小
    num_epochs: 5              # 训练轮数
    learning_rate: 5.0e-5      # 学习率
    weight_decay: 0.01         # 权重衰减
    warmup_ratio: 0.1          # warmup 比例
    gradient_clip: 1.0         # 梯度裁剪

method_settings:
  study_name: "step2_train_router"

  # 必需的教师列表（需要与 Step1 一致）
  required_teachers:
    - "deepseek-r1"
    - "qwen2.5-math-7b-instruct"

  # 模型参数
  dropout: 0.1               # Dropout rate
  max_seq_length: 512        # 最大序列长度
```

### 3. 运行训练

```bash
# 使用默认配置
python step2_train_router.py

# 或指定配置文件
python step2_train_router.py --config configs/step2_train_router.yaml
```

### 4. 输出文件

训练过程中会保存：
- **Checkpoints**: 根据 `study_name` 保存到对应目录
- **日志**: 包含训练和评估指标

## 评估指标

训练过程中会记录以下指标：

1. **整体指标**:
   - MSE (Mean Squared Error): 平均平方误差
   - MAE (Mean Absolute Error): 平均绝对误差

2. **每个教师的指标**:
   - Per-teacher MSE: 每个教师的独立 MSE
   - Per-teacher MAE: 每个教师的独立 MAE

## 数据处理

### 输入格式

从 Step1 加载的数据格式：
```json
{
  "instruction": "问题描述...",
  "responses": {
    "teacher-name": {
      "messages": [...],
      "nte_scores": {
        "likelihood": 0.45,
        "proximity": 0.78,
        "quality": 0.85,
        "nte_score": 6.63  // 这是训练目标
      }
    }
  }
}
```

### 批次数据

训练时的批次格式：
```python
{
    "input_ids": [batch, seq_len],           # Tokenized instructions
    "attention_mask": [batch, seq_len],      # Attention mask
    "target_scores": [batch, num_teachers],  # 目标 NTE 分数
    "masks": [batch, num_teachers]           # 标记哪些教师有有效分数
}
```

## 调试模式

如果要快速测试（只使用少量样本）：

```yaml
dataset_settings:
  distill_settings:
    split:
      train: 100   # 只用 100 个样本
      test: 20     # 只用 20 个样本

trainer_settings:
  dl_settings:
    num_epochs: 2  # 只训练 2 个 epoch
```

## 超参数调优建议

### 学习率
- **全量微调**: 5e-5 到 2e-5（推荐 5e-5）
- **仅微调 Head**: 1e-4 到 5e-4

### 批次大小
- 根据 GPU 显存调整
- 1.5B 模型建议: 4-16

### 训练轮数
- 通常 3-5 个 epoch 足够
- 观察验证集 loss 是否收敛

### Dropout
- 0.1 (默认)
- 如果过拟合，增加到 0.2-0.3

## 后续步骤

训练完成后，路由模型可以用于：
1. **推理**: 给定新问题，预测每个教师的适配分数
2. **教师选择**: 选择分数最高的教师
3. **集成学习**: 基于分数进行加权集成

示例推理代码：
```python
# 加载训练好的路由模型
router = RouterRegressor(...)
router.load_state_dict(torch.load("checkpoint.pt"))
router.eval()

# 预测
instruction = "Solve: 2x + 3 = 7"
inputs = tokenizer(instruction, return_tensors="pt")
with torch.no_grad():
    scores = router(inputs["input_ids"], inputs["attention_mask"])
    # scores: [1, num_teachers]

# 选择最佳教师
best_teacher_idx = scores.argmax(dim=1).item()
best_teacher = required_teachers[best_teacher_idx]
print(f"Best teacher: {best_teacher}, Score: {scores[0, best_teacher_idx]:.2f}")
```

## 常见问题

### Q: 如果某些样本缺少部分教师的分数怎么办？
A: 使用 `masks` 参数标记有效的教师，损失计算时只考虑有效的教师。

### Q: 可以使用更大的模型吗？
A: 可以，修改 `backbone_settings.model_id` 即可，例如使用 Qwen-7B。

### Q: 训练很慢怎么办？
A:
1. 减小批次大小
2. 使用梯度累积
3. 使用更少的训练样本
4. 考虑使用 LoRA 等参数高效微调方法

### Q: 如何评估模型性能？
A: 观察测试集的 MSE 和 MAE，以及每个教师的独立指标。如果 MAE < 1.0，说明预测较准确。
