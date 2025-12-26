# DeBERTa Encoder Backbone 使用示例

## 基本配置

```yaml
backbone_settings:
  type: "encoder"
  name: "deberta-v3-base"
  encoder_settings:
    device_map: "auto"  # 或 "cpu", "cuda:0"
    pooling_mode: "mean"  # "mean", "cls", "max"
```

## 可用模型

- `deberta-v3-base`: microsoft/deberta-v3-base
- `deberta-v3-large`: microsoft/deberta-v3-large
- `deberta-v3-small`: microsoft/deberta-v3-small

## 自定义模型路径

```yaml
backbone_settings:
  type: "encoder"
  name: "deberta-v3-base"
  model_id: "path/to/your/custom/deberta"  # 覆盖默认路径
  encoder_settings:
    device_map: "auto"
    pooling_mode: "mean"
```

## 全局设置

```yaml
global_settings:
  device: "cuda:0"  # 或 "cpu"
  dtype: "float32"  # "float32", "float16", "bfloat16"
```

## Python 使用示例

### 1. 加载 Backbone

```python
from engine.configs import load_config
from engine.backbones import load_backbone

# 加载配置
config = load_config("configs/my_config.yaml")

# 加载 backbone
backbone = load_backbone(config)
```

### 2. 使用 forward() - 获取完整输出

```python
import torch

# 准备输入
text = "This is a test sentence."
tokens = backbone.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
tokens = {k: v.to(backbone.output_device) for k, v in tokens.items()}

# 前向传播
outputs = backbone.forward(**tokens, return_dict=True)

# 访问输出
last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
```

### 3. 使用 encode() - 获取句子表示

```python
# 单个文本
text = "This is a test sentence."
embedding = backbone.encode(text, pooling_mode="mean")
print(f"Embedding shape: {embedding.shape}")  # [hidden_size]

# 批量文本
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence.",
]
embeddings = backbone.encode(texts, pooling_mode="mean")
print(f"Embeddings shape: {embeddings.shape}")  # [batch_size, hidden_size]
```

### 4. 不同的池化模式

```python
# CLS token 池化
cls_embedding = backbone.encode(text, pooling_mode="cls")

# 平均池化（考虑 attention mask）
mean_embedding = backbone.encode(text, pooling_mode="mean")

# 最大池化
max_embedding = backbone.encode(text, pooling_mode="max")
```

### 5. 返回 NumPy 数组

```python
# 返回 numpy 而不是 tensor
embedding_np = backbone.encode(text, return_tensor=False)
print(type(embedding_np))  # <class 'numpy.ndarray'>
```

## 配置参数说明

### backbone_settings

- `type` (str): Backbone 类型，固定为 `"encoder"`
- `name` (str): 模型名称，如 `"deberta-v3-base"`
- `model_id` (str, optional): 自定义模型路径，覆盖默认映射

### encoder_settings

- `device_map` (str, default: "auto"): 设备映射
  - `"auto"`: 自动分配
  - `"cpu"`: 强制使用 CPU
  - `"cuda:0"`: 使用特定 GPU

- `pooling_mode` (str, default: "mean"): 默认池化模式
  - `"cls"`: 使用 [CLS] token 的表示
  - `"mean"`: 平均池化（考虑 attention mask）
  - `"max"`: 最大池化

### global_settings

- `device` (str): 全局设备设置
- `dtype` (str): 数据类型
  - `"float32"`: 32位浮点数（CPU 默认）
  - `"float16"`: 16位浮点数
  - `"bfloat16"`: BFloat16（推荐用于训练）

## 注意事项

1. **PyTorch 版本**: 需要 PyTorch >= 2.1
2. **CPU vs GPU**: CPU 模式会自动强制使用 FP32
3. **池化模式**: 可以在初始化时设置默认值，也可以在 `encode()` 时动态指定
4. **批量处理**: `encode()` 自动处理单文本和批量文本
5. **设备管理**: 模型和输出会自动放在正确的设备上

## 完整配置示例

```yaml
# configs/deberta_example.yaml

global_settings:
  device: "cuda:0"
  dtype: "bfloat16"
  seed: 42

backbone_settings:
  type: "encoder"
  name: "deberta-v3-base"
  encoder_settings:
    device_map: "auto"
    pooling_mode: "mean"

dataset_settings:
  type: "distill"
  name: "distill-openr1-math"
  distill_settings:
    split:
      train: 10000
      test: 1000

trainer_settings:
  type: "dl"
  name: "pytorch"
  dl_settings:
    batch_size: 32
    epochs: 10
    learning_rate: 2e-5
```

## 测试验证

运行结构验证测试：

```bash
python test_deberta_structure.py
```

这将验证：
- 类的导入和继承关系
- 注册表配置
- 方法存在性
- 基本实例化

注意：完整的模型加载测试需要正确的 PyTorch 版本。
