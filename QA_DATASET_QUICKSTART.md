# QA 数据集快速上手

## 🚀 5 分钟快速开始

### 1. 配置文件
```yaml
# configs/my_gsm8k.yaml
dataset_settings:
  type: "qa"
  name: "qa-gsm8k"
  qa_settings:
    split:
      train: 7000
      test: 1319
    hf_path: "openai/gsm8k"
    extract_final_answer: true
```

### 2. 加载数据集
```python
from engine.datas import load_dataset
from engine.configs import load_config

config = load_config("configs/my_gsm8k.yaml")
dataset_bundle = load_dataset(config)

# 访问数据
train_ds = dataset_bundle["splits"]["train"]
test_ds = dataset_bundle["splits"]["test"]
judge = dataset_bundle["judge"]

# 样本示例
sample = train_ds[0]
print(sample["question"])      # 问题
print(sample["final_answer"])  # 最终答案
```

### 3. 使用 Judge 评估
```python
# 单条评估
result = judge("3", "3")
# {"correct": 1, "total": 1, "accuracy": 1.0}

# 批量评估
predictions = ["1", "2", "3"]
references = ["1", "5", "3"]
result = judge(predictions, references)
# {"correct": 2, "total": 3, "accuracy": 0.6667}
```

---

## 📊 数据格式

### 样本结构
```python
{
    "question": "A robe takes 2 bolts of blue fiber...",
    "answer": "It takes 2/2=1 bolt...\n#### 3",
    "final_answer": "3",
    "source_split": "train"
}
```

---

## 🎯 Judge 函数特性

### 容错处理
```python
judge("3", "3")          # ✅ 正确
judge("3.0", "3")        # ✅ 正确
judge("$3", "3")         # ✅ 正确
judge("3,000", "3000")   # ✅ 正确
judge("3.14", "3")       # ❌ 错误
```

### 支持格式
- 整数: `3`, `-5`
- 小数: `3.14`, `0.5`
- 带符号: `$100`, `-$50`
- 带逗号: `1,000`, `10,000`

---

## ⚙️ 配置选项

```yaml
qa_settings:
  # 必需
  split:
    train: 7000        # 或 0.8 (比例) 或 'all' (全部) 或 -1 (占位)
    test: 1319
  
  # 可选
  hf_path: "openai/gsm8k"        # HF 数据集路径（默认值由 Preparer 提供）
  hf_config: "main"               # HF 配置名（可选）
  extract_final_answer: true      # 是否提取最终答案（默认 true）
  load_splits: ["train", "test"]  # 从 HF 加载的原始 splits
```

---

## 🔧 添加新 QA 数据集

### 最小实现（4 步）

1️⃣ **创建 Preparer**
```python
# engine/datas/impl/qa/my_dataset.py
from ...base.qa import BasePreparer, QADataset

class MyDatasetPreparer(BasePreparer):
    def _load_all(self):
        # 加载数据
        return samples
    
    def get(self):
        samples = self._load_all()
        splits, ph = self.split_samples(samples)
        meta = self.build_meta(samples, splits, ph)
        judge = self._build_judge(meta, splits)
        return {"splits": splits, "meta": meta, "judge": judge}
    
    def _build_judge(self, meta, splits):
        # 返回 judge 函数
        def judge(pred, ref, sample=None):
            ...
        return judge
```

2️⃣ **导出**
```python
# engine/datas/impl/qa/__init__.py
from .my_dataset import MyDatasetPreparer  # noqa: F401
```

3️⃣ **注册**
```python
# engine/datas/loader.py
DATASET_REGISTRY = {
    "qa-mydataset": MyDatasetPreparer,
}
```

4️⃣ **使用**
```yaml
dataset_settings:
  type: "qa"
  name: "qa-mydataset"
```

---

## 📝 常见问题

### Q: 如何使用 math_verify？
A: 安装 `pip install math-verify`，GSM8K Preparer 会自动使用。如果未安装，会自动降级到数值比较。

### Q: 如何提取不同格式的答案？
A: 在基类中调用 `extract_answer(text, format_type="gsm8k")` 或 `format_type="math"`。

### Q: 如何添加自定义判定逻辑？
A: 在子类的 `_build_judge()` 方法中实现自定义逻辑。

### Q: 支持哪些 split 配置？
A: 
- 比例: `0.8` (80%)
- 绝对数: `7000`
- 全部: `'all'`
- 占位: `-1`

---

## 🎓 示例：MATH 数据集

```python
# engine/datas/impl/qa/math.py
class MATHPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        self.hf_path = "hendrycks/math"
    
    def _load_all(self):
        dataset = load_dataset(self.hf_path, split="train")
        samples = []
        for item in dataset:
            samples.append({
                "question": item["problem"],
                "answer": item["solution"],
                "final_answer": self.extract_answer(
                    item["solution"], 
                    format_type="math"  # \boxed{} 格式
                ),
                "source_split": "train",
            })
        return samples
```

注册后即可使用：
```yaml
dataset_settings:
  type: "qa"
  name: "qa-math"
  qa_settings:
    split:
      train: 5000
      test: 1000
```

---

## 🔍 调试技巧

### 检查数据集是否注册
```python
from engine.datas.loader import DATASET_REGISTRY
print([k for k in DATASET_REGISTRY.keys() if k.startswith("qa-")])
# ['qa-gsm8k', ...]
```

### 检查样本格式
```python
sample = train_ds[0]
print(f"Keys: {list(sample.keys())}")
print(f"Question: {sample['question'][:100]}...")
print(f"Final Answer: {sample['final_answer']}")
```

### 测试 Judge 函数
```python
# 测试各种边界情况
test_cases = [
    ("3", "3", True),
    ("3.0", "3", True),
    ("$3", "3", True),
    ("3.14", "3", False),
]

for pred, ref, expected in test_cases:
    result = judge(pred, ref)
    is_correct = result["correct"] == 1
    assert is_correct == expected, f"Failed: {pred} vs {ref}"
```

---

## 📚 更多资源

- 完整实现文档: [QA_DATASET_IMPLEMENTATION.md](QA_DATASET_IMPLEMENTATION.md)
- 项目架构文档: [CLAUDE.md](CLAUDE.md)
- 配置示例: [configs/example_qa_gsm8k.yaml](configs/example_qa_gsm8k.yaml)
- 离线测试: `python test_gsm8k_offline.py`
