# 项目架构设计

## 核心流程

整个项目采用配置驱动的方式，通过三个核心 Loader 完成训练准备：

```
配置加载 → 数据加载 → 训练器加载 → 注册步骤 → 训练
   ↓           ↓           ↓           ↓         ↓
Config    Dataset     Trainer    train_step   trainer.run()
Loader     Loader      Loader    eval_step
```

### 使用示例

```python
# 1. 加载配置
from engine.configs import load_config
config = load_config("configs/my_experiment.yaml")

# 2. 加载数据集
from engine.datas import load_dataset
dataset_bundle = load_dataset(config)
# 返回: {"splits": {...}, "meta": {...}, "judge": callable}

# 3. 加载训练器
from engine.trainers import load_trainer
trainer = load_trainer(config, dataset_bundle)

# 4. 注册模型和优化器（在 methods/ 中定义）
trainer.register_model("model", model)
trainer.add_param_group("model", model.parameters())
trainer.setup_optimizers()

# 5. 注册训练/评估步骤
trainer.register_train_step(train_step)
trainer.register_eval_step(eval_step)

# 6. 执行训练
trainer.run()
```

### Methods 目录

**位置**: `methods/`

**作用**: 存放具体的训练方法实现（参考 `methods/example/`）

**内容**:
- 模型定义和初始化
- `train_step(batch, info) -> {group_name: loss_dict}` 训练步骤函数
- `eval_step(batch, info) -> metrics_dict` 评估步骤函数
- 模型注册和优化器配置逻辑

**注意**: 暂时不需要关注 Manager 相关代码，聚焦于上述核心流程即可。

---

## Trainer 设计

**位置**: `engine/trainers/impl/basic_pytorch.py`

### 核心特性
- **多优化器支持**: 通过参数组机制为不同组件配置独立优化器
- **灵活回调**: `register_train_step(fn)` 返回 `{group_name: loss_dict}`，自动处理反向传播
- **两种模式**: 完整训练(`train_full`) / Optuna搜索(`_run_optuna`)
- **完善监控**: 梯度裁剪、loss打印、ETA预估、checkpoint保存
- **模型注册**: `register_model()` 统一管理，通过info字典传递

### 使用流程
```python
trainer = load_trainer(config, dataset_bundle)
trainer.register_model("model", model)
trainer.add_param_group("model", model.parameters())
trainer.setup_optimizers()
trainer.register_train_step(train_step)
trainer.register_eval_step(eval_step)
trainer.run()
```

## Data 设计

**位置**: `engine/datas/`

### 目录结构（与 Backbone 一致）
```
engine/datas/
├── base/                    # 基类（按数据集类型）
│   ├── vqa.py              # VQA基类: BasePreparer, BsesDataset
│   ├── distill.py          # 蒸馏基类: BasePreparer, DistillDataset
│   ├── qa.py               # QA基类: BasePreparer, QADataset
│   └── __init__.py
├── impl/                    # 实现（按数据集类型）
│   ├── vqa/                # VQA数据集实现
│   │   ├── mme.py
│   │   ├── vqa_v2.py
│   │   ├── pope.py
│   │   ├── mmb.py
│   │   ├── scienceqa.py
│   │   ├── gqa.py
│   │   ├── seed_bench.py
│   │   └── __init__.py
│   ├── distill/            # 蒸馏数据集实现
│   │   ├── openr1_math.py
│   │   └── __init__.py
│   ├── qa/                 # QA数据集实现
│   │   ├── gsm8k.py
│   │   └── __init__.py
│   └── __init__.py
├── loader.py               # 注册表和加载器
└── __init__.py
```

### 配置结构（分层设计）
```yaml
# VQA 数据集
dataset_settings:
  type: "vqa"              # 数据集类型
  name: "vqa-vqav2"        # 具体数据集
  vqa_settings:            # VQA特定配置（类似 mllm_settings, dl_settings）
    split:
      train: 14000
      test: 200
    category_priority:
      enable: false
      values:
        - train: "mean"
        - test: "mean"
    fast_load_no_random: true

# 蒸馏数据集
dataset_settings:
  type: "distill"          # 数据集类型
  name: "distill-openr1-math"  # 具体数据集
  distill_settings:        # 蒸馏特定配置
    split:
      train: 10000
      test: 1000
    hf_path: "open-r1/OpenR1-Math-220k"  # HuggingFace 数据集路径（可选）
    hf_split: "train"                     # HuggingFace split（可选）

# QA 数据集
dataset_settings:
  type: "qa"               # 数据集类型
  name: "qa-gsm8k"         # 具体数据集
  qa_settings:             # QA特定配置
    split:
      train: 7000
      test: 1319
    # 注意：数据集路径等特定配置写死在各 Preparer 代码中
```

### 对比 Backbone/Trainer 结构
```
Backbone:                Trainer:                 Data:
├── type: "mllm"        ├── type: "dl"          ├── type: "vqa" / "distill" / "qa"
├── name: "llava"       ├── name: "pytorch"     ├── name: "vqa-mme" / "distill-openr1-math" / "qa-gsm8k"
└── mllm_settings:      └── dl_settings:        └── vqa_settings / distill_settings / qa_settings:
    └── device_map          └── batch_size          └── split
```

### 架构
```
# VQA 数据集
BasePreparer (base/vqa.py)
  └── VQA实现 (impl/vqa/)
      ├── MMEPreparer
      ├── VQAV2Preparer
      ├── POPEPreparer
      ├── MMBenchPreparer
      ├── ScienceQAPreparer
      ├── GQAPreparer
      └── SEEDBenchPreparer

# 蒸馏数据集
BasePreparer (base/distill.py)
  └── Distill实现 (impl/distill/)
      └── OpenR1MathPreparer

# QA 数据集
BasePreparer (base/qa.py)
  └── QA实现 (impl/qa/)
      └── GSM8KPreparer
```

### 核心功能

#### VQA 数据集
1. **统一结构**: `{image, question, answer, category?}` + 字段映射
2. **智能拆分**:
   - 支持比例/绝对数量/占位符(-1)/全量('all')
   - 类别优先分配: mean模式(均衡) / origin模式(保留原始分布)
3. **Judge函数**: 自定义答案判定逻辑，返回 `{correct, total, accuracy}`

#### 蒸馏数据集
1. **统一格式**:
   ```python
   {
       "instruction": str,           # 问题描述
       "responses": {                # 多个教师模型的生成
           "model-name": {           # 模型标识
               "messages": [...],    # 对话历史
               "rewards": {          # 奖励信号
                   "reward_key": float
               }
           }
       },
       "metadata": {...}             # 数据集特定元信息
   }
   ```
2. **智能选择**: 多个生成时根据 reward 自动选择最优响应
3. **数据拆分**: 支持比例/绝对数量/占位符(-1)/全量('all')
4. **注册机制**: `DATASET_REGISTRY` 按类型组织，格式 `"type-name"`
5. **向后兼容**: 自动兼容旧的扁平配置结构

#### QA 数据集
1. **统一结构**: `{question, answer, final_answer, source_split}` + 答案提取
2. **智能拆分**:
   - 支持比例/绝对数量/占位符(-1)/全量('all')
   - 自动从原始 split (train/test) 中合并和重新分配
3. **答案提取**:
   - GSM8K 格式: `#### answer`
   - MATH 格式: `\boxed{answer}`
   - 可扩展其他格式
4. **Judge函数**: 使用 math_verify 或数值比较，返回 `{correct, total, accuracy}`
   - 支持单条判定和批量判定
   - 容错处理（逗号、美元符号、小数点等）

### 扩展新数据集

#### VQA 数据集示例
1. **在对应类型目录创建实现**:
```python
# engine/datas/impl/vqa/my_dataset.py
from ...base.vqa import BasePreparer, BsesDataset

class MyPreparer(BasePreparer):
    def _load_all(self):
        # 加载数据
        return samples

    def get(self):
        samples = self._load_all()
        splits, placeholder = self.split_from_single(samples)
        meta = self.build_meta(...)
        judge = self._build_judge(...)
        return {"splits": splits, "meta": meta, "judge": judge}
```

2. **在类型目录__init__.py导出**:
```python
# engine/datas/impl/vqa/__init__.py
from .my_dataset import MyPreparer  # noqa: F401
```

3. **在loader.py注册**:
```python
# engine/datas/loader.py
from .impl.vqa import MyPreparer

DATASET_REGISTRY = {
    "vqa-mydataset": MyPreparer,  # 格式: "type-name"
    ...
}
```

4. **配置文件使用分层结构**:
```yaml
dataset_settings:
  type: "vqa"
  name: "vqa-mydataset"
  vqa_settings:
    split:
      train: 1000
      test: 200
```

#### 蒸馏数据集示例
1. **在对应类型目录创建实现**:
```python
# engine/datas/impl/distill/my_distill.py
from ...base.distill import BasePreparer, DistillDataset

class MyDistillPreparer(BasePreparer):
    def _convert_to_distill_format(self, raw_item):
        # 转换为统一格式
        return {
            "instruction": raw_item["problem"],
            "responses": {
                "teacher-model": {
                    "messages": raw_item["messages"],
                    "rewards": {"score": raw_item["score"]}
                }
            },
            "metadata": {...}
        }

    def _load_all(self):
        # 加载并转换数据
        raw_data = load_your_data()
        return [self._convert_to_distill_format(item) for item in raw_data]

    def get(self):
        samples = self._load_all()
        splits, placeholder = self.split_samples(samples)
        meta = self.build_meta(samples, splits, placeholder)
        self.base_report(meta)
        return {"splits": splits, "meta": meta, "judge": None}
```

2. **在类型目录__init__.py导出**:
```python
# engine/datas/impl/distill/__init__.py
from .my_distill import MyDistillPreparer  # noqa: F401
```

3. **在loader.py注册**:
```python
# engine/datas/loader.py
from .impl.distill import MyDistillPreparer

DATASET_REGISTRY = {
    "distill-mydataset": MyDistillPreparer,  # 格式: "type-name"
    ...
}
```

4. **配置文件使用分层结构**:
```yaml
dataset_settings:
  type: "distill"
  name: "distill-mydataset"
  distill_settings:
    split:
      train: 10000
      test: 1000
```

#### QA 数据集示例
1. **在对应类型目录创建实现**:
```python
# engine/datas/impl/qa/my_qa.py
from ...base.qa import BasePreparer, QADataset

class MyQAPreparer(BasePreparer):
    def _load_all(self):
        # 从 HuggingFace 或其他源加载
        dataset = load_dataset("my-org/my-dataset")
        samples = []
        for item in dataset["train"]:
            sample = {
                "question": item["question"],
                "answer": item["answer"],
                "final_answer": self.extract_answer(item["answer"], format_type="gsm8k"),
                "source_split": "train",
            }
            samples.append(sample)
        return samples

    def get(self):
        samples = self._load_all()
        splits, placeholder = self.split_samples(samples)
        meta = self.build_meta(samples, splits, placeholder)
        judge = self._build_judge(meta, splits)
        self.base_report(meta)
        return {"splits": splits, "meta": meta, "judge": judge}

    def _build_judge(self, meta, splits):
        # 使用 math_verify 或自定义判定逻辑
        from math_verify import verify_math_answer
        def judge(pred, ref, sample=None):
            # 判定逻辑
            ...
        return judge
```

2. **在类型目录__init__.py导出**:
```python
# engine/datas/impl/qa/__init__.py
from .my_qa import MyQAPreparer  # noqa: F401
```

3. **在loader.py注册**:
```python
# engine/datas/loader.py
from .impl.qa import MyQAPreparer

DATASET_REGISTRY = {
    "qa-mydataset": MyQAPreparer,  # 格式: "type-name"
    ...
}
```

4. **配置文件使用分层结构**:
```yaml
dataset_settings:
  type: "qa"
  name: "qa-mydataset"
  qa_settings:
    split:
      train: 5000
      test: 1000
    hf_path: "my-org/my-dataset"
    extract_final_answer: true
```

## 训练流程

**位置**: `main.py`

### Manager模式
```
preload_fn() → 预加载Backbone+Dataset (冻结Backbone)
    ↓
run_fn() → 创建Trainer + 模型 + 注册回调
    ↓
trainer.run() → 执行训练
```

### 设计优势
- 分离预加载避免重复加载
- 配置驱动，统一管理超参数
- 支持多组件训练(token_merger, layer_pruners, discriminator等)
