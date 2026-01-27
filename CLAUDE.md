# RL-Based Cost-Aware Teacher Selection

基于强化学习的成本感知教师选择系统，用于 LLM 知识蒸馏中的合成数据生成。

---

## 项目结构

```
/
├── main.py              # 训练入口
├── configs/             # 配置加载
│   ├── __init__.py
│   └── loader.py        # load_config(path) -> dict
├── datas/               # 数据加载
│   ├── __init__.py
│   └── loader.py        # BaseDataset, load_dataset(config)
├── models/              # 模型定义
│   ├── __init__.py
│   ├── student.py       # StudentModel, load_student(config)
│   └── teacher.py       # TeacherModel, load_teacher(config, name)
├── teachers/            # 教师池管理
│   ├── __init__.py
│   └── loader.py        # TeacherPool(config, dataset, teachers)
└── rl/                  # 强化学习组件
    ├── __init__.py
    ├── policy.py        # PolicyNetwork
    ├── gradient.py      # compute_gradient, compute_validation_gradient
    ├── reward.py        # compute_reward
    └── env.py           # TeacherSelectionEnv (Gymnasium)
```

---

## 模块说明

### configs/
- `load_config(path)`: 加载 YAML 配置文件，返回字典

### datas/
- `BaseDataset`: PyTorch Dataset，存储 `{instruction, gt, ...}`
- `load_dataset(config)`: 根据配置加载数据集

### models/
- `StudentModel`: 包装 HuggingFace CausalLM，`forward(input_ids, labels) -> loss`
- `TeacherModel`: 包装 HuggingFace CausalLM，`generate(instruction) -> str`
- `load_student(config)`, `load_teachers(config)`: 加载函数

### teachers/
- `TeacherPool`: 管理教师输出获取和缓存
  - 持有 config、dataset、teachers
  - `get_output(instruction, teacher_name) -> str`
  - 优先级：cache → dataset → 调用模型 → 存入 cache

### rl/
- `PolicyNetwork`: 冻结 Encoder + MLP，输出教师选择概率
- `compute_gradient`: 计算样本在 student 最后 N 层的梯度
- `compute_reward`: `R = cosine(g_sample, g_val) * log(1 + ||g||) - λ * cost`
- `TeacherSelectionEnv`: Gymnasium 环境，集成 SB3 PPO

---

## 核心流程

```
config = load_config("config.yaml")
dataset = load_dataset(config)
student = load_student(config)
teachers = load_teachers(config)
teacher_pool = TeacherPool(config, dataset, teachers)
env = TeacherSelectionEnv(instructions, student, teacher_pool, ...)
model = PPO("MultiInputPolicy", env, ...)
model.learn(total_timesteps=10000)
```

---

## 配置示例

```yaml
# 数据集
dataset:
  name: "my_dataset"
  path: "./data/train.json"

# Student 模型
student:
  model_name: "Qwen/Qwen2.5-0.5B"
  device: "cuda"

# Teacher 模型
teachers:
  gpt-4:
    model_name: "openai/gpt-4"
    device: "cuda"
    max_new_tokens: 512
  llama-70b:
    model_name: "meta-llama/Llama-2-70b"
    device: "cuda"

# Teacher 成本
teacher_costs:
  gpt-4: 1.0
  llama-70b: 0.5

# RL 设置
rl:
  val_samples: 200
  lambda_cost: 0.05
  student_update_freq: 100
  last_n_layers: 3

# PPO 设置
ppo:
  learning_rate: 3e-4
  n_steps: 128
  batch_size: 64

# 训练设置
training:
  total_timesteps: 10000
  save_path: "./output/model"

# 缓存目录
cache_dir: "./cache"
```

---

## 工作方式

与 Claude 协作时，采用 **"先说想法，等许可再动手"** 的模式：

1. 用户提出需求
2. Claude 说明想法（位置、实现方案、接口设计）
3. 用户许可后，Claude 执行

### Few-shot 示例

**示例 1：添加新模块**

```
用户: 我需要一个评估模块，评估 student 在测试集上的表现

Claude: 我的想法：
1. 位置: eval/evaluator.py
2. 功能:
   - evaluate(student, test_loader) -> metrics_dict
   - 计算 loss、accuracy 等指标
3. 接口: Evaluator 类或纯函数

是否许可？

用户: 许可

Claude: [开始写代码]
```

**示例 2：修改现有代码**

```
用户: TeacherPool 的缓存要支持过期时间

Claude: 我的想法：
1. 修改 teachers/loader.py
2. cache 结构改为 {key: {value, timestamp}}
3. _get_from_cache 检查是否过期
4. 新增 config 参数 cache_ttl (秒)

是否许可？

用户: 许可

Claude: [开始修改代码]
```

**示例 3：需要澄清**

```
用户: 加个日志功能

Claude: 我的想法：
1. 使用 Python logging 模块
2. 位置: utils/logger.py
3. 各模块导入使用

但需要确认：
- 日志级别由 config 控制还是写死？
- 是否需要输出到文件？

用户: config 控制，输出到文件和终端

Claude: 明白，是否许可开始？

用户: 许可

Claude: [开始写代码]
```
