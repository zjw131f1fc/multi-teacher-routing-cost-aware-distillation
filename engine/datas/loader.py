"""数据集加载入口 (registry 机制)。

当前支持的数据集类型:
    - VQA (Visual Question Answering): MME, VQAv2, POPE, MMBench, ScienceQA, GQA, SEED-Bench
    - Distill (蒸馏训练): OpenR1-Math
    - QA (Question Answering): GSM8K

配置关键字段 (Config.dataset_settings):
    type: str
        数据集类型，例如 'vqa', 'distill', 或 'qa'。
    name: str
        注册表中的数据集键，格式: 'type-name'，例如 'vqa-mme', 'distill-openr1-math', 或 'qa-gsm8k'。
    <type>_settings: Dict
        特定类型的配置。

        VQA 数据集 ('vqa_settings') 包含：
            split: Dict[str, float|int]
                目标 split 及其大小。float 表示总样本比例 (0<val<=1)，int 表示绝对数量 (>=0)。
                计算逻辑在具体 Preparer 中完成；不足时会截断。
            category_priority: Dict
                类别优先级配置，包含:
                    'enable': bool
                        是否启用类别优先级分配。
                    'values': List[Dict[str,str]]
                        按优先级顺序的类别分配模式列表。
                        形式: [ {split: mode}, {split: mode}, ... ]，mode 支持:
                            'mean':   尽量均衡该 split 的各类别数量。
                            'origin': 按当前剩余样本的类别比例进行分配。
                        未出现的 split 默认使用 'origin'。重复 split 的后续项被忽略。
            fast_load_no_random: bool
                是否以快速模式加载数据集（不随机打乱）。

        蒸馏数据集 ('distill_settings') 包含：
            split: Dict[str, float|int]
                目标 split 及其大小（支持 float/int/-1/'all'）。

        QA 数据集 ('qa_settings') 包含：
            split: Dict[str, float|int]
                目标 split 及其大小（支持 float/int/-1/'all'）。

            注意：数据集路径等特定配置写死在各 Preparer 代码中。

配置示例（分层结构，与 backbone/trainer 一致）:
    # 单数据集模式 - VQA 数据集
    dataset_settings:
      type: "vqa"
      name: "vqa-vqav2"
      vqa_settings:
        split:
          train: 14000
          test: 200
        category_priority:
          enable: false
          values:
            - train: "mean"
            - test: "mean"
        fast_load_no_random: true

    # 单数据集模式 - 蒸馏数据集
    dataset_settings:
      type: "distill"
      name: "distill-openr1-math"
      distill_settings:
        split:
          train: 10000
          test: 1000

    # 单数据集模式 - QA 数据集
    dataset_settings:
      type: "qa"
      name: "qa-gsm8k"
      qa_settings:
        split:
          train: 7000
          test: 1319

    # 多数据集模式（dataset_settings 为列表）
    dataset_settings:
      - type: "distill"
        name: "distill-openr1-math"
        distill_settings:
          split:
            train: 'all'
            test: -1
      - type: "qa"
        name: "qa-gsm8k"
        qa_settings:
          split:
            train: -1
            test: 'all'

扩展一个新数据集步骤:
    1. 在 `datas/base/<type>.py` 创建基类（如果类型不存在）。
    2. 在 `datas/impl/<type>/` 下创建实现 (例如 mydataset.py) 并提供 Preparer 接口
       (构造接受 config, 提供 get() 返回 {splits, meta, judge})。
    3. 在 `datas/impl/<type>/__init__.py` 中导出 Preparer 类。
    4. 在这里导入实现中的 Preparer 类并加入 DATASET_REGISTRY。
    5. 在 Config.dataset_settings 中设置 type, name 与 <type>_settings。

注意: 这里不做 try/except 包装，错误将直接抛出以便快速发现配置问题。
"""

from typing import Dict, Type, Any, List, Optional

# VQA datasets
from .impl.vqa import (
    MMEPreparer,
    VQAV2Preparer,
    POPEPreparer,
    MMBenchPreparer,
    ScienceQAPreparer,
    GQAPreparer,
    SEEDBenchPreparer,
)

# Distill datasets
from .impl.distill import (
    OpenR1MathPreparer,
    MixedQAPreparer,
)

# QA datasets
from .impl.qa import (
    GSM8KPreparer,
)

# Registry: 所有支持的数据集
# 格式: "type-name": PreparerClass
DATASET_REGISTRY: Dict[str, Type[Any]] = {
    # VQA datasets
    "vqa-mme": MMEPreparer,
    "vqa-vqav2": VQAV2Preparer,
    "vqa-pope": POPEPreparer,
    "vqa-mmb": MMBenchPreparer,
    "vqa-sqa": ScienceQAPreparer,
    "vqa-gqa": GQAPreparer,
    "vqa-seed": SEEDBenchPreparer,

    # Distill datasets
    "distill-openr1-math": OpenR1MathPreparer,
    "distill-mixed-qa": MixedQAPreparer,  # 蒸馏训练数据 + GSM8K 测试数据

    # QA datasets
    "qa-gsm8k": GSM8KPreparer,

    # 未来可以添加其他类型:
    # "captioning-coco": COCOCaptionPreparer,
    # "detection-coco": COCODetectionPreparer,
    # "qa-math": MATHPreparer,
    # etc.
}


def load_dataset(config: Optional[dict] = None) -> Any:
    """根据名称加载并实例化数据集。

    支持两种模式:
    1. 单数据集: dataset_settings 是 dict，返回单个 bundle
       返回: {'splits':..., 'meta':..., 'judge': callable}

    2. 多数据集: dataset_settings 是 list，返回多个 bundle 的列表
       返回: [
           {'splits':..., 'meta':..., 'judge': callable},
           {'splits':..., 'meta':..., 'judge': callable},
           ...
       ]

    judge: 数据集相关答案判定函数，支持:
      - 单条: judge(pred_str, ref_str, sample) -> {correct,total,accuracy}
      - 批量: judge([pred...], [ref...]) -> {correct,total,accuracy}
    """
    # 兼容 dict 和 SimpleNamespace
    if isinstance(config, dict):
        dataset_settings = config["dataset_settings"]
    else:
        dataset_settings = config.dataset_settings

    # 检查是否为多数据集配置（列表）
    if isinstance(dataset_settings, list):
        # 多数据集模式：依次加载每个数据集
        bundles = []
        for idx, ds_config in enumerate(dataset_settings):
            # 创建临时配置，只包含当前数据集的配置
            temp_config = _create_temp_config(config, ds_config)

            # 加载单个数据集
            bundle = _load_single_dataset(temp_config)
            bundles.append(bundle)

        return bundles
    else:
        # 单数据集模式
        return _load_single_dataset(config)


def _create_temp_config(original_config, dataset_settings_item):
    """创建临时配置，用于加载单个数据集。"""
    from types import SimpleNamespace

    if isinstance(original_config, dict):
        # 深拷贝原始配置
        import copy
        temp_config = copy.deepcopy(original_config)
        temp_config["dataset_settings"] = dataset_settings_item

        # 转换为 SimpleNamespace 以兼容基类
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_namespace(item) for item in d]
            else:
                return d

        temp_config = dict_to_namespace(temp_config)
    else:
        # SimpleNamespace
        import copy
        temp_config = copy.deepcopy(original_config)
        temp_config.dataset_settings = dict_to_namespace(dataset_settings_item) if isinstance(dataset_settings_item, dict) else dataset_settings_item

    return temp_config


def _load_single_dataset(config):
    """加载单个数据集，返回 bundle。"""
    # 兼容 dict 和 SimpleNamespace
    if isinstance(config, dict):
        name = config["dataset_settings"]["name"]
    else:
        name = config.dataset_settings.name

    if name not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' is not registered. 已注册: {list_datasets()}")

    dataset_cls = DATASET_REGISTRY[name]
    return dataset_cls(config=config).get()


def list_datasets() -> List[str]:
	"""返回当前已注册的数据集名称列表。"""
	return list(DATASET_REGISTRY.keys())

