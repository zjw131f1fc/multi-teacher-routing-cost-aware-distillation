"""QA (Question Answering) Dataset 基类。

用于纯文本问答数据集，如 GSM8K, MATH 等。
具体实现应放在 impl/qa/ 中。
"""

from typing import List, Dict, Any, Tuple
import random


class QADataset:
    """简单的 Dataset 包装类，用于包装样本列表。"""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def to_list(self):
        return self.samples


class BasePreparer:
    """QA 数据集基类，提供简单的 split 功能和答案提取。

    子类需要：
      - 实现数据加载（从 HuggingFace 或其他源）
      - 调用 split_samples() 进行拆分
      - 调用 build_meta() 构建元信息
      - 实现 _build_judge() 构建判定函数
      - 通过 get() 返回 {"splits": ..., "meta": ..., "judge": callable}

    支持的 split 配置：
      - float (0<v<=1): 按比例分配
      - int (>=0): 绝对数量
      - int (-1): 占位符（仅引用一个样本）
      - str ('all'): 全部数据
    """

    def __init__(self, config):
        self.config = config
        ds_cfg = self.config.dataset_settings

        # 支持分层配置: dataset_settings.qa_settings 或直接从 dataset_settings 读取（向后兼容）
        # 兼容 dict 和 SimpleNamespace
        if isinstance(ds_cfg, dict):
            has_qa_settings = "qa_settings" in ds_cfg
            qa_cfg = ds_cfg["qa_settings"] if has_qa_settings else ds_cfg
        else:
            # SimpleNamespace
            has_qa_settings = hasattr(ds_cfg, "qa_settings")
            qa_cfg = ds_cfg.qa_settings if has_qa_settings else ds_cfg

        # 获取 split 配置（兼容两种类型）
        if isinstance(qa_cfg, dict):
            self.split_cfg: Dict[str, Any] = qa_cfg.get("split", {})
            self.extract_final_answer: bool = qa_cfg.get("extract_final_answer", True)
        else:
            split_obj = getattr(qa_cfg, "split", {})
            # 如果是 SimpleNamespace，转为 dict
            if hasattr(split_obj, '__dict__'):
                self.split_cfg: Dict[str, Any] = vars(split_obj)
            else:
                self.split_cfg: Dict[str, Any] = split_obj
            self.extract_final_answer: bool = getattr(qa_cfg, "extract_final_answer", True)

        # 获取 seed（兼容两种类型）
        global_settings = self.config.global_settings
        if isinstance(global_settings, dict):
            self.seed: int = global_settings["seed"]
        else:
            self.seed: int = global_settings.seed

    def compute_split_target_sizes(self, total: int) -> Dict[str, int]:
        """根据配置计算每个 split 的目标大小。"""
        targets: Dict[str, int] = {}
        for split_name, v in self.split_cfg.items():
            if isinstance(v, float) and 0 < v <= 1:
                targets[split_name] = int(total * v)
            elif isinstance(v, int) and v >= 0:
                targets[split_name] = v if v <= total else total
            elif isinstance(v, int) and v == -1:
                targets[split_name] = 1  # 占位仅需引用一个样本
            elif isinstance(v, str) and v == 'all':
                targets[split_name] = total
            else:
                targets[split_name] = 0
        return targets

    def split_samples(self, samples: List[Dict[str, Any]]) -> Tuple[Dict[str, QADataset], List[str]]:
        """从单一样本列表中按配置拆分成多个 split。

        Args:
            samples: 原始样本列表

        Returns:
            (split_datasets, placeholder_splits)
            - split_datasets: {split_name: QADataset}
            - placeholder_splits: 占位或全量 split 的名称列表
        """
        total = len(samples)
        random.seed(self.seed)
        shuffled = samples[:]
        random.shuffle(shuffled)

        targets = self.compute_split_target_sizes(total)
        result: Dict[str, QADataset] = {}
        ptr = 0
        placeholder_splits: List[str] = []

        for name, size in targets.items():
            raw_v = self.split_cfg[name]
            is_placeholder = isinstance(raw_v, int) and raw_v == -1
            is_all = isinstance(raw_v, str) and raw_v == 'all'

            end = ptr + size
            if end > total:
                end = total

            if is_all:
                # 'all' 模式：引用全部数据
                subset = samples[:]
                placeholder_splits.append(name)
            else:
                subset = shuffled[ptr:end]
                if is_placeholder:
                    placeholder_splits.append(name)
                else:
                    ptr = end

            result[name] = QADataset(subset)

        return result, placeholder_splits

    def extract_answer(self, answer_text: str, format_type: str = "gsm8k") -> str:
        """提取最终答案。

        Args:
            answer_text: 完整答案文本
            format_type: 答案格式类型
                - "gsm8k": GSM8K 格式 (#### answer)
                - "math": MATH 格式 (\boxed{answer})
                - "plain": 直接返回原文本

        Returns:
            提取的最终答案
        """
        if not self.extract_final_answer:
            return answer_text.strip()

        if format_type == "gsm8k":
            # GSM8K 格式: #### 3
            if "####" in answer_text:
                return answer_text.split("####")[-1].strip()
        elif format_type == "math":
            # MATH 格式: \boxed{answer}
            import re
            match = re.search(r'\\boxed\{([^}]+)\}', answer_text)
            if match:
                return match.group(1).strip()

        # 默认返回去除空白的原文本
        return answer_text.strip()

    def build_meta(self, all_samples: List[Dict[str, Any]], split_datasets: Dict[str, QADataset], placeholder_splits: List[str]) -> Dict[str, Any]:
        """构建数据集元信息。"""
        split_details: Dict[str, Dict[str, str]] = {}
        for name in split_datasets.keys():
            raw_v = self.split_cfg.get(name)
            is_placeholder = isinstance(raw_v, int) and raw_v == -1
            is_all = isinstance(raw_v, str) and raw_v == 'all'

            if is_all:
                split_type = 'all'
            elif is_placeholder:
                split_type = 'placeholder'
            else:
                split_type = 'normal'

            split_details[name] = {"type": split_type}

        meta = {
            "total": len(all_samples),
            "split_sizes": {n: len(ds) for n, ds in split_datasets.items()},
            "placeholder_splits": placeholder_splits,
            "split_details": split_details,
        }
        return meta

    def base_report(self, meta: Dict[str, Any]):
        """打印数据集加载报告。"""
        logger = getattr(self.config, "logger", None)
        if logger is None:
            return

        logger.info(f"[Dataset] Total: {meta['total']} | Splits: {meta['split_sizes']}")

        if "split_details" in meta:
            details = meta["split_details"]
            parts = []
            for name in sorted(details.keys()):
                d = details[name]
                parts.append(f"{name}({d['type']})")
            logger.info("[Dataset] Split Details: " + ", ".join(parts))

    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, QADataset]):
        """构建 judge 函数（子类需要实现）。"""
        raise NotImplementedError
