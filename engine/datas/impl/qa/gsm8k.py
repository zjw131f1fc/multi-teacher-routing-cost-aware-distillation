"""GSM8K 数据集准备器。

数据来源: HuggingFace datasets (openai/gsm8k)

原始字段:
    - question: 问题文本
    - answer: 完整答案（包含推理步骤和最终答案，格式: "推理过程\n#### 最终答案"）

转换后的统一格式:
    {
        "question": str,           # 问题文本
        "answer": str,             # 完整答案（包含推理过程）
        "final_answer": str,       # 最终答案（从 #### 后提取）
        "source_split": str,       # 原始 split（train/test）
    }

使用 math_verify 作为 judge 函数来验证答案正确性。
"""

from typing import Dict, Any, List
from datasets import load_dataset
from ...base.qa import BasePreparer, QADataset


class GSM8KDataset(QADataset):
    pass


class GSM8KPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)

        # HuggingFace 数据集路径（写死在代码中）
        self.hf_path = "openai/gsm8k"
        self.hf_config = "main"
        self.load_splits = ["train", "test"]

    def _load_all(self) -> List[Dict[str, Any]]:
        """从 HuggingFace 加载数据并转换格式。"""
        logger = getattr(self.config, "logger", None)
        if logger:
            logger.info(f"[GSM8K] Loading from HuggingFace: {self.hf_path} (config={self.hf_config})")

        # 加载 HuggingFace 数据集
        dataset = load_dataset(self.hf_path, self.hf_config)

        samples = []
        for split_name in self.load_splits:
            if split_name not in dataset:
                if logger:
                    logger.warning(f"[GSM8K] Split '{split_name}' not found in dataset, skipping")
                continue

            for item in dataset[split_name]:
                sample = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "final_answer": self.extract_answer(item["answer"], format_type="gsm8k"),
                    "source_split": split_name,
                }
                samples.append(sample)

        if logger:
            logger.info(f"[GSM8K] Loaded {len(samples)} samples from splits: {self.load_splits}")

        return samples

    def get(self):
        """加载、拆分并返回数据集 bundle。"""
        # 加载所有样本
        all_samples = self._load_all()

        # 使用基类的 split_samples 进行拆分
        split_datasets, placeholder_splits = self.split_samples(all_samples)

        # 构建元信息
        meta = self.build_meta(all_samples, split_datasets, placeholder_splits)

        # 构建 judge（使用 math_verify）
        judge = self._build_judge(meta, split_datasets)

        # 打印报告
        self.base_report(meta)
        self._print_split_report(split_datasets)

        # 返回 bundle
        return {
            "splits": split_datasets,
            "meta": meta,
            "judge": judge
        }

    def _print_split_report(self, split_datasets: Dict[str, QADataset]):
        """打印每个 split 的详细信息。"""
        logger = getattr(self.config, "logger", None)
        if logger is None:
            return

        for split_name, ds in split_datasets.items():
            # 统计原始 split 分布
            source_splits: Dict[str, int] = {}
            for i in range(len(ds)):
                source = ds[i].get("source_split", "unknown")
                source_splits[source] = source_splits.get(source, 0) + 1

            source_info = ", ".join(f"{src}:{cnt}" for src, cnt in sorted(source_splits.items()))
            logger.info(f"[GSM8K] Split '{split_name}': {len(ds)} samples (sources: {source_info})")

    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, QADataset]):
        """构建使用 math_verify 的 judge 函数。

        math_verify 会验证数学答案的正确性，支持各种格式的数字表示。
        """
        try:
            from math_verify import verify_math_answer
            use_math_verify = True
        except ImportError:
            use_math_verify = False
            logger = getattr(self.config, "logger", None)
            if logger:
                logger.warning("[GSM8K] math_verify not installed, falling back to simple numeric comparison")

        def normalize_number(s: str) -> float:
            """提取并规范化数字（备用方案）。"""
            import re
            if s is None:
                return None
            # 移除逗号、空格、美元符号等
            s = str(s).replace(",", "").replace("$", "").strip()
            # 提取第一个数字（支持负数和小数）
            match = re.search(r'-?\d+\.?\d*', s)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
            return None

        def judge(pred, ref, sample=None):
            """判定答案是否正确。

            Args:
                pred: 预测答案（字符串或字符串列表）
                ref: 参考答案（字符串或字符串列表）
                sample: 原始样本（可选）

            Returns:
                {"correct": int, "total": int, "accuracy": float}
            """
            # 批量模式
            if isinstance(pred, list):
                if not isinstance(ref, list):
                    raise TypeError("批量判定时 ref 也应为列表")
                total = len(pred)
                if len(ref) != total:
                    raise ValueError("pred/ref 长度不一致")

                correct = 0
                for p, r in zip(pred, ref):
                    if use_math_verify:
                        # 使用 math_verify
                        is_correct = verify_math_answer(str(p), str(r))
                    else:
                        # 备用：数值比较
                        p_num = normalize_number(str(p))
                        r_num = normalize_number(str(r))
                        is_correct = (p_num is not None and r_num is not None and abs(p_num - r_num) < 1e-5)

                    if is_correct:
                        correct += 1

                return {"correct": correct, "total": total, "accuracy": correct / total if total > 0 else 0.0}

            # 单条模式
            if use_math_verify:
                is_correct = verify_math_answer(str(pred), str(ref))
            else:
                p_num = normalize_number(str(pred))
                r_num = normalize_number(str(ref))
                is_correct = (p_num is not None and r_num is not None and abs(p_num - r_num) < 1e-5)

            return {"correct": 1 if is_correct else 0, "total": 1, "accuracy": 1.0 if is_correct else 0.0}

        return judge
