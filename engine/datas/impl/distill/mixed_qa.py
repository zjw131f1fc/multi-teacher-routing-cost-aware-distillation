"""混合数据集: 蒸馏数据 + QA 测试数据

用途: 将蒸馏训练数据（如 OpenR1-Math）与 QA 测试数据（如 GSM8K）拼接
使得蒸馏训练时可以用真实的 QA 数据集进行评估

数据格式:
- train split: 来自蒸馏数据集
- test split: 来自 QA 数据集（转换为蒸馏格式）
"""

from typing import Dict, Any, List
from ...base.distill import BasePreparer, DistillDataset
from ..qa.gsm8k import GSM8KPreparer


class MixedQAPreparer(BasePreparer):
    """混合数据集准备器: 蒸馏数据 + QA 测试数据"""

    def __init__(self, config):
        super().__init__(config)

        # 从配置中获取蒸馏数据集和 QA 数据集的配置
        ds_cfg = self.config.dataset_settings

        # 蒸馏数据集配置（写死在代码中）
        self.distill_hf_path = "open-r1/OpenR1-Math-220k"
        self.distill_hf_split = "train"

        # QA 数据集使用 GSM8K（通过 GSM8KPreparer 加载）
        # 无需额外配置，GSM8KPreparer 会自动处理

    def _convert_qa_to_distill_format(self, qa_sample: Dict[str, Any]) -> Dict[str, Any]:
        """将 QA 数据集样本转换为蒸馏格式

        QA 格式:
        {
            "question": str,
            "answer": str,
            "final_answer": str,
            "source_split": str
        }

        转换为蒸馏格式:
        {
            "instruction": str,
            "responses": {},  # 空的 responses（测试时不需要）
            "metadata": {
                "question": str,
                "answer": str,
                "final_answer": str,
                "source": "gsm8k"
            }
        }
        """
        return {
            "instruction": qa_sample["question"],
            "responses": {},  # 测试数据不需要教师响应
            "metadata": {
                "question": qa_sample["question"],
                "answer": qa_sample["answer"],
                "final_answer": qa_sample["final_answer"],
                "source": "gsm8k",
                "source_split": qa_sample.get("source_split", "test")
            }
        }

    def _load_distill_data(self) -> List[Dict[str, Any]]:
        """加载蒸馏数据（OpenR1-Math）"""
        logger = getattr(self.config, "logger", None)
        if logger:
            logger.info(f"[MixedQA] Loading distill data from: {self.distill_hf_path}")

        from datasets import load_dataset
        from .openr1_math import OpenR1MathPreparer

        # 使用 OpenR1MathPreparer 的转换逻辑
        dataset = load_dataset(self.distill_hf_path, split=self.distill_hf_split)

        preparer = OpenR1MathPreparer(self.config)
        samples = []
        for item in dataset:
            converted = preparer._convert_to_distill_format(dict(item))
            samples.append(converted)

        if logger:
            logger.info(f"[MixedQA] Loaded {len(samples)} distill samples")

        return samples

    def _load_qa_data(self) -> tuple[List[Dict[str, Any]], Any]:
        """加载 QA 数据（GSM8K）并返回测试集样本和 judge 函数"""
        logger = getattr(self.config, "logger", None)
        if logger:
            logger.info("[MixedQA] Loading QA data (GSM8K)")

        # 创建临时配置用于加载 GSM8K
        from types import SimpleNamespace

        # 使用当前配置的 qa_settings（如果有的话）
        qa_settings = None
        ds_cfg = self.config.dataset_settings

        if isinstance(ds_cfg, dict):
            qa_settings = ds_cfg.get("qa_settings", {})
        else:
            qa_settings = getattr(ds_cfg, "qa_settings", {})

        # 如果没有 qa_settings，使用默认配置
        if not qa_settings:
            qa_settings = {
                "split": {
                    "train": -1,  # 占位符，不使用
                    "test": "all"  # 使用全部测试数据
                }
            }

        # 构建临时配置
        temp_config = SimpleNamespace(
            dataset_settings=SimpleNamespace(
                type="qa",
                name="qa-gsm8k",
                qa_settings=SimpleNamespace(**qa_settings) if isinstance(qa_settings, dict) else qa_settings
            ),
            global_settings=self.config.global_settings,
            logger=logger
        )

        # 使用 GSM8KPreparer 加载数据
        gsm8k_preparer = GSM8KPreparer(temp_config)
        gsm8k_bundle = gsm8k_preparer.get()

        # 提取测试集和 judge
        test_samples = gsm8k_bundle["splits"]["test"].to_list()
        judge = gsm8k_bundle["judge"]

        if logger:
            logger.info(f"[MixedQA] Loaded {len(test_samples)} QA test samples")

        return test_samples, judge

    def _load_all(self):
        """加载所有数据（蒸馏 + QA）"""
        # 不需要实现，直接在 get() 中处理
        pass

    def get(self):
        """加载、拆分并返回数据集 bundle"""
        logger = getattr(self.config, "logger", None)

        # 1. 加载蒸馏数据作为训练集
        distill_samples = self._load_distill_data()

        # 2. 加载 QA 数据作为测试集
        qa_samples, judge = self._load_qa_data()

        # 3. 转换 QA 数据为蒸馏格式
        qa_samples_converted = [
            self._convert_qa_to_distill_format(sample)
            for sample in qa_samples
        ]

        # 4. 应用配置的 split（只对蒸馏数据进行拆分）
        # 使用基类的 split_samples 对蒸馏数据进行拆分
        all_distill_samples = distill_samples
        distill_splits, placeholder = self.split_samples(all_distill_samples)

        # 5. 构建最终的 splits
        # train: 来自蒸馏数据的 train split
        # test: 来自 QA 数据（转换后）
        final_splits = {}

        if "train" in distill_splits:
            final_splits["train"] = distill_splits["train"]

        # 测试集使用 QA 数据
        final_splits["test"] = DistillDataset(qa_samples_converted)

        # 6. 构建元信息
        total_samples = len(all_distill_samples) + len(qa_samples_converted)
        split_sizes = {
            name: len(ds) for name, ds in final_splits.items()
        }

        meta = {
            "total": total_samples,
            "split_sizes": split_sizes,
            "placeholder_splits": placeholder,
            "split_details": {
                "train": {
                    "source": "distill (OpenR1-Math)",
                    "count": len(final_splits.get("train", []))
                },
                "test": {
                    "source": "qa (GSM8K)",
                    "count": len(qa_samples_converted)
                }
            }
        }

        # 7. 打印报告
        if logger:
            logger.info("=" * 60)
            logger.info("[MixedQA] Dataset Summary:")
            logger.info(f"  Total samples: {meta['total']}")
            logger.info(f"  Train (distill): {meta['split_details']['train']['count']}")
            logger.info(f"  Test (QA): {meta['split_details']['test']['count']}")
            logger.info("=" * 60)

        # 8. 返回 bundle（包含 judge 函数）
        return {
            "splits": final_splits,
            "meta": meta,
            "judge": judge  # 使用 GSM8K 的 judge 函数
        }
