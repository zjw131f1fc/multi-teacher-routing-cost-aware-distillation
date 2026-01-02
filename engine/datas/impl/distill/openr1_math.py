"""OpenR1-Math 蒸馏数据集准备器。

数据来源: HuggingFace datasets (open-r1/OpenR1-Math-220k)
模型: DeepSeek-R1 (单一模型，可能包含多个生成样本)

原始字段:
    - problem: 问题描述
    - solution: 完整解答
    - answer: 最终答案
    - problem_type: 问题类型（如 "Algebra"）
    - question_type: 问题分类（如 "math-word-problem"）
    - source: 数据来源
    - uuid: 唯一标识
    - is_reasoning_complete: 推理是否完整
    - generations: 生成的响应列表
    - correctness_math_verify: 数学验证正确性列表
    - correctness_llama: LLaMA 评估正确性列表
    - finish_reasons: 完成原因列表
    - correctness_count: 正确数量
    - messages: 对话消息列表（每个元素是一次生成的完整对话）

转换后的统一格式:
    {
        "instruction": str,           # 问题描述
        "responses": {                # DeepSeek-R1 的多个生成
            "deepseek-r1": {          # 单生成时
                "messages": [...],    # 对话历史
                "rewards": {
                    "math_verify": float,
                    "llama": float
                }
            },
            "deepseek-r1-gen0": {...}, # 多生成时（带索引）
            "deepseek-r1-gen1": {...},
            ...
        },
        "metadata": {                 # 数据集特定元信息
            "problem": str,
            "answer": str,
            "solution": str,
            "problem_type": str,
            "question_type": str,
            "source": str,
            "uuid": str,
            "is_reasoning_complete": bool,
            "finish_reasons": list,
            "correctness_count": int
        }
    }
"""

from typing import Dict, Any
from datasets import load_dataset
from ...base.distill import BasePreparer, DistillDataset


class OpenR1MathDataset(DistillDataset):
    pass


class OpenR1MathPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        ds_cfg = self.config.dataset_settings

        # HuggingFace 数据集路径（兼容 dict 和 SimpleNamespace）
        if isinstance(ds_cfg, dict):
            self.hf_path = ds_cfg.get("hf_path", "open-r1/OpenR1-Math-220k")
            self.hf_split = ds_cfg.get("hf_split", "train")
        else:
            self.hf_path = getattr(ds_cfg, "hf_path", "open-r1/OpenR1-Math-220k")
            self.hf_split = getattr(ds_cfg, "hf_split", "train")

    def _convert_to_distill_format(self, raw_item: Dict[str, Any]) -> Dict[str, Any]:
        """将原始数据转换为蒸馏格式。

        OpenR1-Math 数据集包含单个模型 (deepseek-r1) 的多个生成结果。

        转换为:
        {
            "instruction": str,
            "responses": {
                "deepseek-r1": {
                    "messages": [...],
                    "rewards": {...}
                }
            },
            "metadata": {...}
        }
        """
        # instruction 就是 problem
        instruction = raw_item.get("problem", "")

        # 构建 responses
        # OpenR1-Math 只有一个模型：deepseek-r1
        # messages 包含多个生成的对话历史
        # correctness_math_verify 和 correctness_llama 是对应的 reward 列表
        messages_list = raw_item.get("messages", [])
        correctness_math = raw_item.get("correctness_math_verify", [])
        correctness_llama = raw_item.get("correctness_llama", [])

        # 处理 None 值
        if messages_list is None:
            messages_list = []
        if correctness_math is None:
            correctness_math = []
        if correctness_llama is None:
            correctness_llama = []

        # 如果有多个生成，选择最好的作为 deepseek-r1，其他的加索引
        if len(messages_list) == 0:
            responses = {}
        else:
            # 只有一个生成，直接使用 deepseek-r1
            responses = {
                "deepseek-r1": {
                    "messages": messages_list,
                    "rewards": {
                        "math_verify": correctness_math[0] if len(correctness_math) > 0 else None,
                        "llama": correctness_llama[0] if len(correctness_llama) > 0 else None
                    }
                }
            }

        # 构建 metadata（包含原始的元信息）
        metadata = {
            "problem": raw_item.get("problem", ""),
            "answer": raw_item.get("answer", ""),
            "solution": raw_item.get("solution", ""),
            "problem_type": raw_item.get("problem_type", ""),
            "question_type": raw_item.get("question_type", ""),
            "source": raw_item.get("source", ""),
            "uuid": raw_item.get("uuid", ""),
            "is_reasoning_complete": raw_item.get("is_reasoning_complete", None),
            "finish_reasons": raw_item.get("finish_reasons", []),
            "correctness_count": raw_item.get("correctness_count", None)
        }

        return {
            "instruction": instruction,
            "responses": responses,
            "metadata": metadata
        }

    def _load_all(self):
        """从 HuggingFace 加载数据并转换格式。"""
        logger = getattr(self.config, "logger", None)
        if logger:
            logger.info(f"[OpenR1Math] Loading from HuggingFace: {self.hf_path} (split={self.hf_split})")

        # 获取 HuggingFace cache 目录
        hf_cache_dir = None
        global_settings = getattr(self.config, "global_settings", None) or self.config.get("global_settings")
        if global_settings:
            if isinstance(global_settings, dict):
                hf_cache_dir = global_settings.get("hf_cache_dir")
            else:
                hf_cache_dir = getattr(global_settings, "hf_cache_dir", None)

        # 加载 HuggingFace 数据集（使用缓存目录，优先使用缓存）
        dataset = load_dataset(
            self.hf_path,
            split=self.hf_split,
            cache_dir=hf_cache_dir,  # 使用配置的缓存目录
            download_mode="reuse_cache_if_exists"  # 优先使用缓存，避免不必要的下载
        )

        # 转换为蒸馏格式
        samples = []
        for item in dataset:
            converted = self._convert_to_distill_format(dict(item))
            samples.append(converted)

        if logger:
            logger.info(f"[OpenR1Math] Loaded and converted {len(samples)} samples")

        return samples

    def get(self):
        """加载、拆分并返回数据集 bundle。"""
        # 加载所有样本
        all_samples = self._load_all()

        # 使用基类的 split_samples 进行拆分
        split_datasets, placeholder_splits = self.split_samples(all_samples)

        # 构建元信息
        meta = self.build_meta(all_samples, split_datasets, placeholder_splits)

        # 打印报告
        self.base_report(meta)

        # 返回 bundle（judge 为 None）
        return {
            "splits": split_datasets,
            "meta": meta,
            "judge": None
        }
