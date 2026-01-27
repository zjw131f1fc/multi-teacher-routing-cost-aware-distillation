import json
import copy
from pathlib import Path
from typing import Dict, Union, Optional, Any

from ..datas import BaseDataset


class TeacherPool:
    """教师池，管理教师输出的获取和缓存"""

    def __init__(
        self,
        config: dict,
        dataset: BaseDataset,
        teachers: Dict[str, Any],
    ):
        """
        Args:
            config: 配置字典
            dataset: 数据集
            teachers: 教师模型字典 {teacher_name: TeacherModel}
        """
        self.config = copy.deepcopy(config)
        self.dataset = dataset
        self.teachers = teachers
        self.cache: Dict[str, str] = {}
        self._load_cache()

    def _get_cache_path(self) -> Path:
        """获取 cache 文件路径"""
        cache_dir = Path(self.config.get("cache_dir", "./cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = self.config.get("dataset", {}).get("name", "default")
        return cache_dir / f"{dataset_name}_teacher_cache.json"

    def _load_cache(self) -> None:
        """加载 cache 文件"""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def _save_cache(self) -> None:
        """保存 cache 文件"""
        cache_path = self._get_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _make_cache_key(self, instruction: str, teacher_name: str) -> str:
        """生成 cache key"""
        return f"{teacher_name}||{instruction}"

    def _get_from_cache(self, instruction: str, teacher_name: str) -> Optional[str]:
        """从 cache 获取"""
        key = self._make_cache_key(instruction, teacher_name)
        return self.cache.get(key)

    def _get_from_dataset(self, instruction: str, teacher_name: str) -> Optional[str]:
        """从数据集获取"""
        for sample in self.dataset.samples:
            if sample.get("instruction") == instruction:
                teacher_outputs = sample.get("teacher_outputs", {})
                return teacher_outputs.get(teacher_name)
        return None

    def _call_teacher(self, instruction: str, teacher_name: str) -> str:
        """调用教师模型生成"""
        if teacher_name not in self.teachers:
            raise ValueError(f"Teacher '{teacher_name}' not found in pool")
        return self.teachers[teacher_name].generate(instruction)

    def _get_single(self, instruction: str, teacher_name: str) -> str:
        """获取单条教师输出"""
        # 1. 从 cache 获取
        result = self._get_from_cache(instruction, teacher_name)
        if result is not None:
            return result

        # 2. 从数据集获取
        result = self._get_from_dataset(instruction, teacher_name)
        if result is not None:
            # 存入 cache
            key = self._make_cache_key(instruction, teacher_name)
            self.cache[key] = result
            self._save_cache()
            return result

        # 3. 调用教师模型
        result = self._call_teacher(instruction, teacher_name)
        # 存入 cache
        key = self._make_cache_key(instruction, teacher_name)
        self.cache[key] = result
        self._save_cache()
        return result

    def get_output(
        self,
        instruction: Union[str, Dict[str, str]],
        teacher_name: Optional[str] = None,
    ) -> Union[str, Dict[str, str]]:
        """
        获取教师输出

        单条模式: get_output("instruction", "gpt-4") -> str
        并行模式: get_output({"inst1": "gpt-4", "inst2": "llama"}) -> dict
        """
        if isinstance(instruction, str):
            # 单条模式
            if teacher_name is None:
                raise ValueError("单条模式需要指定 teacher_name")
            return self._get_single(instruction, teacher_name)
        else:
            # 并行模式
            results = {}
            for inst, t_name in instruction.items():
                results[inst] = self._get_single(inst, t_name)
            return results
