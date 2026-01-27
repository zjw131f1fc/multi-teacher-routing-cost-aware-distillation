from torch.utils.data import Dataset
from typing import Dict, Any, List


class BaseDataset(Dataset):
    """基础数据集类"""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """返回包含 instruction, gt 及其他信息的字典"""
        return self.samples[idx]


def load_dataset(config: dict) -> BaseDataset:
    """根据配置加载数据集"""
    dataset_config = config.get("dataset", {})

    # TODO: 根据 dataset_config 加载实际数据
    # 目前返回空数据集作为占位
    samples = []

    return BaseDataset(samples)
