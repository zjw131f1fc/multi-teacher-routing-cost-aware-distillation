import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """加载YAML配置文件"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
