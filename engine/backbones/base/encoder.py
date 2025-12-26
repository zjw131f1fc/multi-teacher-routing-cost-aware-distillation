"""Encoder (编码器模型) Backbone 基类。

定义 Encoder 类型 Backbone 的通用接口和基础功能。
适用于纯编码器模型如 BERT, DeBERTa, RoBERTa 等。
具体实现应放在 impl/encoder/ 中。
"""

from typing import Dict, Any, Optional


class BaseEncoderBackbone:
    """Encoder Backbone 基类。

    子类需要实现具体模型加载和编码方法。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化 Backbone。

        参数:
            config: Config 实例，包含 backbone_settings 等配置
        """
        self.config = config
        if config is not None:
            self.backbone_cfg = config.backbone_settings  # type: ignore
            self.model_name = self.backbone_cfg['name']
        else:
            self.backbone_cfg = {}
            self.model_name = 'unknown'

        # 模型和分词器
        self.model = None
        self.tokenizer = None

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.model_name}')>"
