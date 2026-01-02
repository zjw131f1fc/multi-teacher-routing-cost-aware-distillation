"""Base classes for dataset preparation.

Provides common functionality for different dataset types:
- vqa: Visual Question Answering base classes
- distill: Distillation dataset base classes
- qa: Question Answering base classes
"""
# VQA base classes (保留原名以兼容现有实现)
from .vqa import BasePreparer, BsesDataset  # noqa: F401

# Distill base classes (使用别名避免冲突)
from .distill import BasePreparer as DistillBasePreparer, DistillDataset  # noqa: F401

# QA base classes (使用别名避免冲突)
from .qa import BasePreparer as QABasePreparer, QADataset  # noqa: F401
