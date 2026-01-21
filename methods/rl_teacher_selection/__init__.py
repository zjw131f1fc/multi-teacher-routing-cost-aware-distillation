"""RL-based Cost-Aware Teacher Selection for LLM Distillation"""

from .policy import PolicyNetwork
from .mock_data import (
    get_mock_instructions,
    get_mock_student_model,
    get_mock_teachers,
    get_mock_val_loader,
    get_mock_data,
    MockStudentModel,
    MockTeacher,
)
from .gradient import (
    compute_gradient,
    compute_validation_gradient,
    get_last_n_layers,
)
from .reward import compute_reward
from .teacher_pool import Teacher, TeacherPool
from .sb3_env import TeacherSelectionEnv

__all__ = [
    "PolicyNetwork",
    "get_mock_instructions",
    "get_mock_student_model",
    "get_mock_teachers",
    "get_mock_val_loader",
    "get_mock_data",
    "MockStudentModel",
    "MockTeacher",
    "compute_gradient",
    "compute_validation_gradient",
    "get_last_n_layers",
    "compute_reward",
    "Teacher",
    "TeacherPool",
    "TeacherSelectionEnv",
]
