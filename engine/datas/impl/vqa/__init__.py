"""VQA (Visual Question Answering) dataset implementations."""
from .mme import MMEPreparer  # noqa: F401
from .vqa_v2 import VQAV2Preparer  # noqa: F401
from .pope import POPEPreparer  # noqa: F401
from .mmb import MMBenchPreparer  # noqa: F401
from .scienceqa import ScienceQAPreparer  # noqa: F401
from .gqa import GQAPreparer  # noqa: F401
from .seed_bench import SEEDBenchPreparer  # noqa: F401
