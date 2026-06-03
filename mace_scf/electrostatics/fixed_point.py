"""
Legacy FixedPoint class — kept only so that torch.load can deserialize
old saved models.  Calling forward() will raise an error directing the
user to convert the model to FixedPointCore via scripts/convert_models.py.
"""
from typing import Any, Callable, Dict, List, Optional, Type
import numpy as np
import torch
from e3nn import o3

from mace.modules import InteractionBlock

from .fixed_point_core import FixedPointCore


class FixedPoint(FixedPointCore):
    """
    Backward-compatible shell around FixedPointCore.

    torch.load will reconstruct instances of this class from old checkpoints.
    All submodules are identical to FixedPointCore (inherited __init__), so the
    state dict loads correctly.  The only behavioural change is that forward()
    raises an error instead of running the old monolithic code path.
    """

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "FixedPoint.forward() has been removed. "
            "Convert this model to FixedPointCore using:\n"
            "  python scripts/convert_models.py <model_path> --output_path <output_path>\n"
            "Then load the converted model and use FixedPointSCFRunner for evaluation."
        )
