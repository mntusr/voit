from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DepthData(Protocol):
    depth: np.ndarray
    """
    The depth map. Format: ``Im_``
    """
    depth_mask: np.ndarray
