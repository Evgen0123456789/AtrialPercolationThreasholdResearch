from typing import Tuple

import numpy as np


def print_volume(mask: np.ndarray, spacing: Tuple[float, float, float], msg: str):
    voxels = np.sum(mask)
    ml = voxels * np.prod(spacing) * 1e-3
    print(f"  Volume {msg}: {voxels} voxels ({ml:.2f} ml)")