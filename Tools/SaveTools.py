import os
from typing import Tuple

import SimpleITK as sitk
import numpy as np


def save_mask(mask, spacing, direction, origin, out: str, name: str):
    img = sitk.GetImageFromArray(mask)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    img.SetOrigin(origin)
    sitk.WriteImage(img, os.path.join(out, name))


def save_nrrd_with_metadata(
        data: np.ndarray,
        reference: sitk.Image,
        bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        spacing_scale: float = 1.0,
        filename: str = "output.nrrd",
        dtype: np.dtype = np.float32
):
    img = sitk.GetImageFromArray(data.astype(dtype))

    original_spacing = reference.GetSpacing()
    fine_spacing = tuple(s * spacing_scale for s in original_spacing)
    img.SetSpacing(fine_spacing)
    print(f"Fine spacing: {fine_spacing}")

    # Origin: reference origin + bbox offset
    origin = [
        reference.GetOrigin()[0] + bbox[0][0] * fine_spacing[0],
        reference.GetOrigin()[1] + bbox[1][0] * fine_spacing[1],
        reference.GetOrigin()[2] + bbox[2][0] * fine_spacing[2]
    ]
    img.SetOrigin(origin)
    img.SetDirection(reference.GetDirection())

    sitk.WriteImage(img, filename)
    print(f"  Saved: {filename} (shape={data.shape}, spacing={fine_spacing})")


def save_points(field: np.ndarray, indices: np.ndarray, spacing, filepath):
    """
    Сохраняет указанные точки с толщиной в простой формат.
    """
    point_value = field[indices[:, 0], indices[:, 1], indices[:, 2]]

    np.savez_compressed(
        filepath,
        points=indices,
        thickness=point_value,
        spacing=spacing
    )
    print(f"  Saved: {filepath}")
