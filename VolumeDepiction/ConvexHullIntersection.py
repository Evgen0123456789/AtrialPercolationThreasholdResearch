from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# =============================================================================
# Convex Hull (Multi-planar Intersection)
# =============================================================================
__all__ = ["get_convex_hull_intersection"]


def _process_slice(args: Tuple[np.ndarray, int]) -> Tuple[int, np.ndarray]:
    """
    Get a convex hull for a down-dimensioned array
    :param args: a slice and an index of slice along target axis 
    :return: pair of index and hull mask
    """
    target_slice, i = args
    pts = np.column_stack(np.where(target_slice)).astype(np.int32)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts[:, ::-1])
        img = np.zeros_like(target_slice)
        cv2.fillConvexPoly(img, hull, 1)
        return i, img
    return i, np.zeros_like(target_slice)


def _hull_axis(mask: np.ndarray, axis: int, n_jobs: Optional[int] = None) -> np.ndarray:
    """
    Create a convex hull stack of slices along given axis
    :param mask: a binary mask
    :param axis: a target axis
    :param n_jobs: a count of processors to use with
    :return: a binary mask
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    n_slices = mask.shape[axis]
    appropriate_indexes = [i for i in range(n_slices) if np.any(np.take(mask, i, axis=axis))]
    args_list = [(np.take(mask, i, axis=axis).copy(), i) for i in appropriate_indexes]

    print(f"  Axis {axis}: {len(appropriate_indexes)}/{n_slices} non-empty slices")


    results = [None] * n_slices
    with Pool(n_jobs) as pool:
        for i, img in tqdm(pool.imap(_process_slice, args_list), total=len(args_list), desc=f'Axis {axis}'):
            results[i] = img.astype(np.bool_)

    for i in range(n_slices):
        if results[i] is None:
            shape = list(mask.shape)
            shape.pop(axis)
            results[i] = np.zeros(shape, dtype=np.bool_)

    return np.stack(results, axis=axis)


def get_convex_hull_intersection(mask: np.ndarray, n_jobs: Optional[int] = None) -> np.ndarray:
    """
    Get an intersection of convex hulls for a mask along every axis to create an approximate inner volume depiction
    :param mask: numpy array (it supposed to be binary)
    :param n_jobs: a count of CPU processors to use with
    :return: a mask
    """
    hulls = [_hull_axis(mask, i, n_jobs=n_jobs) for i in range(len(mask.shape))]
    return (np.logical_and(*hulls)).astype(np.uint8)