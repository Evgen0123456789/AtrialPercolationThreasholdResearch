import os

import numpy as np
import numba as nb
from scipy import ndimage
import SimpleITK as sitk

# =============================================================================
# Utilities
# =============================================================================


@nb.njit(fastmath=True, parallel=True)
def Dice(test: np.ndarray, pred: np.ndarray)->float:
    T = np.count_nonzero(test)
    P = np.count_nonzero(pred)
    F = np.count_nonzero(test * pred)
    return 2 * F / (T + P + 1e-8)


def get_largest_domain(mask: np.ndarray):
    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        largest_component = np.argmax(component_sizes[1:]) + 1
        return (labeled == largest_component).astype(np.uint8), num_features # Extract the biggest domain
    return mask, num_features


def get_bounding_box(mask: np.ndarray, pad: int):
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    bounding_box = list(map(lambda x: (x.min() - pad, x.max() + pad + 1), coords))
    x0, x1 = max(0, bounding_box[0][0]), min(mask.shape[0], bounding_box[0][1])
    y0, y1 = max(0, bounding_box[1][0]), min(mask.shape[1], bounding_box[1][1])
    z0, z1 = max(0, bounding_box[2][0]), min(mask.shape[2], bounding_box[2][1])

    return (x0, x1), (y0, y1), (z0, z1)


def restore_to_original(
        cropped: np.ndarray,
        orig_shape: tuple,
        bbox: tuple,
        dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Restores solution on cropped grid to original size.
    """
    restored = np.zeros(orig_shape, dtype=dtype)

    x0, x1 = bbox[0]
    y0, y1 = bbox[1]
    z0, z1 = bbox[2]

    # Вычисляем размер области по bbox
    dx_bbox = x1 - x0
    dy_bbox = y1 - y0
    dz_bbox = z1 - z0

    # Фактический размер cropped после всех трансформаций
    sx, sy, sz = cropped.shape

    insert_x = min(dx_bbox, sx)
    insert_y = min(dy_bbox, sy)
    insert_z = min(dz_bbox, sz)

    restored[x0:x0 + insert_x, y0:y0 + insert_y, z0:z0 + insert_z] = \
        cropped[:insert_x, :insert_y, :insert_z].astype(dtype)

    return restored


def ComparePredAndTest(name: str, path2test: str, path2pred: str):
    test = sitk.GetArrayFromImage(sitk.ReadImage(path2test))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(path2pred))

    test = (test > 0).astype(int)
    pred = (pred == 1).astype(int)


    pred, _ = get_largest_domain(pred)
    dice = Dice(test, pred)

    return {
        'Sample': name,
        'Test_Unique_Values': str(np.unique(test)),
        'Pred_Unique_Values': str(np.unique(pred)),
        'Dice_Score': dice
    }


def review_samples(path: str):
    for fname in os.listdir(path):
        train = sitk.GetArrayFromImage(sitk.ReadImage(path + '/' + fname))
        return "Test values:", np.unique(train), "Shape:", train.shape


def IOU(a, b):
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = np.sum(a_bool & b_bool)
    union = np.sum(a_bool | b_bool)
    return inter / (union + 1e-8)


def compute_zone_dice(
        pred_awt: np.ndarray,
        gt_awt: np.ndarray,
        thickness_bins: np.ndarray = None,
        units=""
) -> dict:
    if thickness_bins is None:
        thickness_bins = np.linspace(0, 5, 16)
    results = {}
    for i in range(len(thickness_bins) - 1):
        t_min, t_max = thickness_bins[i], thickness_bins[i + 1]

        zone_gt = (gt_awt > t_min) & (gt_awt <= t_max)
        zone_pred = (pred_awt > t_min) & (pred_awt <= t_max)

        dice = Dice(zone_gt, zone_pred)
        results[f"{t_min:.2f}-{t_max:.2f} {units}"] = dice
    return results


def bootstrap_ci(data1, data2, metric_func, n_boot=1000, alpha=0.05):
    """Bootstrap confidence interval для разницы метрик"""
    diffs = []
    n = len(data1)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        diffs.append(metric_func(data1[idx], data2[idx]))
    ci_low = np.percentile(diffs, 100 * alpha / 2)
    ci_high = np.percentile(diffs, 100 * (1 - alpha / 2))
    return ci_low, ci_high
