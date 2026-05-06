import os
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import numba as nb
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import ball

import SimpleITK as sitk

# =============================================================================
# Utilities
# =============================================================================

def get_bounding_box(mask: np.ndarray, pad: int):
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    bounding_box = list(map(lambda x: (x.min() - pad, x.max() + pad + 1), coords))
    x0, x1 = max(0, bounding_box[0][0]), min(mask.shape[0], bounding_box[0][1])
    y0, y1 = max(0, bounding_box[1][0]), min(mask.shape[1], bounding_box[1][1])
    z0, z1 = max(0, bounding_box[2][0]), min(mask.shape[2], bounding_box[2][1])

    return (x0, x1), (y0, y1), (z0, z1)

def plt_save_np_hist(hist, bins, title, xlable, ylable, name):
    centers = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(6, 4))
    plt.bar(centers, hist, width=bins[1] - bins[0], edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.title(title)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches='tight')
    plt.close()
    return

def print_volume(mask: np.ndarray, spacing: Tuple[float, float, float], msg: str):
    voxels = np.sum(mask)
    ml = voxels * np.prod(spacing) * 1e-3
    print(f"  Volume {msg}: {voxels} voxels ({ml:.2f} ml)")

def restore_to_original(
        cropped: np.ndarray,
        orig_shape: tuple,
        bbox: tuple,
        dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Восстанавливает кропнутое решение в оригинальный размер.
    """
    restored = np.zeros(orig_shape, dtype=dtype)

    x0, x1 = bbox[0]
    y0, y1 = bbox[1]
    z0, z1 = bbox[2]

    # Вычисляем размер области по bbox
    dx_bbox = x1 - x0  # 121
    dy_bbox = y1 - y0  # 140
    dz_bbox = z1 - z0  # 204

    # Фактический размер cropped после всех трансформаций
    sx, sy, sz = cropped.shape

    insert_x = min(dx_bbox, sx)
    insert_y = min(dy_bbox, sy)
    insert_z = min(dz_bbox, sz)

    restored[x0:x0 + insert_x, y0:y0 + insert_y, z0:z0 + insert_z] = \
        cropped[:insert_x, :insert_y, :insert_z].astype(dtype)

    return restored


def save_arr_as_image(arr:np.ndarray, spacing, origin, direction, name):
    img_awt = sitk.GetImageFromArray(arr)
    img_awt.SetSpacing(spacing)
    img_awt.SetOrigin(origin)
    img_awt.SetDirection(direction)
    sitk.WriteImage(img_awt, name)


def save_nrrd_with_metadata(
        data: np.ndarray,
        reference: sitk.Image,
        bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        spacing_scale: float = 1.0,
        filename: str = "output.nrrd",
        dtype: np.dtype = np.float32
):
    """
    Step 11: Save with correct NRRD metadata for 3DSlicer.
    spacing_scale: 1.0 for coarse, 0.25 for fine (×4 grid)
    """
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
#endregion

def compute_zone_dice(pred_awt: np.ndarray, gt_awt: np.ndarray,
                      thickness_bins: np.ndarray = None, units = "") -> dict:
    """
    Вычисляет Dice для зон стенки, разбитых по интервалам толщины.
    thickness_bins: например, [0, 1.5, 2.5, 4.0, 12.0] мм
    """
    if thickness_bins is None:
        thickness_bins = np.linspace(0, 5, 16)
    results = {}
    for i in range(len(thickness_bins) - 1):
        t_min, t_max = thickness_bins[i], thickness_bins[i + 1]
        # Зона по референсу (GT)
        zone_gt = (gt_awt > t_min) & (gt_awt <= t_max)
        # Предсказание попадает в ту же зону
        zone_pred = (pred_awt > t_min) & (pred_awt <= t_max)

        intersection = np.sum(zone_gt & zone_pred)
        union = np.sum(zone_gt) + np.sum(zone_pred)
        dice = 2 * intersection / (union + 1e-8)

        results[f"{t_min:.2f}-{t_max:.2f} {units}"] = dice
    return results


def thickness_error_histogram(pred_awt: np.ndarray, gt_awt: np.ndarray,
                              bins: np.ndarray = None) -> tuple:
    """
    Возвращает гистограмму |pred - gt| только в области пересечения стенок.
    """
    if bins is None:
        bins = np.linspace(-10, 10, 201)  # -10-10 мм с шагом 0.1 мм

    # Только где обе маски определяют стенку
    valid = (pred_awt > 0) & (gt_awt > 0)
    abs_err = (pred_awt[valid] - gt_awt[valid])

    hist, edges = np.histogram(abs_err, bins=bins, density=False)
    return hist, edges, abs_err.mean(), abs_err.std()

@nb.njit(fastmath = True)
def num_bounded_comparts(line: np.ndarray):
    n = 1
    guard = 32
    for i in nb.prange(guard, line.shape[0] - guard):
        if line[i - 1] == line[i] + 1:
            n+=1
    return n


@nb.njit(fastmath = True)
def depict_line(in_line: np.ndarray, out_line: np.ndarray):
    n = False
    guard = 32
    for i in nb.prange(guard, in_line.shape[0] - guard):
        if in_line[i] == 1:
            out_line[i, 0] += 1
        else:
            if in_line[i-1] == 1:
                n = not n
            if n:
                out_line[i, 1] += 1 
    return

@nb.njit(fastmath = True)
def depict_surface(in_layer: np.ndarray, out_layer: np.ndarray):
    guard = 32
    for i in nb.prange(guard, in_layer.shape[0] - guard):
        p = num_bounded_comparts(in_layer[i])
        if p % 2 == 1 and p > 1:
            depict_line(in_layer[i], out_layer[i])
    for i in nb.prange(guard, in_layer.shape[1] - guard):
        p = num_bounded_comparts(in_layer[:, i])
        if p % 2 == 1 and p > 1:
            depict_line(in_layer[:, i], out_layer[:, i])
    return


@nb.njit
def depict_voluem(in_mask: np.ndarray,
                    out_mask: np.ndarray):
    guard = 32
    for i in nb.prange(guard, in_mask.shape[0] - guard):
        depict_surface(in_mask[i], out_mask[i])

    for i in nb.prange(guard, in_mask.shape[1] - guard):
        depict_surface(in_mask[:, i, :], out_mask[:, i, :])

    for i in nb.prange(guard, in_mask.shape[2] - guard):
        depict_surface(in_mask[:, :, i], out_mask[:, :, i])



@nb.njit(fastmath=True, parallel=True)
def MedianFilter(arr: np.ndarray):
    brr = np.zeros_like(arr)
    guard: int = 20
    kernel = 1
    th = 5
    
    for i in nb.prange(guard, arr.shape[0] - guard):
        for j in nb.prange(guard, arr.shape[1] - guard):
            for k in nb.prange(guard, arr.shape[2] - guard):
                brr[i, j, k, ] = 1 if arr[i-kernel: i+kernel, j-kernel: j+kernel, k-kernel: k+kernel].sum() > th else 0
    return brr

@nb.njit(fastmath=True, parallel=True, cache=True)
def MedianClassifier(arr: np.ndarray):
    brr = np.zeros_like(arr)
    guard: int = 32
    kernel = 1
    
    
    for i in nb.prange(guard, arr.shape[0] - guard):
        for j in nb.prange(guard, arr.shape[1] - guard):
            for k in nb.prange(guard, arr.shape[2] - guard):
                if arr[i, j, k, 0] == 1.0:
                    brr[i, j, k, 0] = 1.0
                    continue
                criterion = 0
                
                for _i in nb.prange(i-kernel, i+kernel + 1):
                    for _j in nb.prange(j-kernel, j+kernel + 1):
                        for _k in nb.prange(k-kernel, k+kernel + 1):
                            if arr[_i, _j, _k, 0] == 1:
                                continue
                            elif arr[_i, _j, _k, 1] == 1:
                                criterion += 1
                            else:
                                criterion -= 1
                if criterion == 0: 
                    continue
                elif criterion < 0:
                    brr[i, j, k, 1] = 0
                else:
                    brr[i, j, k, 1] = 1
    return brr

@nb.njit(fastmath=True, parallel=True)
def EstimateThicknessVoluem2Area(arr: np.ndarray):
    v: int = 0
    s: int = 0
    guard: int = 25
    
    for i in nb.prange(guard, arr.shape[0] - guard):
        for j in nb.prange(guard, arr.shape[1] - guard):
            for k in nb.prange(guard, arr.shape[2] - guard):
                if arr[i, j, k] > 0:
                    v+=1
                    if arr[i+1, j, k] == 0:
                        s+=1
                    if arr[i-1, j, k] == 0:
                        s+=1
                    if arr[i, j+1, k] == 0:
                        s+=1
                    if arr[i, j-1, k] == 0:
                        s+=1
                    if arr[i, j, k+1] == 0:
                        s+=1
                    if arr[i, j, k-1] == 0:
                        s+=1
    return v / s

@nb.njit(fastmath=True, parallel=True)
def Dice(test: np.ndarray, pred: np.ndarray):
    return 2 * np.sum(test * pred) / (test.sum() + pred.sum())
                    
            

def ComparePredAndTest(name: str, path2test: str, path2pred: str):
    test = sitk.GetArrayFromImage(sitk.ReadImage(path2test))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(path2pred))
        
    test = (test > 0).astype(int)
    pred = (pred == 1).astype(int)
    
    dx, dy, dz = pred.shape
    
    pred[:dx//5, :, :] = 0
    pred[:, :dy//5, :] = 0
    pred[:, :, :dz//5] = 0
    pred[4*dx//5:, :, :] = 0
    pred[:, 4*dy//5:, :] = 0
    pred[:, :, 4*dz//5:] = 0
    dice = Dice(test, pred)
    
    return {
        'Sample': name,
        'Test_Unique_Values': str(np.unique(test)),
        'Test_Shape': test.shape,
        'Pred_Unique_Values': str(np.unique(pred)),
        'Pred_Shape': pred.shape,
        'Dice_Score': dice
    }
        
def review(path: str):
    for fname in os.listdir(path):
        train = sitk.GetArrayFromImage(sitk.ReadImage(path + '/' + fname))
        return("Test values:", np.unique(train), "Shape:", train.shape)
        
# def GifCompare(path2test: str, path2pred: str):
#     fig = plt.figure()
#     camera = Camera(fig)
#     test = sitk.GetArrayFromImage(sitk.ReadImage(path2test))
#     pred = sitk.GetArrayFromImage(sitk.ReadImage(path2pred))
#     r = test == 1
#     g = pred == 1
#     b = pred == 2
#     pict = np.stack((r, g, b), axis = 3).astype(np.float64)
#     #print(pict.shape)
#     for l in pict:
#         plt.imshow(l)
#         camera.snap()
#     anim = camera.animate()
#     return anim # anim.save(Sin.Network_with_artifitial_cover.v2.gif)

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, fastmath=True)
def sample_thickness_numba(voxel_indices, awt_field, thickness_out):
    """
    Numba-ускоренное сэмплирование толщины на воксельных индексах.
    """
    n = voxel_indices.shape[0]
    for i in prange(n):
        ix = voxel_indices[i, 0]
        iy = voxel_indices[i, 1]
        iz = voxel_indices[i, 2]

        if 0 <= ix < awt_field.shape[0] and \
                0 <= iy < awt_field.shape[1] and \
                0 <= iz < awt_field.shape[2]:
            thickness_out[i] = awt_field[ix, iy, iz]
        else:
            thickness_out[i] = 0.0

    return thickness_out


def save_surface_points_optimized(field, indices, spacing, filepath):
    key_values = np.zeros(len(indices), dtype=np.float64)
    key_values = sample_thickness_numba(indices, field, key_values)

    np.savez_compressed(
        filepath,
        voxel_indices=indices,
        thickness=key_values,
        spacing=spacing
    )
    print(f"  Saved: {filepath} ({len(indices)} voxels)")

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


def save_mask(mask, spacing, direction, origin, out, name):
    img = sitk.GetImageFromArray(mask)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    img.SetOrigin(origin)
    sitk.WriteImage(img, os.path.join(out, f"{name}.nrrd"))
    
if __name__=="__main__":
    
    path2pred = "/home/evgeniy/Рабочий стол/Научная работа/Данные/Датасеты/Dataset003_LA_with_perepheria_like_onion/predicted"
    path2test = "/home/evgeniy/Рабочий стол/Научная работа/Данные/Датасеты/Dataset003_LA_with_perepheria_like_onion/labelsTs"
    path2train = "/home/evgeniy/Рабочий стол/Научная работа/Данные/Датасеты/Dataset003_LA_with_perepheria_like_onion/labelsTr"
    path2reports = "/home/evgeniy/Рабочий стол/Научная работа/Отчёты"
    path2animation = '/home/evgeniy/Рабочий стол/Научная работа/Данные/Изображения'
    
    #print(os.listdir(path2pred))
    #print(os.listdir(path2test))
    #print(os.listdir(path2train))
    
    sample = path2test + '/' + "Kar.nrrd"
    reff = sitk.ReadImage(sample)
    test = sitk.GetArrayFromImage(reff)
    test = (test > 0).astype(int)
    X, Y, Z = test.shape
    res = np.zeros((X, Y, Z, 3))
    
    morph = ball(2)
    res[:, :, :, 0] = test
    res[:, :, :, 1] = binary_erosion(binary_dilation(test, morph), morph)
    print(Dice(res[:, :, :, 0], res[:, :, :, 1]))
    # depict_voluem(test, res)
    #res = (res > 0).astype(np.float64)
    #res = MedianClassifier(res)
    #res = MedianClassifier(res)
    #res = MedianClassifier(res)
    #for l in res:
    #    plt.imshow(l)
    #    plt.show()
    
    out = sitk.GetImageFromArray(res[:, :, :, 1])
    out.CopyInformation(reff)
    sitk.WriteImage(out, path2animation + "/Kar_morphology.nrrd")
