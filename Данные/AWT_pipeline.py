import gc
import sys
import time
from multiprocessing import freeze_support

import numpy as np
import SimpleITK as sitk

from VolumeDepiction.ConvexHullIntersection import get_convex_hull_intersection
from VolumeDepiction.GridTools import upscale_grid, create_field_and_interior_indices
from VolumeDepiction.SurfaceExtraction import extract_surface
from WallThicknessTools.LaplasSolver import solve_laplace_interior
from WallThicknessTools.PairedDEMethod import compute_wall_thickness_interior
from WorkTools import print_volume, get_bounding_box, thickness_error_histogram, compute_zone_dice, plt_save_np_hist


def filter_outliers(points, thickness, max_distance_mm=50, thickness_range=(0.3, 6.0)):
    """
    Убирает точки, которые:
    - дальше max_distance_mm от центра облака
    - имеют толщину вне разумных пределов
    """
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)

    mask = (distances < max_distance_mm) & \
           (thickness >= thickness_range[0]) & \
           (thickness <= thickness_range[1])

    return points[mask], thickness[mask], mask

def quick_plot_surface_points(filepath, clim=(0.5, 4.0), save_path=None):
    import pyvista as pv
    import numpy as np

    data = np.load(filepath)
    points = data['points']
    thickness = data['thickness']
    stype = data['surface_type']

    # Фильтрация
    points_filt, thickness_filt, kept_mask = filter_outliers(
        points, thickness, max_distance_mm=40, thickness_range=(0.4, 5.0)
    )
    stype_filt = stype[kept_mask]

    cloud = pv.PolyData(points_filt)
    cloud['thickness'] = thickness_filt
    cloud['surface'] = stype_filt

    plotter = pv.Plotter(window_size=[1200, 800])

    # Эндокард
    endo = cloud.extract_points(stype_filt == 0)
    plotter.add_points(endo, scalars='thickness', cmap='magma',
                       clim=clim, point_size=3, render_points_as_spheres=False,
                       label='Endocardium')

    # Эпикард (полупрозрачный)
    epi = cloud.extract_points(stype_filt == 1)
    plotter.add_points(epi, scalars='thickness', cmap='magma',
                       clim=clim, point_size=2, opacity=0.4,
                       label='Epicardium')

    # Цветовая шкала
    plotter.add_scalar_bar(title='Wall Thickness (mm)', vertical=True)

    # Камера: смотрим на предсердие
    plotter.camera_position = 'xy'  # или 'yz', 'xz' — подбери
    plotter.show()

    # Сохранение
    if save_path:
        plotter.screenshot(save_path, transparent_background=False, scale=2)
        print(f"✓ Скриншот сохранён: {save_path}")


    return thickness_filt

def save_surface_points(awt_field, epi_indices, endo_indices, spacing, filepath):
    """
    Сохраняет граничные точки с толщиной в простой формат.
    """
    # Эпикард
    epi_thickness = awt_field[epi_indices[:, 0], epi_indices[:, 1], epi_indices[:, 2]]

    # Эндокард
    endo_thickness = awt_field[endo_indices[:, 0], endo_indices[:, 1], endo_indices[:, 2]]

    np.savez_compressed(
            filepath.replace('.npz', '_endo.npz'),
            points=endo_indices,
            thickness=endo_thickness,
            spacing=spacing
    )

    np.savez_compressed(
            filepath.replace('.npz', '_epi.npz'),
            points=epi_indices,
            thickness=epi_thickness,
            spacing=spacing
    )

    print(f"✓ Saved: {filepath.replace('.npz', '_endo.npz')}")
    print(f"✓ Saved: {filepath.replace('.npz', '_epi.npz')}")

# =============================================================================
# Main Pipeline (EXACT USER PIPELINE)
# =============================================================================

def one_file(path: str):
    print(path)
    # print("=" * 60, "\nBI-ATRIAL WALL THICKNESS PIPELINE\n", "=" * 60)

    # Step 0: Load
    print("\n[0/7] Loading...")
    N: int = 1
    read_status = False
    while not read_status:
        if N > 100:
            return None, None, None
        try:
            heart = sitk.ReadImage(path)
            space = sitk.GetArrayFromImage(heart)
            wall_mask = (space > 0).astype(np.uint8)
            original_spacing = heart.GetSpacing()
            original_shape = wall_mask.shape
            # del heart
            read_status = True
        except RuntimeError:
            N+=1
            gc.collect()
            time.sleep(0.1)
    print(N, "requests to read were asked.")

    print_volume(wall_mask, original_spacing, "wall")
    print(f"  Spacing: {original_spacing} mm")
    print(f"  Original shape: {original_shape}")

    # Step 1: Bounding Box
    print("\n[1/7] Computing bounding box...")
    pad = 2
    bbox = get_bounding_box(wall_mask, pad)
    (x0, x1), (y0, y1), (z0, z1) = bbox
    # print(f"  Bounding box: {bbox}")

    reduced_wall_mask = wall_mask[x0:x1, y0:y1, z0:z1]
    print(f"  Reduced shape: {reduced_wall_mask.shape}")

    print("\n[2/7] Building convex hull (intersection of 3 axes)...")
    convex_hull = get_convex_hull_intersection(reduced_wall_mask, n_jobs=12)
    print_volume(convex_hull, original_spacing, "Convex Hull")
    # print("Wall entirely in convex hull:", np.all(reduced_wall_mask <= convex_hull))

    print("\n[3/11] Extracting cavity and surfaces...")
    erosion_iters = 7
    dilation_iters = 7

    endo_coarse, epi_coarse, cavity_coarse = extract_surface(
        reduced_wall_mask,
        convex_hull,
        erosion_iters,
        dilation_iters
    )

    print("\n[4/7] Upscaling grid (×4)...")
    first_factor = 4
    another_factor = 2

    wall_fine = upscale_grid(reduced_wall_mask, factor=first_factor)
    epi_fine = upscale_grid(epi_coarse, factor=first_factor)
    endo_fine = upscale_grid(endo_coarse, factor=first_factor)
    del epi_coarse, endo_coarse, cavity_coarse, convex_hull

    fine_spacing = tuple(s / first_factor for s in original_spacing)

    print("\n[5/7] Creating field and interior indices (strip + downscale)...")
    wall_field, epi_field, endo_field, interior_indices, episurface_indices, endosurface_indices =\
        create_field_and_interior_indices(
            wall_fine, epi_fine, endo_fine, factor=another_factor
    )
    del wall_fine, epi_fine, endo_fine
    field_spacing = tuple(another_factor * s for s in fine_spacing)
    

    print("\n[6/7] Solving Laplace equation (interior nodes only)...")

    # Initialize u with boundary values
    u_field = np.zeros_like(wall_field, dtype=np.float64)
    u_field[epi_field > 0] = 100.0  # Epicardium (Paper: 100)
    u_field[endo_field > 0] = 300.0  # Endocardium (Paper: 300)
    u_field[endo_field * epi_field > 0] = 200.0  # Boundary assumption

    # Solve on interior nodes only
    u_field, grad_field, stats_laplace = solve_laplace_interior(
        u_field,
        interior_indices,
        spacing=field_spacing,
        max_iter=5000,
        tol=1e-6
    )

    del u_field
    gc.collect()
    print("\n[7/7] Computing wall thickness (Coupled PDE)...")
    # d_epi, d_endo,
    awt_field, stats = compute_wall_thickness_interior(
        grad_field,
        interior_indices,
        episurface_indices,
        endosurface_indices,
        spacing=field_spacing
    )
    print(np.std(awt_field[awt_field!=0]))

    restored_awt = np.zeros_like(wall_mask, dtype=awt_field.dtype)
    # restored_u = np.zeros_like(wall_mask, dtype=u_field.dtype)

    restored_awt[x0:x1, y0:y1, z0: z1] = awt_field[::2, ::2, ::2]
    # restored_u[x0:x1, y0:y1, z0: z1] = u_field[::2, ::2, ::2]

    awt_img = sitk.GetImageFromArray(restored_awt)
    awt_img.SetSpacing(heart.GetSpacing())
    awt_img.SetDirection(heart.GetDirection())
    awt_img.SetOrigin(heart.GetOrigin())
    sitk.WriteImage(awt_img, path.split("nrrd")[0] + "AWT.nrrd")

    save_surface_points(
        awt_field,
        episurface_indices,
        endosurface_indices,
        field_spacing,
        filepath='surface_points.npz')
    print(stats)


if __name__ == '__main__':
    import os
    import sys

    name = "Ero.nrrd"
    # path = "Датасеты/Dataset001_LA_Wall/labelsTs"
    path = "Датасеты/Dataset001_LA_Wall/strided_predictions"
    one_file(os.path.join(path, name))
    gc.collect()

    quick_plot_surface_points(
        'surface_points.npz',
        clim=(0.001, 5.0),  # фиксируем диапазон: 0.5-4 мм
        save_path='atrium_view.png'
    )
