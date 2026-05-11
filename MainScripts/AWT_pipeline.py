import gc
import os
import time
import multiprocessing

import numpy as np
import SimpleITK as sitk
import skimage.measure

from VolumeDepiction.ConvexHullIntersection import get_convex_hull_intersection
from VolumeDepiction.GridTools import upscale_grid, create_field_and_interior_indices
from VolumeDepiction.SurfaceExtraction import external_and_internal_surfaces
from WallThicknessTools.LaplasSolver import solve_laplace_interior
from WallThicknessTools.PairedDEMethod import compute_wall_thickness_interior
from Tools.WorkTools import get_bounding_box, get_largest_domain
from Tools.SaveTools import save_mask
from Tools.LoggingTools import print_volume


def get_beta1(mask: np.ndarray):
    verts, faces, _, _ = skimage.measure.marching_cubes(mask, level=0.5)
    return 2 - (len(verts) - len(faces) / 2)


def process(path: str, name: str, out: str):
    print("\n[0/7] Loading...")
    path = os.path.join(path, name)
    N: int = 1
    while N < 100:
        try:
            heart = sitk.ReadImage(path)
            space = sitk.GetArrayFromImage(heart)
            wall_mask = (space > 0).astype(np.uint8)
            wall_mask, _ = get_largest_domain(wall_mask)
            original_spacing = heart.GetSpacing()
            job_spacing = original_spacing[::-1]
            direction = heart.GetDirection()
            origin = heart.GetOrigin()
            original_shape = wall_mask.shape
            del space
            break
        except RuntimeError:
            N+=1
            gc.collect()
            time.sleep(0.1)
    else:
        print("Возникла неизвестная ошибка, не смогли загрузить файл за 100 попыток.")
        return
    print(f"  {N} requests to read were asked.")
    print_volume(wall_mask, original_spacing, "wall")
    print(f"  Spacing: {original_spacing} mm\n  Original shape: {original_shape}")
    out = os.path.join(out, name.split(".nrrd")[0])
    if not os.path.exists(out): os.mkdir(out)

    print("\n[1/7] Computing bounding box...")
    pad = 2 # Constant, may be more but now there are no reasons for it.
    bbox = get_bounding_box(wall_mask, pad)
    (z0, z1), (y0, y1), (x0, x1) = bbox
    reduced_wall_mask = wall_mask[z0:z1, y0:y1, x0:x1]
    new_origin = list(heart.GetOrigin())
    new_origin[0] += z0 * original_spacing[0]  # Z
    new_origin[1] += y0 * original_spacing[1]  # Y
    new_origin[2] += x0 * original_spacing[2]  # X
    print(f"  Reduced shape: {reduced_wall_mask.shape}")

    print("\n[2/7] Building convex hull (intersection of 3 axes)...")
    convex_hull = get_convex_hull_intersection(
        reduced_wall_mask,
        n_jobs=multiprocessing.cpu_count()
    )
    print_volume(convex_hull, job_spacing, "Convex Hull")

    print("\n[3/11] Extracting surfaces...")
    erosion_iters = 7   # There are now reasons for exactly 7, but for correct restoring the both should be equal.
    dilation_iters = 7
    endo_coarse, epi_coarse = external_and_internal_surfaces(
        reduced_wall_mask,
        convex_hull,
        erosion_iters,
        dilation_iters
    )

    print("\n[4/7] Upscaling grid (×4)...")
    # We include elements' vertices, faces and edges centers into grid for exact calculation of surfaces.
    first_factor = 4    # Divide every block onto 4 elements along each side.
    second_factor = 2  # Unite every 2 blocks along each side into one unit.
    wall_fine = upscale_grid(reduced_wall_mask, factor=first_factor)
    print(f"  Independent paths through the wall:  {get_beta1(wall_fine)}")
    epi_fine = upscale_grid(epi_coarse, factor=first_factor)
    endo_fine = upscale_grid(endo_coarse, factor=first_factor)
    fine_spacing = tuple(s / first_factor for s in job_spacing)
    del epi_coarse, endo_coarse, reduced_wall_mask, convex_hull
    gc.collect()

    print("\n[5/7] Creating field and interior indices (strip + downscale)...")
    wall_field, epi_field, endo_field, interior_indices, episurface_indices, endosurface_indices =\
        create_field_and_interior_indices(
            wall_fine, epi_fine, endo_fine, factor=second_factor
    )
    field_spacing = (second_factor * fine_spacing[0], second_factor * fine_spacing[1], second_factor * fine_spacing[2])
    del wall_fine, epi_fine, endo_fine

    print("\n[6/7] Solving Laplace equation (interior nodes only)...")
    u_field = np.zeros_like(wall_field, dtype=np.float64)
    u_field[epi_field > 0] = 100.0  # Epicardium (Paper: 100)
    u_field[endo_field > 0] = 300.0  # Endocardium (Paper: 300)
    u_field[endo_field * epi_field > 0] = 200.0  # Boundary assumption
    u_field, grad_field, _ = solve_laplace_interior(
        u_field,
        interior_indices,
        field_spacing,
        max_iter=5000,
        tol=1e-6
    )
    save_mask(u_field, original_spacing, direction, new_origin, out, "Dirichlet_solution.nrrd")
    del u_field, epi_field, endo_field
    gc.collect()

    print("\n[7/7] Computing wall thickness (Coupled PDE)...")
    awt_field, stats = compute_wall_thickness_interior(
        grad_field,
        interior_indices,
        episurface_indices,
        endosurface_indices,
        field_spacing
    )
    save_mask(awt_field, original_spacing, direction, new_origin, out, "AWT.nrrd")
    with open(os.path.join(out, "AWT_stats.txt"), 'w') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v[0]} {v[1]}\n")
    del awt_field, grad_field, interior_indices, episurface_indices, endosurface_indices
    gc.collect()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Программа для вычисления получения предсердия. "
                                                 "Так же дополнительно вычисляет маску пула крови и "
                                                 "решение задачи Дирихле.")

    parser.add_argument("-p", "--path", required=True, type=str, help="Расположение файла")
    parser.add_argument("-n", "--names", nargs="*", type=str, help="Имя образца", default=[])
    parser.add_argument("-o", "--out", type=str, help="Папка для сохранения", default="../Results")
    args = parser.parse_args()

    if args.names:
        names = [n for n in args.names
                 if n.endswith(".nrrd") and os.path.isfile(os.path.join(args.path, n))]
    else:
        names = [f for f in os.listdir(args.path)
                 if f.endswith(".nrrd") and os.path.isfile(os.path.join(args.path, f))]
    if not names:
        print("Предупреждение: файлы .nrrd не найдены.")
        exit(1)

    for name in names:
        print("Sample:", name)
        process(args.path, name, args.out)
        gc.collect()
