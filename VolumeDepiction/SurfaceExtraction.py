import numpy as np
from scipy import ndimage

from Tools.LoggingTools import print_volume
from Tools.WorkTools import get_largest_domain


def extract_cavity(wall_mask:np.ndarray,
                    convex_hull: np.ndarray,
                    erosion_iterations:int=8,
                    dilation_iterations:int=8):
    structure = ndimage.generate_binary_structure(3, 1)
    potential_cavity = convex_hull * (1 - wall_mask) # Get close non-wall voxels
    potential_cavity, num_features = get_largest_domain(potential_cavity)
    if num_features ==0:
        print("Empty cavity")
        return np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1))
    eroded_hull = ndimage.binary_erosion(
        input=convex_hull,
        structure=structure,
        iterations=erosion_iterations)
    cavity = eroded_hull & potential_cavity
    cavity, num_features = get_largest_domain(cavity)
    # for _ in range(dilation_iterations):
    #     cavity = ndimage.binary_dilation(
    #         input=cavity,
    #         structure=structure,
    #         iterations=1
    #     ) & (1 - wall_mask)
    #     cavity, _ = get_largest_domain(cavity)
    cavity = ndimage.binary_dilation(
        input=cavity,
        structure=structure,
        iterations=dilation_iterations,
        mask= convex_hull & (1 - wall_mask)
    )
    return cavity


def external_and_internal_surfaces(wall_mask:np.ndarray,
                                   convex_hull: np.ndarray,
                                   erosion_iterations:int=8,
                                   dilation_iterations:int=8,
                                   dim:tuple[float|int, float|int, float|int]=(1, 1, 1)):
    """
    This function extracts outside boundary voxels.
    Endocardial: under the wall.
    Epicardial: out of the wall.
    Cavity: the volume under the wall.
    In case of empty object returns 1x1 arrays.
    """
    assert wall_mask.size == convex_hull.size, "Hull/mask sizes mismatch."
    structure = ndimage.generate_binary_structure(3, 1)
    cavity = extract_cavity(wall_mask, convex_hull, erosion_iterations, dilation_iterations)

    wall_dilated = ndimage.binary_dilation(wall_mask, structure=structure, iterations=1)
    boundary_layer = wall_dilated & (1 - wall_mask)
    epicardial_surface = boundary_layer & (1 - cavity)
    endocardial_surface = boundary_layer & cavity

    print(f"  Эпикард вне стенки: {np.count_nonzero(epicardial_surface & wall_mask) == 0}")
    print(f"  Эндокард вне стенки: {np.count_nonzero(endocardial_surface & wall_mask) == 0}")

    # Проверка на пересечение
    overlap = np.count_nonzero(epicardial_surface & endocardial_surface)
    if overlap > 0:
        print(f"<|====================|> WARNING: {overlap} вокселей перекрываются!")

    print_volume(epicardial_surface, dim, "of epicardium")
    print_volume(endocardial_surface, dim, "of endocardium")

    # Диагностика
    print(f"\n  Диагностика:")
    print(f"  wall_mask: {np.count_nonzero(wall_mask)} вокселей")
    print(f"  cavity: {np.count_nonzero(cavity)} вокселей")
    print(f"  boundary_layer: {np.count_nonzero(boundary_layer)} вокселей")
    print(f"  wall_mask * cavity: {np.count_nonzero(wall_mask & cavity)} (должно быть 0)")

    return endocardial_surface, epicardial_surface
