# =============================================================================
# Surface Extraction (Original Resolution - BEFORE upscaling)
# =============================================================================
import numpy as np
from scipy import ndimage

from WorkTools import print_volume


def extract_surface(wall_mask, convex_hull,
                    erosion_iterations=8,
                    dilation_iterations=8,
                    dim=(1, 1, 1)):
    """
    Извлекает поверхности как внешние воксели (не из wall_mask).
    Epicardial: вне стенки, вне полости (внешняя поверхность)
    Endocardial: вне стенки, внутри полости (внутренняя поверхность)
    """
    structure = ndimage.generate_binary_structure(3, 1)

    # -------------------------------------------------------------------------
    # Шаг 1-4: Извлечение полости
    # -------------------------------------------------------------------------
    potential_cavity = convex_hull * (1 - wall_mask)
    potential_cavity = ndimage.binary_opening(potential_cavity, iterations=1)

    labeled, num_features = ndimage.label(potential_cavity)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        largest_component = np.argmax(component_sizes[1:]) + 1
        potential_cavity = (labeled == largest_component).astype(np.uint8)
        print_volume(potential_cavity, dim, "потенциальной полости")
    else:
        print("Пустая полость")
        return np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1))

    eroded_cavity = ndimage.binary_erosion(convex_hull, structure=structure,
                                           iterations=erosion_iterations)
    print_volume(eroded_cavity, dim, "после эрозии")

    constrained_core = eroded_cavity & potential_cavity
    labeled, num_features = ndimage.label(constrained_core)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        largest_component = np.argmax(component_sizes[1:]) + 1
        constrained_core = (labeled == largest_component).astype(np.uint8)
    print_volume(constrained_core, dim, "ядра после ограничения")

    cavity = constrained_core.copy()
    print(f"  Запуск итеративной дилатации ({dilation_iterations} шагов)...")

    for step in range(dilation_iterations):
        cavity_dilated = ndimage.binary_dilation(cavity, structure=structure, iterations=1)
        cavity = cavity_dilated & potential_cavity

        labeled, num_features = ndimage.label(cavity)
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            largest_component = np.argmax(component_sizes[1:]) + 1
            cavity = (labeled == largest_component).astype(np.uint8)

        if step % 2 == 0:
            print(f"    Шаг {step + 1:2d}: объём = {np.sum(cavity):>8} вокселей")

    cavity = ndimage.binary_opening(cavity, iterations=1)
    cavity = ndimage.binary_closing(cavity, iterations=1)
    labeled, num_features = ndimage.label(cavity)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        largest_component = np.argmax(component_sizes[1:]) + 1
        cavity = (labeled == largest_component).astype(np.uint8)

    wall_dilated = ndimage.binary_dilation(wall_mask, structure=structure, iterations=1)
    boundary_layer = wall_dilated & (1 - wall_mask)
    epicardial_surface = boundary_layer & (1 - cavity)
    endocardial_surface = boundary_layer & cavity
    epi_outside = np.sum(epicardial_surface & wall_mask) == 0
    endo_outside = np.sum(endocardial_surface & wall_mask) == 0

    print(f"  Эпикард вне стенки: {epi_outside}")
    print(f"  Эндокард вне стенки: {endo_outside}")

    # Проверка на пересечение
    overlap = np.sum(epicardial_surface & endocardial_surface)
    if overlap > 0:
        print(f"  WARNING: {overlap} вокселей перекрываются!")

    print_volume(epicardial_surface, dim, "эпикарда")
    print_volume(endocardial_surface, dim, "эндокарда")

    # Диагностика
    print(f"\n  Диагностика:")
    print(f"  wall_mask: {np.sum(wall_mask)} вокселей")
    print(f"  cavity: {np.sum(cavity)} вокселей")
    print(f"  boundary_layer: {np.sum(boundary_layer)} вокселей")
    print(f"  wall_mask * cavity: {np.sum(wall_mask & cavity)} (должно быть 0)")

    return endocardial_surface, epicardial_surface, cavity
