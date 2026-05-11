import numpy as np
import pyvista as pv

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
    plotter.add_points(endo, cmap='magma',
                       clim=clim, point_size=3, render_points_as_spheres=False,
                       label='Endocardium')

    # Эпикард (полупрозрачный)
    epi = cloud.extract_points(stype_filt == 1)
    plotter.add_points(epi, cmap='magma',
                       clim=clim, point_size=2, opacity=0.4,
                       label='Epicardium')

    plotter.show()
    if save_path:
        plotter.screenshot(save_path, transparent_background=False, scale=2)
        print(f" Скриншот сохранён: {save_path}")

if __name__=="__main__":
    quick_plot_surface_points(
        '../Данные/surface_points.npz',
        clim=(0.001, 5.0),  # фиксируем диапазон: 0.5-4 мм
        save_path='atrium_view.png'
    )