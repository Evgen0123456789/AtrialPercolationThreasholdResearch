from math import sqrt
from typing import Tuple, Dict

import numpy as np
from numba import jit, prange


# =============================================================================
# Coupled PDE Solver (Interior Nodes Only)
# =============================================================================

@jit(nopython=True, parallel=True, fastmath=True)
def _solve_trajectory_pde_interior(
        distance: np.ndarray,
        tz: np.ndarray,
        ty: np.ndarray,
        tx: np.ndarray,
        indices: np.ndarray,
        spacing: Tuple[float, float, float],
        direction: int
) -> None:
    """
    Solves trajectory PDE on interior nodes only.
    Boundary nodes have D=0 (fixed), excluded from indices.
    """
    n = len(indices)
    h_z, h_y, h_x = spacing

    for _ in range(50):
        for idx in prange(n):
            i, j, k = indices[idx, 0], indices[idx, 1], indices[idx, 2]

            gz = tz[i, j, k]
            gy = ty[i, j, k]
            gx = tx[i, j, k]

            mag = sqrt(gz * gz + gy * gy + gx * gx)
            if mag < 1e-10:
                continue

            tz_n = gz / mag
            ty_n = gy / mag
            tx_n = gx / mag

            # Upwind scheme (Eq. 12-13 from paper)
            d_z = distance[i - 1, j, k] if tz_n > 0 else distance[i + 1, j, k]
            d_y = distance[i, j - 1, k] if ty_n > 0 else distance[i, j + 1, k]
            d_x = distance[i, j, k - 1] if tx_n > 0 else distance[i, j, k + 1]

            abs_tz = abs(tz_n) / h_z
            abs_ty = abs(ty_n) / h_y
            abs_tx = abs(tx_n) / h_x

            denom = abs_ty + abs_tz + abs_tx

            numer = abs_tz * d_z + abs_ty * d_y + abs_tx * d_x
            d_new = (numer + direction) / denom

            distance[i, j, k] = d_new


def compute_wall_thickness_interior(
        grad: Tuple[np.ndarray, np.ndarray, np.ndarray],
        interior_indices: np.ndarray,
        episurface_indices: np.ndarray,
        endosurface_indices: np.ndarray,
        spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, Dict[str, Tuple[float, str]]]:
    """
    Computes AWT using coupled PDE (Eq. 7-9 from paper).
    D=0 at boundary nodes (already set before calling).
    """
    print("  Computing trajectory functions (interior nodes only)...")

    gz, gy, gx = grad
    grad_mag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2) + 1e-10
    tz = gz / grad_mag
    ty = gy / grad_mag
    tx = gx / grad_mag

    print(f"  {len(interior_indices)} interior nodes")
    print(f"  {len(episurface_indices)} epicardial nodes")
    print(f"  {len(endosurface_indices)} endocardial nodes")

    # Solve PDE from epicardium (Eq. 12)
    d_epi = np.zeros_like(tx, dtype=np.float64)
    _solve_trajectory_pde_interior(
        d_epi,
        tz, ty, tx,
        np.vstack((interior_indices, endosurface_indices)),
        spacing, direction=1)

    # Solve PDE from endocardium (Eq. 13)
    d_endo = np.zeros_like(tx, dtype=np.float64)
    _solve_trajectory_pde_interior(
        d_endo,
        -tz, -ty, -tx,
        np.vstack((interior_indices, episurface_indices)),
        spacing, direction=1)

    # print("Epi hist:", np.histogram(d_epi))
    # print("Endo hist:", np.histogram(d_endo))
    # AWT = D_epi + D_endo (Eq. 9)
    awt = d_epi + d_endo
    # Добавьте статистику в compute_wall_thickness_interior:
    awt_for_stats = awt[awt > 0]
    stats = {
        "[PDE] Mean thickness": (awt_for_stats.mean(), "mm"),
        "[PDE] Median thickness": (np.median(awt_for_stats), "mm"),
        "[PDE] 25th percentile": (np.percentile(awt_for_stats, 25), "mm"),
        "[PDE] 75th percentile": (np.percentile(awt_for_stats, 75), "mm"),
        "[PDE] 95th percentile": (np.percentile(awt_for_stats, 95), "mm"),
        "[PDE] STD": (np.std(awt_for_stats), "mm")
    }
    del awt_for_stats
    return awt, stats