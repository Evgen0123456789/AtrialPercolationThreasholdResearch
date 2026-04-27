import time
from typing import Tuple

import numpy as np
from numba import jit, prange


# =============================================================================
# Laplace Solver (SOR - Fine Grid, Interior Nodes Only)
# =============================================================================

@jit(nopython=True, parallel=True, fastmath=True)
def _sor_step_interior(
        u_curr: np.ndarray,
        u_next: np.ndarray,
        indices: np.ndarray,
        w_x: float, w_y: float, w_z: float, w_sum: float,
        omega: float = 1.8
) -> float:
    """
    SOR step - iterates ONLY over interior nodes (boundary excluded by indices).
    """
    n = len(indices)
    max_diff = 0.0
    for idx in prange(n):
        i, j, k = indices[idx, 0], indices[idx, 1], indices[idx, 2]

        u_new = (
                        w_x * (u_curr[i - 1, j, k] + u_curr[i + 1, j, k]) +
                        w_y * (u_curr[i, j - 1, k] + u_curr[i, j + 1, k]) +
                        w_z * (u_curr[i, j, k - 1] + u_curr[i, j, k + 1])
                ) / w_sum

        u_old = u_curr[i, j, k]
        u_next[i, j, k] = (1 - omega) * u_old + omega * u_new

        diff = abs(u_curr[i, j, k] - u_next[i, j, k])
        max_diff = max(diff, max_diff)
    return max_diff


def solve_laplace_interior(
        u: np.ndarray,
        interior_indices: np.ndarray,
        spacing: Tuple[float, float, float],
        max_iter: int = 5000,
        tol: float = 1e-6,
        verbose: bool = True
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    """
    Solves Laplace on interior nodes only (boundary values fixed in u).
    """

    #region weights
    dx, dy, dz = spacing
    w_x, w_y, w_z = 1.0 / (dx ** 2), 1.0 / (dy ** 2), 1.0 / (dz ** 2)
    w_sum = 2.0 * (w_x + w_y + w_z)
    #endregion

    #region useful objects
    u_next = u.copy()
    start = time.time()
    converged = False
    residual = 0.0
    n_interior = len(interior_indices)
    if verbose:
        print(f"[Laplace] {n_interior} interior nodes, spacing={spacing}")
    #endregion

    for it in range(max_iter):
        residual = _sor_step_interior(u, u_next, interior_indices, w_x, w_y, w_z, w_sum, omega=0.9)
        u, u_next = u_next, u

        #region verbose
        if verbose and (it % 100 == 0 or residual < tol):
            print(f"[Iter {it:5d}] residual = {residual:.2e}, t = {time.time() - start:.2f}s")
        #endregion

        if residual < tol:
            converged = True
            if verbose:
                print(f"[Converged] at iteration {it}")
            break

    gx, gy, gz = np.gradient(u, dx, dy, dz)
    stats = {'converged': converged, 'residual': residual, 'time': time.time() - start}
    return u, (gx, gy, gz), stats
