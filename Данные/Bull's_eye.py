from math import sqrt, tan
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import cg
from scipy.interpolate import griddata
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree
from numba import jit, prange


# =============================================================================
# 1. Загрузка
# =============================================================================
def load_endo_points(filepath):
    data = np.load(filepath)
    return data['points'], data['thickness'], data['spacing']


# =============================================================================
# 2. Воксели → меш (с децимацией)
# =============================================================================
def indices_to_mesh_decimated(voxel_indices, spacing, target_voxels=50000):
    if len(voxel_indices) > target_voxels:
        print(f"  Subsampling voxels: {len(voxel_indices)} → {target_voxels}")
        np.random.seed(42)
        sample_idx = np.random.choice(len(voxel_indices), target_voxels, replace=False)
        voxel_indices = voxel_indices[sample_idx]

    min_idx = voxel_indices.min(axis=0)
    max_idx = voxel_indices.max(axis=0)
    shape = (max_idx - min_idx + 3).astype(int)
    mask = np.zeros(shape, dtype=np.float64)

    shifted_indices = voxel_indices - min_idx
    valid_mask = (
            (shifted_indices[:, 0] >= 0) & (shifted_indices[:, 0] < shape[0]) &
            (shifted_indices[:, 1] >= 0) & (shifted_indices[:, 1] < shape[1]) &
            (shifted_indices[:, 2] >= 0) & (shifted_indices[:, 2] < shape[2])
    )
    valid_indices = shifted_indices[valid_mask]
    mask[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1.0

    vertices, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    vertices = (vertices + min_idx) * np.array(spacing)

    return vertices, faces, voxel_indices


# =============================================================================
# 3. NUMBA: Котангентные веса (ГЛАВНАЯ ОПТИМИЗАЦИЯ)
# =============================================================================
@jit(nopython=True, parallel=True, fastmath=True)
def compute_cotangent_weights_numba(vertices: np.ndarray, faces: np.ndarray, n: int)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-ускоренное вычисление котангентных весов.
    Возвращает: row_idx, col_idx, weights для COO матрицы.
    """
    max_entries = len(faces) * 6
    row_idx = np.zeros(max_entries, dtype=np.int32)
    col_idx = np.zeros(max_entries, dtype=np.int32)
    weights = np.zeros(max_entries, dtype=np.float64)
    # entry_count = 0

    for f in prange(len(faces)):
        i, j, k = faces[f, 0], faces[f, 1], faces[f, 2]
        vi, vj, vk = vertices[i], vertices[j], vertices[k]

        e_ij = vj - vi
        e_jk = vk - vj
        e_ki = vi - vk

        len_ij = sqrt(np.sum(e_ij ** 2)) + 1e-10
        len_jk = sqrt(np.sum(e_jk ** 2)) + 1e-10
        len_ki = sqrt(np.sum(e_ki ** 2)) + 1e-10

        # Ручной clamp (вместо np.clip который не работает в numba)
        cos_i = -np.dot(e_ki, e_ij) / (len_ki * len_ij)
        cos_i = -1.0 if cos_i < -1.0 else 1.0 if cos_i > 1.0 else cos_i

        cos_j = -np.dot(e_ij, e_jk) / (len_ij * len_jk)
        cos_j = -1.0 if cos_j < -1.0 else 1.0 if cos_j > 1.0 else cos_j

        cos_k = -np.dot(e_jk, e_ki) / (len_jk * len_ki)
        cos_k = -1.0 if cos_k < -1.0 else 1.0 if cos_k > 1.0 else cos_k

        angle_i = np.arccos(cos_i)
        angle_j = np.arccos(cos_j)
        angle_k = np.arccos(cos_k)

        cot_i = 1.0 / (tan(angle_i) + 1e-10)
        cot_j = 1.0 / (tan(angle_j) + 1e-10)
        cot_k = 1.0 / (tan(angle_k) + 1e-10)

        # (i,j) и (j,i)
        entry_count = f * 6
        row_idx[entry_count], col_idx[entry_count], weights[entry_count] = i, j, cot_k
        entry_count += 1
        row_idx[entry_count], col_idx[entry_count], weights[entry_count] = j, i, cot_k
        entry_count += 1

        # (j,k) и (k,j)
        row_idx[entry_count], col_idx[entry_count], weights[entry_count] = j, k, cot_i
        entry_count += 1
        row_idx[entry_count], col_idx[entry_count], weights[entry_count] = k, j, cot_i
        entry_count += 1

        # (k,i) и (i,k)
        row_idx[entry_count], col_idx[entry_count], weights[entry_count] = k, i, cot_j
        entry_count += 1
        row_idx[entry_count], col_idx[entry_count], weights[entry_count] = i, k, cot_j

    return row_idx[:len(faces) * 6], col_idx[:len(faces) * 6], weights[:len(faces) * 6]

# @jit(nopython=True, parallel=True, fastmath=True)
def compute_cotangent_weights_fast(vertices, faces):
    """Обёртка: numba → scipy sparse."""
    n = len(vertices)
    row_idx, col_idx, weights = compute_cotangent_weights_numba(vertices, faces, n)
    M = coo_matrix((weights, (row_idx, col_idx)), shape=(n, n))
    return M.tocsr()


# =============================================================================
# 4. Граница на квадрат
# =============================================================================
def map_boundary_to_square(vertices, boundary_indices):
    n = len(vertices)
    boundary_coords = np.zeros((n, 2))
    if len(boundary_indices) < 3:
        return boundary_coords

    boundary_verts = vertices[boundary_indices]
    center = np.mean(boundary_verts, axis=0)
    angles = np.arctan2((boundary_verts - center)[:, 1], (boundary_verts - center)[:, 0])
    sorted_indices = boundary_indices[np.argsort(angles)]

    edge_lengths = np.array([
        np.linalg.norm(vertices[sorted_indices[(i + 1) % len(sorted_indices)]] - vertices[sorted_indices[i]])
        for i in range(len(sorted_indices))
    ])
    cumlen = np.cumsum([0] + list(edge_lengths))

    for idx, s in zip(sorted_indices, cumlen):
        t = (s / cumlen[-1]) * 4.0
        if t < 1:
            boundary_coords[idx] = [t, 0]
        elif t < 2:
            boundary_coords[idx] = [1, t - 1]
        elif t < 3:
            boundary_coords[idx] = [3 - t, 1]
        else:
            boundary_coords[idx] = [0, 4 - t]

    return boundary_coords

# @jit(nopython=True, parallel=True, fastmath=True)
def M_filling(internal_indices: List[int], weights: csr_matrix, idx_map: Dict[int, int], M: lil_matrix):
    for new_i, old_i in enumerate(internal_indices):
        row_start = weights.indptr[old_i]
        row_end = weights.indptr[old_i + 1]
        for ptr in prange(row_start, row_end):
            old_j = weights.indices[ptr]
            w = weights.data[ptr]
            if w == 0:
                continue
            if old_j in internal_indices:
                new_j = idx_map[old_j]
                M[2 * new_i, 2 * new_j] += w
                M[2 * new_i + 1, 2 * new_j + 1] += w
                M[2 * new_i, 2 * new_i] -= w
                M[2 * new_i + 1, 2 * new_i + 1] -= w

# =============================================================================
# 5. Гармоническая параметризация
# =============================================================================
def flatten_mesh_harmonic(vertices, faces, boundary_indices):
    n = len(vertices)
    internal_indices = list(set(range(n)) - set(boundary_indices))
    n_internal = len(internal_indices)

    if n_internal == 0:
        return np.zeros((n, 2))

    print(f"  Building sparse matrix ({n_internal} internal nodes)...")
    weights = compute_cotangent_weights_fast(vertices, faces)

    M = lil_matrix((2 * n_internal, 2 * n_internal))
    idx_map = {old: new for new, old in enumerate(internal_indices)}
    print("M filling")
    M_filling(internal_indices, weights, idx_map, M)

    M = M.tocsr()
    boundary_coords = map_boundary_to_square(vertices, boundary_indices)

    C = np.zeros(2 * n_internal)
    print("C filling")
    for new_i, old_i in enumerate(internal_indices):
        row_start = weights.indptr[old_i]
        row_end = weights.indptr[old_i + 1]
        for ptr in range(row_start, row_end):
            old_j = weights.indices[ptr]
            w = weights.data[ptr]
            if old_j in boundary_indices and w != 0:
                C[2 * new_i] -= w * boundary_coords[old_j, 0]
                C[2 * new_i + 1] -= w * boundary_coords[old_j, 1]

    print(f"  Solving linear system (2×{n_internal} equations)...")
    U_flat, info = cg(M, C, tol=1e-8, maxiter=3000)

    if info != 0:
        print(f"  ⚠️ CG did not converge (info={info})")

    uv = np.zeros((n, 2))
    uv[internal_indices, 0] = U_flat[0::2]
    uv[internal_indices, 1] = U_flat[1::2]
    uv[boundary_indices] = boundary_coords[boundary_indices]

    return uv


# =============================================================================
# 6. Найти границу
# =============================================================================
def find_boundary(vertices, axis=2, percentile=10):
    coords = vertices[:, axis]
    threshold = np.percentile(coords, percentile)
    return np.where(coords <= threshold)[0]


# =============================================================================
# 7. Проекция на 2D
# =============================================================================
def project_to_2d_grid(uv, thickness, grid_size=256, clim=(0.5, 4.0)):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = griddata(uv, thickness, np.column_stack([X.ravel(), Y.ravel()]), method='linear').reshape(grid_size, grid_size)
    tri = Triangulation(uv[:, 0], uv[:, 1])
    Z[tri.get_mask_for_grid(X, Y)] = np.nan
    return np.clip(Z, clim[0], clim[1]), X, Y


# =============================================================================
# 8. Основная функция
# =============================================================================
def plot_bulls_eye(endo_path, output_prefix='bullseye', clim=(0.5, 4.0)):
    print(f"Loading: {endo_path}")
    voxel_indices, thickness, spacing = load_endo_points(endo_path)
    print(f"  {len(voxel_indices)} voxels")

    print("Creating mesh (with decimation)...")
    vertices, faces, sampled_voxel_indices = indices_to_mesh_decimated(
        voxel_indices, spacing, target_voxels=50000
    )
    print(f"  {len(vertices)} vertices, {len(faces)} faces")

    # Пересэмплирование толщины
    tree = cKDTree(vertices)
    voxel_coords = sampled_voxel_indices * np.array(spacing)
    distances, indices = tree.query(voxel_coords, k=1)

    vertex_thickness = np.zeros(len(vertices))
    for i, idx in enumerate(indices):
        vertex_thickness[idx] = max(vertex_thickness[idx], thickness[i])

    print("Finding boundary...")
    boundary_indices = find_boundary(vertices, axis=2, percentile=10)
    print(f"  {len(boundary_indices)} boundary vertices")

    print("Flattening (numba accelerated)...")
    uv = flatten_mesh_harmonic(vertices, faces, boundary_indices)
    print("Done")

    print("Projecting to 2D...")
    Z, X, Y = project_to_2d_grid(uv, vertex_thickness, grid_size=256, clim=clim)

    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    im = ax1.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='magma', clim=clim)
    ax1.set_title('Flat Thickness Map (Bull\'s Eye)')
    ax1.set_xlabel('u');
    ax1.set_ylabel('v');
    ax1.set_aspect('equal')
    plt.colorbar(im, ax=ax1, label='Thickness (mm)')
    sc = ax2.scatter(uv[:, 0], uv[:, 1], c=vertex_thickness, cmap='magma', clim=clim, s=2)
    ax2.set_title('Vertex Thickness on 2D')
    ax2.set_xlabel('u');
    ax2.set_ylabel('v');
    ax2.set_aspect('equal')
    plt.colorbar(sc, ax=ax2, label='Thickness (mm)')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}.png")
    plt.show()
    np.savez_compressed(f'{output_prefix}_data.npz', uv=uv, thickness=vertex_thickness, grid_Z=Z)
    print(f"✓ Data saved: {output_prefix}_data.npz")


# =============================================================================
# 9. Запуск
# =============================================================================
if __name__ == '__main__':
    plot_bulls_eye(
        endo_path='surface_points_endo.npz',
        output_prefix='LA_bullseye',
        clim=(0.5, 4.0)
    )