import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances


def load_endo_points(filepath):
    data = np.load(filepath)
    return data['points'], data['thickness'], data['spacing']


def compute_geodesic_distances(voxel_indices, spacing):
    """
    Приближение геодезических расстояний через граф ближайших соседей.
    """
    coords = voxel_indices * np.array(spacing)
    n = len(coords)

    # Строим граф k-ближайших соседей
    print("  Building k-NN graph...")
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=10)

    # Создаём разреженную матрицу смежности
    rows = np.repeat(np.arange(n), 10)
    cols = indices.flatten()
    data = distances.flatten()

    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = (graph + graph.T) / 2  # делаем симметричной

    return graph


def umap_embedding(voxel_indices, thickness, spacing, n_components=2,
                   n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    UMAP embedding для 3D→2D.
    """
    coords = voxel_indices * np.array(spacing)

    print(f"UMAP embedding: {len(coords)} points → {n_components}D")
    print(f"  n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")

    if metric == 'precomputed':
        # Геодезические расстояния
        distances = compute_geodesic_distances(voxel_indices, spacing)
        embedding = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='precomputed',
            random_state=42
        ).fit_transform(distances)
    else:
        # Обычные евклидовы расстояния
        embedding = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        ).fit_transform(coords)

    return embedding


def plot_umap(endo_path, output_prefix='umap', metric='euclidean'):
    print(f"Loading: {endo_path}")
    voxel_indices, thickness, spacing = load_endo_points(endo_path)
    print(f"  {len(voxel_indices)} voxels")

    print("Running UMAP...")
    embedding = umap_embedding(
        voxel_indices, thickness, spacing,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric=metric
    )

    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # 2D scatter
    sc = ax1.scatter(embedding[:, 0], embedding[:, 1], c=thickness,
                     cmap='magma', s=2, alpha=0.7)
    ax1.set_title(f'UMAP 2D Embedding ({metric})')
    ax1.set_xlabel('UMAP-1');
    ax1.set_ylabel('UMAP-2')
    ax1.set_aspect('equal')
    plt.colorbar(sc, ax=ax1, label='Thickness (mm)')

    # 3D original (для сравнения)
    coords = voxel_indices * np.array(spacing)
    ax2 = fig.add_subplot(122, projection='3d')
    sc3d = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=thickness, cmap='magma', s=1, alpha=0.5)
    ax2.set_title('Original 3D')
    plt.colorbar(sc3d, ax=ax2, label='Thickness (mm)')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_{metric}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_{metric}.png")
    plt.show()

    # Сохраняем embedding
    np.savez_compressed(
        f'{output_prefix}_{metric}_data.npz',
        embedding=embedding,
        thickness=thickness,
        coords_3d=coords
    )


if __name__ == '__main__':
    # Вариант 1: евклидовы расстояния (быстро)
    plot_umap(
        '/home/evgeniy/Рабочий стол/Научная работа/Данные/surface_points_endo.npz',
              output_prefix='LA_umap',
              metric='euclidean'
              )

    # Вариант 2: геодезические расстояния (медленнее, но лучше для поверхности)
    # plot_umap('surface_points_endo.npz', output_prefix='LA_umap', metric='precomputed')