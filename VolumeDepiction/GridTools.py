from typing import Tuple

import numpy as np
from numba import jit, prange

# =============================================================================
# Grid Operations (Upscale, Strip, Downscale, etc.)
# =============================================================================

def upscale_grid(coarse_mask: np.ndarray, factor: int = 4) -> np.ndarray:
    """
    Upscale by repeating each voxel factor×factor×factor times.
    Each original voxel goes to cube of factor ^ N_dims new voxels.
    """
    fine = np.copy(coarse_mask)
    for axis in np.arange(len(coarse_mask.shape)):
        fine = np.repeat(fine, factor, axis=axis)
    return fine.astype(np.uint8)


def strip_boundary_layers(fine_mask: np.ndarray, layers: int = 1) -> np.ndarray:
    """
    Remove external layers from grid.
    This ensures boundary nodes are excluded from PDE indices.
    """
    assert layers > 0 and isinstance(layers, int), IndexError("'Layer' should be non-negative integer!")
    return fine_mask[layers:-layers, layers:-layers, layers:-layers].copy()


@jit(nopython=True, parallel=True, fastmath=True)
def downscale_with_stride(fine_mask: np.ndarray, block_size: int = 2, stride: int = 2) -> np.ndarray:
    """
    Downscale with filter and stride.
    Applies ones filter (block_size³) with given stride.
    For binary masks: OR operation over each block.
    """
    shape = fine_mask.shape
    new_shape = (
        (shape[0] - block_size) // stride + 1,
        (shape[1] - block_size) // stride + 1,
        (shape[2] - block_size) // stride + 1
    )

    downsampled = np.zeros(new_shape, dtype=fine_mask.dtype)

    for i in prange(new_shape[0]):
        for j in prange(new_shape[1]):
            for k in prange(new_shape[2]):
                block = fine_mask[
                    i * stride:i * stride + block_size,
                    j * stride:j * stride + block_size,
                    k * stride:k * stride + block_size
                ]
                downsampled[i, j, k] = np.any(block)

    return downsampled.astype(np.uint8)


def create_field_and_interior_indices(
        wall_fine: np.ndarray,
        epi_fine: np.ndarray,
        endo_fine: np.ndarray,
        factor: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline:
    1. Strip boundary layers from upscaled masks
    2. Downscale with 2×2×2 filter, stride 2
    3. Field = interior wall voxels
    4. Interior indices = nodes where PDE will be solved
    """
    print("[Grid] Creating field and interior indices...")

    # Strip external layers (remove boundary voxels)
    wall_stripped = strip_boundary_layers(wall_fine, layers=1)
    epi_stripped = strip_boundary_layers(epi_fine, layers=1)
    endo_stripped = strip_boundary_layers(endo_fine, layers=1)

    print(f"  Stripped shape: {wall_stripped.shape} (was {wall_fine.shape})")

    # Downscale with 2×2×2 filter, stride 2
    # This creates nodes on faces/edges/vertices of original fine grid
    wall_field = downscale_with_stride(wall_stripped, block_size=factor, stride=factor)
    epi_field = downscale_with_stride(epi_stripped, block_size=factor, stride=factor)
    endo_field = downscale_with_stride(endo_stripped, block_size=factor, stride=factor)

    print(f"  Field shape: {wall_field.shape}")

    # Create interior indices (exclude boundary nodes)
    boundary_field = (epi_field > 0) | (endo_field > 0)
    interior_mask = wall_field > boundary_field
    interior_indices = np.argwhere(interior_mask).astype(np.int32)
    episurface_indices = np.argwhere((epi_field > 0) & wall_field)
    endosurface_indices = np.argwhere((endo_field > 0) & wall_field)

    return wall_field, epi_field, endo_field, interior_indices, episurface_indices, endosurface_indices
