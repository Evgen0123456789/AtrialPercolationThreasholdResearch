import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def count_boundary_transitions(line: np.ndarray):
    n = 1
    guard = 32
    for i in nb.prange(guard, line.shape[0] - guard):
        if line[i - 1] == line[i] + 1:
            n += 1
    return n


@nb.njit(fastmath=True)
def mark_inner_line(in_line: np.ndarray, out_line: np.ndarray):
    n = False
    guard = 32
    for i in nb.prange(guard, in_line.shape[0] - guard):
        if in_line[i] == 1:
            out_line[i, 0] += 1
        else:
            if in_line[i - 1] == 1:
                n = not n
            if n:
                out_line[i, 1] += 1
    return


@nb.njit(fastmath=True)
def mark_inner_surface(in_layer: np.ndarray, out_layer: np.ndarray):
    guard = 32
    for i in nb.prange(guard, in_layer.shape[0] - guard):
        p = count_boundary_transitions(in_layer[i])
        if p % 2 == 1 and p > 1:
            mark_inner_line(in_layer[i], out_layer[i])
    for i in nb.prange(guard, in_layer.shape[1] - guard):
        p = count_boundary_transitions(in_layer[:, i])
        if p % 2 == 1 and p > 1:
            mark_inner_line(in_layer[:, i], out_layer[:, i])
    return


@nb.njit
def mark_inner_volume(in_mask: np.ndarray,
                  out_mask: np.ndarray):
    guard = 32
    for i in nb.prange(guard, in_mask.shape[0] - guard):
        mark_inner_surface(in_mask[i], out_mask[i])

    for i in nb.prange(guard, in_mask.shape[1] - guard):
        mark_inner_surface(in_mask[:, i, :], out_mask[:, i, :])

    for i in nb.prange(guard, in_mask.shape[2] - guard):
        mark_inner_surface(in_mask[:, :, i], out_mask[:, :, i])
