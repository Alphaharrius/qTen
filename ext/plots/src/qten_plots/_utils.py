from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree


def compute_bonds(
    coords: torch.Tensor, dim: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate bond lines connecting nearest neighbors.

    Uses a KDTree for O(N log N) neighbor search instead of O(N²) pairwise
    distances, making this feasible for large lattices (100k+ sites).

    Returns (x_lines, y_lines, z_lines) where arrays contain coordinates
    separated by NaN (line-break sentinel for both Plotly and Matplotlib).
    z_lines is None when *dim* != 3.
    """
    _empty = np.empty(0, dtype=np.float64)
    n = coords.size(0)
    if n < 2:
        return _empty, _empty.copy(), (_empty.copy() if dim == 3 else None)

    pts = coords.numpy().astype(np.float64)
    tree = cKDTree(pts)

    dd, _ = tree.query(pts, k=2)
    min_dist = float(dd[:, 1].min())
    if np.isinf(min_dist):
        return _empty, _empty.copy(), (_empty.copy() if dim == 3 else None)

    pairs = tree.query_pairs(r=min_dist + 1e-4, output_type="ndarray")
    if len(pairs) == 0:
        return _empty, _empty.copy(), (_empty.copy() if dim == 3 else None)

    p1 = pts[pairs[:, 0]]
    p2 = pts[pairs[:, 1]]
    n_bonds = len(pairs)
    n_cols = pts.shape[1]

    segments = np.empty((n_bonds, 3, n_cols), dtype=np.float64)
    segments[:, 0, :] = p1
    segments[:, 1, :] = p2
    segments[:, 2, :] = np.nan
    flat = segments.reshape(-1, n_cols)

    x_lines = flat[:, 0]
    y_lines = flat[:, 1]
    z_lines = flat[:, 2] if dim == 3 and n_cols >= 3 else None

    return x_lines, y_lines, z_lines
