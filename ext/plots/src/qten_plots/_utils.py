from typing import Optional, Tuple, List

import torch


def compute_bonds(
    coords: torch.Tensor, dim: int
) -> Tuple[
    List[Optional[float]], List[Optional[float]], Optional[List[Optional[float]]]
]:
    """
    Generate bond lines connecting nearest neighbors using PyTorch.
    Returns (x_lines, y_lines, z_lines) where lists contain coordinates separated by None.
    z_lines is None if dim != 3.
    """
    if coords.size(0) < 2:
        return [], [], None if dim != 3 else None

    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dists = torch.norm(diff, dim=-1)

    dists.fill_diagonal_(float("inf"))

    min_dist = torch.min(dists)
    if torch.isinf(min_dist):
        return [], [], None if dim != 3 else None

    tol = 1e-4
    pairs = torch.nonzero(dists <= min_dist + tol)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]

    if pairs.size(0) == 0:
        return [], [], None if dim != 3 else None

    p1 = coords[pairs[:, 0]]
    p2 = coords[pairs[:, 1]]

    p1_np = p1.numpy()
    p2_np = p2.numpy()

    x_lines: List[Optional[float]] = []
    y_lines: List[Optional[float]] = []
    z_lines: Optional[List[Optional[float]]] = [] if dim == 3 else None
    nan = None

    for i in range(len(p1_np)):
        x_lines.extend([p1_np[i, 0], p2_np[i, 0], nan])
        y_lines.extend([p1_np[i, 1], p2_np[i, 1], nan])
        if dim == 3 and z_lines is not None:
            z_lines.extend([p1_np[i, 2], p2_np[i, 2], nan])

    return x_lines, y_lines, z_lines
