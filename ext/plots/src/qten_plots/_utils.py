from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice, Offset, ReciprocalLattice
from qten.symbolics.state_space import (
    BzPath,
    MomentumSpace,
    brillouin_zone,
    same_rays,
)


_COMMON_MARKER_ALIASES: dict[str, str] = {
    "o": "circle",
    "circle": "circle",
    "s": "square",
    "square": "square",
    "d": "diamond",
    "D": "diamond",
    "diamond": "diamond",
    "+": "cross",
    "plus": "cross",
    "cross": "cross",
    "x": "x",
}

_MARKER_TO_MPL: dict[str, str] = {
    "circle": "o",
    "square": "s",
    "diamond": "D",
    "cross": "+",
    "x": "x",
}

_MARKER_TO_PLOTLY: dict[str, str] = {
    "circle": "circle",
    "square": "square",
    "diamond": "diamond",
    "cross": "cross",
    "x": "x",
}


def normalize_pointcloud_marker(marker: str | None) -> str | None:
    if marker is None:
        return None
    try:
        return _COMMON_MARKER_ALIASES[marker]
    except KeyError as e:
        supported = ", ".join(sorted(set(_COMMON_MARKER_ALIASES.values())))
        raise ValueError(
            f"Unsupported PointCloud marker {marker!r}. Supported markers: {supported}."
        ) from e


def pointcloud_marker_for_mpl(marker: str | None, *, default: str) -> str:
    canonical = normalize_pointcloud_marker(marker)
    return _MARKER_TO_MPL[canonical] if canonical is not None else default


def pointcloud_marker_for_plotly(marker: str | None, *, default: str) -> str:
    canonical = normalize_pointcloud_marker(marker)
    return _MARKER_TO_PLOTLY[canonical] if canonical is not None else default


def pointcloud_size_for_mpl(size: float | None, *, default_area: float) -> float:
    """
    Convert a backend-agnostic marker size into matplotlib scatter area.

    Plotly marker size is interpreted approximately as a linear marker extent,
    while matplotlib `scatter(..., s=...)` expects area in points^2. We therefore
    square explicit `PointCloud.size` values for matplotlib, while preserving the
    historical default scatter areas when no explicit size is supplied.
    """
    if size is None:
        return default_area
    return float(size) ** 2


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


def unwrap_periodic_offsets(
    offsets: Sequence[Offset], *, use_lattice_coords: bool = False
) -> np.ndarray:
    """
    Return display coordinates for offsets with an optional periodic unwrapping step.

    For lattice offsets, the stored representatives are canonical wrapped images.
    This helper chooses a nearby branch in boundary coordinates so plotting can
    show a spatially continuous cluster, while leaving the underlying offsets
    unchanged.
    """
    if len(offsets) == 0:
        return np.empty((0, 0), dtype=float)

    first_space = offsets[0].space
    if not all(offset.space == first_space for offset in offsets):
        raise ValueError("All offsets must belong to the same space to unwrap.")

    if not (
        isinstance(first_space, Lattice)
        and isinstance(first_space.boundaries, PeriodicBoundary)
    ):
        if use_lattice_coords:
            return np.stack(
                [
                    np.array(offset.rep.evalf(), dtype=float).reshape(-1)
                    for offset in offsets
                ]
            )
        return np.stack([offset.to_vec(np.ndarray).reshape(-1) for offset in offsets])

    lattice = first_space
    reps = np.stack(
        [
            np.array(offset.rebase(lattice.affine).rep.evalf(), dtype=float).reshape(-1)
            for offset in offsets
        ]
    )
    boundary_basis = np.array(lattice.boundaries.basis.evalf(), dtype=float)
    boundary_basis_inv = np.linalg.inv(boundary_basis)

    coeffs = reps @ boundary_basis_inv.T
    angles = 2.0 * np.pi * coeffs
    mean_angles = np.arctan2(np.sin(angles).mean(axis=0), np.cos(angles).mean(axis=0))
    centers = (mean_angles / (2.0 * np.pi)) % 1.0

    centered = coeffs - centers
    centered -= np.round(centered)
    unwrapped_coeffs = centers + centered
    unwrapped_coeffs -= np.round(unwrapped_coeffs.mean(axis=0))
    unwrapped_reps = unwrapped_coeffs @ boundary_basis.T

    if use_lattice_coords:
        return unwrapped_reps

    lattice_basis = np.array(lattice.basis.evalf(), dtype=float)
    return unwrapped_reps @ lattice_basis.T


def analyze_bandstructure_sampling(
    k_space: MomentumSpace,
) -> tuple[np.ndarray, Optional[ReciprocalLattice], bool, int, Optional[np.ndarray]]:
    """
    Return Cartesian k-samples plus metadata needed to choose path vs surface plots.

    The effective dimensionality is computed from the sampled Cartesian points rather
    than the ambient reciprocal-lattice dimension. This prevents degenerate 2D
    meshes such as shape `(N, 1)` from being rendered as surfaces.
    """
    k_points = list(k_space)
    if len(k_points) == 0:
        return np.array([]), None, False, 0, None

    recip = k_points[0].space
    basis_mat = np.array(recip.basis.evalf()).astype(float)
    k_fracs = [np.array(k.rep).astype(float).flatten() for k in k_points]
    k_cart = np.stack(k_fracs) @ basis_mat.T

    is_surface_compatible_2d_bz = False
    surface_order = None
    if (
        isinstance(recip, ReciprocalLattice)
        and recip.dim == 2
        and all(k.space == recip for k in k_points)
    ):
        canonical_k_space = brillouin_zone(recip)
        if same_rays(k_space, canonical_k_space):
            is_surface_compatible_2d_bz = True
            current_positions = {k: i for i, k in enumerate(k_space.elements())}
            surface_order = np.array(
                [current_positions[k] for k in canonical_k_space.elements()], dtype=int
            )

    reciprocal_lattice = recip if isinstance(recip, ReciprocalLattice) else None
    effective_dim = _effective_dimensionality(k_cart)
    return (
        k_cart,
        reciprocal_lattice,
        is_surface_compatible_2d_bz,
        effective_dim,
        surface_order,
    )


def band_path_positions(k_space: MomentumSpace, k_cart: np.ndarray) -> np.ndarray:
    """
    Build cumulative 1D path coordinates for a sampled momentum path.

    Consecutive momentum points live on a periodic reciprocal torus, so the raw
    Cartesian difference between wrapped samples can contain an artificial jump at
    the Brillouin-zone boundary. We therefore compute step lengths from the
    minimum-image displacement in fractional coordinates before converting back
    to Cartesian space.
    """
    k_points = list(k_space)
    if len(k_points) <= 1 or len(k_cart) == 0:
        return np.array([0.0], dtype=float)

    recip = k_points[0].space
    basis_mat = np.array(recip.basis.evalf(), dtype=float)
    k_fracs = np.stack(
        [np.array(k.rep.evalf(), dtype=float).flatten() for k in k_points], axis=0
    )
    frac_diffs = k_fracs[1:] - k_fracs[:-1]
    wrapped_frac_diffs = frac_diffs - np.round(frac_diffs)
    cart_diffs = wrapped_frac_diffs @ basis_mat.T
    dists = np.linalg.norm(cart_diffs, axis=1)
    return np.concatenate((np.array([0.0], dtype=float), np.cumsum(dists)))


def _effective_dimensionality(points: np.ndarray, tol: float = 1e-12) -> int:
    if points.size == 0 or len(points) <= 1:
        return 0

    centered = points - points[0]
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values.size == 0:
        return 0

    scale = float(singular_values[0])
    if scale <= tol:
        return 0
    return int(np.count_nonzero(singular_values > tol * scale))


def interpolate_path_on_grid(
    bz_path: BzPath,
    grid_k_space: MomentumSpace,
    eigvals: np.ndarray,
) -> np.ndarray:
    """Interpolate eigenvalues from a regular BZ grid onto a path.

    Uses trilinear interpolation with periodic padding so that bands are
    smooth even when path k-points fall between grid points.

    Returns an array of shape ``(len(bz_path.path_order), n_bands)``.
    """
    from scipy.interpolate import RegularGridInterpolator

    grid_k_points = list(grid_k_space)
    grid_recip = grid_k_points[0].space
    dim = grid_recip.dim
    n_bands = eigvals.shape[1]

    grid_fracs = np.array(
        [np.array(k.rep, dtype=float).flatten() for k in grid_k_points]
    )
    grid_fracs = grid_fracs % 1.0

    axes: list[np.ndarray] = []
    for d in range(dim):
        unique_vals = np.sort(np.unique(np.round(grid_fracs[:, d], 10)))
        axes.append(unique_vals)
    shape = tuple(len(ax) for ax in axes)

    # Map each grid k-point to its (i, j, k, ...) position and fill the array.
    evals_grid = np.empty((*shape, n_bands))
    evals_grid[:] = np.nan
    for frac, ev in zip(grid_fracs, eigvals):
        idx = tuple(
            int(np.searchsorted(axes[d], round(frac[d], 10))) for d in range(dim)
        )
        evals_grid[idx] = ev

    # Pad one extra slice at the end of each axis (periodic copy of the
    # first slice) so interpolation wraps smoothly across the BZ boundary.
    for d in range(dim):
        evals_grid = np.concatenate(
            [evals_grid, np.take(evals_grid, [0], axis=d)], axis=d
        )
        axes[d] = np.append(axes[d], 1.0)

    # Resolve path fractional coordinates in the *grid's* reciprocal lattice.
    path_k_points = list(bz_path.k_space)
    path_recip = path_k_points[0].space
    path_fracs_unique = np.array(
        [np.array(k.rep, dtype=float).flatten() for k in path_k_points]
    )
    if path_recip != grid_recip:
        path_basis = np.array(path_recip.basis.evalf(), dtype=float)
        grid_basis = np.array(grid_recip.basis.evalf(), dtype=float)
        path_cart = path_fracs_unique @ path_basis.T
        path_fracs_unique = path_cart @ np.linalg.inv(grid_basis).T

    path_fracs = path_fracs_unique[list(bz_path.path_order)] % 1.0

    result = np.empty((len(bz_path.path_order), n_bands))
    for b in range(n_bands):
        interp = RegularGridInterpolator(
            tuple(axes),
            evals_grid[..., b],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        result[:, b] = interp(path_fracs)

    return result
