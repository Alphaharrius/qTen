from __future__ import annotations

import math
from itertools import combinations, product

import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix

from .spatials import AffineSpace, Lattice, Offset, OffsetType


def _cutoff_from_sites(
    sites_with_distances: list[tuple[float, Offset[Lattice]]], n_nearest: int
) -> float | None:
    shell_count = 0
    previous_distance: float | None = None
    for distance, _ in sites_with_distances:
        if previous_distance is None or not math.isclose(
            distance, previous_distance, rel_tol=1e-9, abs_tol=1e-9
        ):
            shell_count += 1
            previous_distance = distance
            if shell_count == n_nearest:
                return distance
    return None


def nearest_sites(
    lattice: Lattice, center: Offset[AffineSpace] | Offset[Lattice], n_nearest: int
) -> tuple[Offset[Lattice], ...]:
    """
    Return lattice sites through the `n_nearest`-th distinct distance shell.

    Sites are ordered by increasing distance from `center`, with lattice-site
    ordering used to break ties deterministically. `n_nearest=1` returns the
    nearest-distance shell, `n_nearest=2` returns the first two distinct
    distance shells, and so on. The center itself is included only when it
    coincides with a lattice site.

    Parameters
    ----------
    `lattice` : `Lattice`
        Finite lattice whose sites define the candidate region.
    `center` : `Offset[AffineSpace] | Offset[Lattice]`
        Center used to rank lattice sites by distance. The center may be an
        arbitrary offset in the lattice affine space and does not need to lie
        on a lattice site.
    `n_nearest` : `int`
        Number of distinct distance shells to include. `0` returns an empty
        region. If `n_nearest` exceeds the number of distinct distance shells
        in the finite lattice, all sites are returned.

    Returns
    -------
    `tuple[Offset[Lattice], ...]`
        Tuple of lattice sites whose distances from `center` lie in the first
        `n_nearest` distinct distance shells, ordered by increasing distance
        and then by the lattice-site ordering.

    Raises
    ------
    `ValueError`
        If `n_nearest` is negative or if `center.dim` does not match
        `lattice.dim`.
    """
    if n_nearest < 0:
        raise ValueError(f"n_nearest must be non-negative, got {n_nearest}.")
    if center.dim != lattice.dim:
        raise ValueError(
            f"center must have dimension {lattice.dim} to match the lattice, got {center.dim}."
        )
    if n_nearest == 0:
        return ()

    unit_cell_sites = tuple(lattice.unit_cell.values())
    total_sites = math.prod(lattice.shape) * len(unit_cell_sites)
    center_rep = center.rebase(lattice).rep
    origin_cell = tuple(
        int(math.floor(float(center_rep[i, 0]))) for i in range(lattice.dim)
    )

    discovered: dict[Offset[Lattice], float] = {}
    included_count = -1
    stable_after_cutoff = False
    max_local_radius = max(shape // 2 for shape in lattice.shape)

    for radius in range(max_local_radius + 1):
        ranges = [range(cell - radius, cell + radius + 1) for cell in origin_cell]
        for cell_offset in product(*ranges):
            if radius and all(
                abs(cell_offset[i] - origin_cell[i]) < radius
                for i in range(lattice.dim)
            ):
                continue
            cell_rep = ImmutableDenseMatrix(cell_offset)
            for site in unit_cell_sites:
                candidate = Offset(
                    rep=ImmutableDenseMatrix(cell_rep + site.rep), space=lattice
                )
                if candidate in discovered:
                    continue
                discovered[candidate] = center.distance(candidate)

        if len(discovered) == total_sites:
            break

        if len(discovered) < n_nearest:
            continue

        local_sites = sorted(
            ((distance, site) for site, distance in discovered.items()),
            key=lambda item: (item[0], item[1]),
        )
        cutoff_distance = _cutoff_from_sites(local_sites, n_nearest)
        if cutoff_distance is None:
            continue

        new_included_count = sum(
            1
            for distance, _ in local_sites
            if distance < cutoff_distance
            or math.isclose(distance, cutoff_distance, rel_tol=1e-9, abs_tol=1e-9)
        )
        if new_included_count == included_count:
            stable_after_cutoff = True
            break
        included_count = new_included_count

    if stable_after_cutoff or len(discovered) == total_sites:
        sites_with_distances = sorted(
            ((distance, site) for site, distance in discovered.items()),
            key=lambda item: (item[0], item[1]),
        )
    else:
        sites_with_distances = sorted(
            (
                (center.distance(candidate), candidate)
                for cell in lattice.boundaries.representatives()
                for site in unit_cell_sites
                for candidate in (
                    Offset(rep=ImmutableDenseMatrix(cell + site.rep), space=lattice),
                )
            ),
            key=lambda item: (item[0], item[1]),
        )

    cutoff_distance = _cutoff_from_sites(sites_with_distances, n_nearest)

    if cutoff_distance is None:
        return tuple(site for _, site in sites_with_distances)

    return tuple(
        site
        for distance, site in sites_with_distances
        if distance < cutoff_distance
        or math.isclose(distance, cutoff_distance, rel_tol=1e-9, abs_tol=1e-9)
    )


def center_of_region(region: tuple[OffsetType, ...]) -> OffsetType:
    """
    Return the arithmetic center of a non-empty region of offsets or momenta.

    Parameters
    ----------
    `region` : `tuple[Offset, ...] | tuple[Momentum, ...]`
        Non-empty tuple of spatial points. All entries must share the same
        concrete type and affine space.

    Returns
    -------
    `Offset | Momentum`
        Arithmetic mean of the region coordinates, returned as the same type as
        the input entries.

    Raises
    ------
    `ValueError`
        If `region` is empty.
    `TypeError`
        If region entries do not all share the same concrete type and space.
    """
    if len(region) == 0:
        raise ValueError("region must be non-empty.")

    first = region[0]
    point_type = type(first)
    total = first.rep

    for point in region[1:]:
        if type(point) is not point_type:
            raise TypeError("region entries must all have the same concrete type.")
        if point.space != first.space:
            raise TypeError("region entries must all belong to the same space.")
        total += point.rep

    return point_type(rep=ImmutableDenseMatrix(total / len(region)), space=first.space)


def _circumcenter(points: np.ndarray) -> np.ndarray | None:
    base = points[0]
    linear = 2.0 * (points[1:] - base)
    rhs = np.sum(points[1:] ** 2, axis=1) - np.sum(base**2)

    if np.linalg.matrix_rank(linear) < points.shape[1]:
        return None

    return np.linalg.solve(linear, rhs)


def interstitial_centers(region: tuple[OffsetType, ...]) -> tuple[OffsetType, ...]:
    """
    Return centers of locally maximal empty spheres supported by `region`.

    Candidate gap points are built as circumcenters of local simplices formed
    from nearby sites. A candidate is retained when its defining sites are
    equidistant from the center and no input point lies strictly closer. This
    recovers square-lattice plaquette centers and also produces non-trivial
    void centers for lattices such as diamond.

    Parameters
    ----------
    `region` : `tuple[Offset, ...] | tuple[Momentum, ...]`
        Spatial points defining the candidate corner set. All entries must
        share the same concrete type and affine space.

    Returns
    -------
    `tuple[Offset, ...] | tuple[Momentum, ...]`
        Interstitial centers, returned as the same concrete type as the inputs
        and ordered lexicographically by point coordinates.

    Raises
    ------
    `TypeError`
        If region entries do not all share the same concrete type and space.
    """
    if len(region) == 0:
        return ()

    first = region[0]
    point_type = type(first)

    for point in region[1:]:
        if type(point) is not point_type:
            raise TypeError("region entries must all have the same concrete type.")
        if point.space != first.space:
            raise TypeError("region entries must all belong to the same space.")

    if len(region) < first.dim + 1:
        return ()

    coords = np.stack([point.to_vec(np.ndarray) for point in region])
    pairwise_distances = np.linalg.norm(coords[:, np.newaxis, :] - coords, axis=2)
    np.fill_diagonal(pairwise_distances, np.inf)

    neighborhood_size = min(len(region), max(8, 4 * first.dim + 2))
    min_support = max(first.dim + 1, 4)
    tolerance = 1e-7
    centers_by_rep: dict[tuple, OffsetType] = {}

    for seed_idx in range(len(region)):
        nearest = np.argsort(pairwise_distances[seed_idx])[: neighborhood_size - 1]
        for neighbor_indices in combinations(nearest, first.dim):
            simplex_indices = (seed_idx, *neighbor_indices)
            center_cart = _circumcenter(coords[list(simplex_indices)])
            if center_cart is None:
                continue

            radius = np.linalg.norm(coords[seed_idx] - center_cart)
            distances = np.linalg.norm(coords - center_cart, axis=1)
            if np.any(distances < radius - tolerance):
                continue

            if (
                np.count_nonzero(
                    np.isclose(distances, radius, atol=tolerance, rtol=tolerance)
                )
                < min_support
            ):
                continue

            center_rep = first.space.basis.inv() @ ImmutableDenseMatrix(center_cart)
            center_rep = ImmutableDenseMatrix(
                [sy.nsimplify(coord) for coord in center_rep]
            )
            center = point_type(rep=center_rep, space=first.space)
            centers_by_rep[tuple(center.rep)] = center

    return tuple(sorted(centers_by_rep.values()))
