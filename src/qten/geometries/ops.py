from __future__ import annotations

import math
from itertools import product

from sympy import ImmutableDenseMatrix

from . import AffineSpace, Lattice, Offset


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
    distance shells, and so on.

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
