"""
Geometry helper operations for lattice regions and momentum paths.

This module contains functional helpers built on top of
[`AffineSpace`][qten.geometries.spatials.AffineSpace],
[`Lattice`][qten.geometries.spatials.Lattice],
[`Offset`][qten.geometries.spatials.Offset], and
[`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]. The helpers
construct common real-space regions, nearest-neighbor site selections, strip
geometries, reciprocal-space paths, and related geometry data without adding
stateful wrapper classes.

Repository usage
----------------
Use this module when an operation derives a new region or path from existing
geometry objects rather than defining a new geometry type. Class definitions
and the core spatial algebra live in [`qten.geometries.spatials`][qten.geometries.spatials].
"""

from __future__ import annotations

import math
from itertools import combinations, product
from typing import Literal, cast

import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix

from .spatials import AffineSpace, Lattice, Offset, OffsetType, ReciprocalLattice


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


def _primitive_integer_direction(
    direction: Offset[Lattice],
) -> tuple[Lattice, int, int]:
    lattice = direction.space
    if lattice.dim != 2:
        raise ValueError(
            f"get_strip_region_2d currently supports only 2D lattices, got dim={lattice.dim}."
        )

    if direction.rep != ImmutableDenseMatrix([sy.nsimplify(x) for x in direction.rep]):
        raise ValueError("direction must have exact symbolic coordinates.")

    coords = [sy.nsimplify(x) for x in direction.rep]
    if any(coord.is_integer is not True for coord in coords):
        raise ValueError(
            "direction must be an integer lattice translation with no unit-cell fractional offset."
        )

    dx, dy = (int(coords[0]), int(coords[1]))
    if dx == 0 and dy == 0:
        raise ValueError("direction must be non-zero.")

    g = math.gcd(abs(dx), abs(dy))
    return lattice, dx // g, dy // g


def _strip_direction_data(
    direction: Offset[Lattice],
) -> tuple[Lattice, float, float, int, int]:
    lattice = direction.space
    if lattice.dim != 2:
        raise ValueError(
            f"get_strip_region_2d currently supports only 2D lattices, got dim={lattice.dim}."
        )

    coords = [sy.nsimplify(x) for x in direction.rep]
    if any(not coord.is_real for coord in coords):
        raise ValueError("direction must have real coordinates.")
    if coords[0] == 0 and coords[1] == 0:
        raise ValueError("direction must be non-zero.")

    if all(coord.is_integer is True for coord in coords):
        _, px, py = _primitive_integer_direction(direction)
        return lattice, float(px), float(py), px, py

    if any(not coord.is_rational for coord in coords):
        raise ValueError("direction must have exact integer or rational coordinates.")

    den_lcm = 1
    for coord in coords:
        den_lcm = math.lcm(den_lcm, int(cast(sy.Rational, coord).q))

    integer_coords = [int(cast(sy.Rational, coord) * den_lcm) for coord in coords]
    g = math.gcd(abs(integer_coords[0]), abs(integer_coords[1]))
    px = integer_coords[0] // g
    py = integer_coords[1] // g
    return lattice, float(coords[0]), float(coords[1]), px, py


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
    lattice : Lattice
        Finite lattice whose sites define the candidate region.
    center : Offset[AffineSpace] | Offset[Lattice]
        Center used to rank lattice sites by distance. The center may be an
        arbitrary offset in the lattice affine space and does not need to lie
        on a lattice site.
    n_nearest : int
        Number of distinct distance shells to include. `0` returns an empty
        region. If `n_nearest` exceeds the number of distinct distance shells
        in the finite lattice, all sites are returned.

    Returns
    -------
    tuple[Offset[Lattice], ...]
        Tuple of lattice sites whose distances from `center` lie in the first
        n_nearest distinct distance shells, ordered by increasing distance
        and then by the lattice-site ordering.

    Raises
    ------
    ValueError
        If `n_nearest` is negative or if `center.dim` does not match
        lattice.dim.
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


def get_strip_region_2d(
    direction: Offset[Lattice],
    *,
    length_step: int,
    width_step: int,
    trim_step: int = 0,
    side: Literal["lhs", "rhs"] = "rhs",
    origin: Offset[AffineSpace] | Offset[Lattice] | None = None,
) -> tuple[Offset[Lattice], ...]:
    r"""
    Return a 2D rectangular strip region in primitive-strip lattice coordinates.

    This helper is defined only for 2D lattices.

    Let `r0` be the supplied [`origin`][qten.geometries.spatials.AffineSpace.origin] (or the lattice origin when omitted).
    Let \((d_x, d_y)\) be the supplied direction coordinates. Let
    \(p = (p_x, p_y)\) be the associated primitive integer direction, and let
    \(n = (-p_y, p_x)\) be the primitive integer normal. `side="lhs"` grows
    toward positive \(n\) and `side="rhs"` grows toward negative \(n\).

    A lattice site belongs to the strip when some periodic image of that site
    satisfies both of the following:

    - Longitudinal bound:
      \[
      \mathrm{trim\_step}(d_x^2 + d_y^2)
      \le d_x(r_x-r_{0x}) + d_y(r_y-r_{0y})
      \le (\mathrm{length\_step}-1)(d_x^2+d_y^2).
      \]
    - Transverse bound:
      \[
      0
      \le s[-p_y(r_x-r_{0x}) + p_x(r_y-r_{0y})]
      \le \mathrm{width\_step}-1.
      \]

    where \(s = 1\) for `"lhs"` and \(s = -1\) for `"rhs"`.

    For integer directions, \((d_x, d_y) = (p_x, p_y)\). For rational directions,
    longitudinal shell spacing is computed from the supplied direction
    `(dx, dy)`, while transverse shelling is computed from the primitive
    integer direction `p`.

    `width_step` counts the transverse shell thickness including the main axis
    row. `trim_step` is a tail trimmer only: it advances the strip start along
    the longitudinal axis without affecting the transverse width.

    Parameters
    ----------
    direction : Offset[Lattice]
        Non-zero lattice translation on a 2D lattice whose primitive direction
        defines the strip axis.
    length_step : int
        Number of strip shells from the origin along the primitive direction.
    width_step : int
        Number of transverse shell rows including the main axis row.
    trim_step : int
        Number of longitudinal shells trimmed from the tail near the origin.
    side : Literal["lhs", "rhs"]
        Side on which transverse width shells are accumulated relative to the
        strip direction. `"lhs"` uses the positive lattice normal and `"rhs"`
        uses the negative lattice normal.
    origin : Offset[AffineSpace] | Offset[Lattice] | None
        Anchor point for the strip coordinates. If omitted, the zero offset in
        the lattice space is used. When provided, it is rebased into the
        lattice before evaluating strip membership.

    Returns
    -------
    tuple[Offset[Lattice], ...]
        Deduplicated lattice sites in the strip, ordered by the lattice-site
        ordering.

    Raises
    ------
    ValueError
        If the direction is invalid, the lattice is not 2D, or any step count
        is out of range.
    """
    if length_step < 0:
        raise ValueError(f"length_step must be non-negative, got {length_step}.")
    if width_step < 0:
        raise ValueError(f"width_step must be non-negative, got {width_step}.")
    if trim_step < 0:
        raise ValueError(f"trim_step must be non-negative, got {trim_step}.")
    if trim_step > length_step:
        raise ValueError(
            f"trim_step must not exceed length_step, got {trim_step} and {length_step}."
        )
    if side not in ("lhs", "rhs"):
        raise ValueError(f"side must be 'lhs' or 'rhs', got {side!r}.")
    if length_step == 0 or width_step == 0:
        return ()

    lattice, dx, dy, px, py = _strip_direction_data(direction)
    if origin is None:
        origin = lattice.origin()
    if origin.dim != lattice.dim:
        raise ValueError(
            f"origin must have dimension {lattice.dim} to match the lattice, got {origin.dim}."
        )
    origin_rep = np.array(origin.rebase(lattice).rep, dtype=float).reshape(-1)
    all_sites = lattice.cartes()
    if len(all_sites) == 0:
        return ()

    normal_x = -py
    normal_y = px
    normal_sign = 1 if side == "lhs" else -1
    longitudinal_min = trim_step * (dx * dx + dy * dy)
    longitudinal_max = (length_step - 1) * (dx * dx + dy * dy)
    transverse_min = 0
    transverse_max = width_step - 1

    boundary_basis = np.array(lattice.boundaries.basis.tolist(), dtype=int)
    image_shifts = [
        boundary_basis @ np.array(shift, dtype=int)
        for shift in product((-1, 0, 1), repeat=lattice.dim)
    ]

    region: list[Offset[Lattice]] = []
    for site in all_sites:
        base_rep = np.array(site.rep, dtype=float).reshape(-1)
        include = False
        for shift in image_shifts:
            image_rep = base_rep + shift
            relative_rep = image_rep - origin_rep
            along = dx * relative_rep[0] + dy * relative_rep[1]
            across = normal_sign * (
                normal_x * relative_rep[0] + normal_y * relative_rep[1]
            )
            if (
                along >= longitudinal_min - 1e-9
                and along <= longitudinal_max + 1e-9
                and across >= transverse_min - 1e-9
                and across <= transverse_max + 1e-9
            ):
                include = True
                break
        if include:
            region.append(site)

    return tuple(sorted(region))


def center_of_region(region: tuple[OffsetType, ...]) -> OffsetType:
    """
    Return the arithmetic center of a non-empty region of offsets or momenta.

    Parameters
    ----------
    region : tuple[Offset, ...] | tuple[Momentum, ...]
        Non-empty tuple of spatial points. All entries must share the same
        concrete type and affine space.

    Returns
    -------
    Offset | Momentum
        Arithmetic mean of the region coordinates, returned as the same type as
        the input entries.

    Raises
    ------
    ValueError
        If `region` is empty.
    TypeError
        If region entries do not all share the same concrete type and space.
    """
    if len(region) == 0:
        raise ValueError("region must be non-empty.")

    first = region[0]
    point_type = type(first)

    for point in region[1:]:
        if type(point) is not point_type:
            raise TypeError("region entries must all have the same concrete type.")
        if point.space != first.space:
            raise TypeError("region entries must all belong to the same space.")

    if isinstance(first.space, Lattice):
        boundary_basis = np.array(first.space.boundaries.basis.evalf(), dtype=float)
        wrapped = np.stack(
            [
                np.array(
                    (first.space.boundaries.basis.inv() @ point.rep).evalf(),
                    dtype=float,
                ).reshape(-1)
                for point in region
            ]
        )
        reference = wrapped[0]
        unwrapped = wrapped.copy()
        for i in range(1, len(region)):
            delta = wrapped[i] - reference
            unwrapped[i] = wrapped[i] - np.round(delta)

        mean_boundary = unwrapped.mean(axis=0) % 1.0
        mean_rep_np = boundary_basis @ mean_boundary
        mean_rep = ImmutableDenseMatrix([sy.nsimplify(x) for x in mean_rep_np])
        return point_type(rep=mean_rep, space=first.space)

    if isinstance(first.space, ReciprocalLattice):
        wrapped = np.stack(
            [np.array(point.rep.evalf(), dtype=float).reshape(-1) for point in region]
        )
        reference = wrapped[0]
        unwrapped = wrapped.copy()
        for i in range(1, len(region)):
            delta = wrapped[i] - reference
            unwrapped[i] = wrapped[i] - np.round(delta)

        mean_rep_np = unwrapped.mean(axis=0) % 1.0
        mean_rep = ImmutableDenseMatrix([sy.nsimplify(x) for x in mean_rep_np])
        return point_type(rep=mean_rep, space=first.space)

    total = first.rep
    for point in region[1:]:
        total += point.rep

    return point_type(rep=ImmutableDenseMatrix(total / len(region)), space=first.space)


def region_centering(
    region: tuple[OffsetType, ...], center: OffsetType
) -> tuple[OffsetType, ...]:
    """
    Translate a region so that its arithmetic center lands at `center`.

    Parameters
    ----------
    region : tuple[Offset, ...] | tuple[Momentum, ...]
        Region to translate. All entries must share the same concrete type and
        affine space.
    center : Offset | Momentum
        Target center for the translated region. It must have the same
        concrete type and affine space as the region entries.

    Returns
    -------
    tuple[Offset, ...] | tuple[Momentum, ...]
        Region translated by `center - center_of_region(region)`. Empty input
        returns an empty tuple.

    Raises
    ------
    TypeError
        If region entries do not all share the same concrete type and space,
        or if `center` does not match them.
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

    if type(center) is not point_type:
        raise TypeError(
            "center must have the same concrete type as the region entries."
        )
    if center.space != first.space:
        raise TypeError("center must belong to the same space as the region entries.")

    translation = cast(OffsetType, center - center_of_region(region))
    return tuple(cast(OffsetType, point + translation) for point in region)


def region_tile(
    region: tuple[OffsetType, ...],
    bases: tuple[OffsetType, ...],
    counts: tuple[int, ...],
) -> tuple[OffsetType, ...]:
    """
    Tile a region by integer combinations of the supplied translation bases.

    The returned region contains translations of every point in `region` by
    offsets

    .. math::

        \\sum_i n_i b_i, \\qquad 0 \\le n_i < \\mathrm{counts}[i],

    where `b_i` are the entries of `bases`.

    Parameters
    ----------
    region : tuple[Offset, ...] | tuple[Momentum, ...]
        Region to translate. All entries must share the same concrete type and
        affine space.
    bases : tuple[Offset, ...] | tuple[Momentum, ...]
        Translation basis vectors. All entries must share the same concrete
        type and affine space as the region entries.
    counts : tuple[int, ...]
        Number of repetitions along each translation basis. Each entry must be
        non-negative.

    Returns
    -------
    tuple[Offset, ...] | tuple[Momentum, ...]
        Deduplicated tiled region, ordered by the point ordering.

    Raises
    ------
    TypeError
        If region or basis entries do not all share the same concrete type and
        space.
    ValueError
        If `counts` has the wrong length or contains negative entries.
    """
    if len(region) == 0:
        return ()

    if len(bases) != len(counts):
        raise ValueError(
            f"bases and counts must have the same length, got {len(bases)} and {len(counts)}."
        )
    if any(count < 0 for count in counts):
        raise ValueError(f"counts must be non-negative, got {counts}.")
    if any(count == 0 for count in counts):
        return ()

    first = region[0]
    point_type = type(first)

    for point in region[1:]:
        if type(point) is not point_type:
            raise TypeError("region entries must all have the same concrete type.")
        if point.space != first.space:
            raise TypeError("region entries must all belong to the same space.")

    for basis in bases:
        if type(basis) is not point_type:
            raise TypeError(
                "basis entries must all have the same concrete type as the region entries."
            )
        if basis.space != first.space:
            raise TypeError(
                "basis entries must all belong to the same space as the region entries."
            )

    tiled: dict[OffsetType, None] = {}
    for coefficients in product(*(range(count) for count in counts)):
        translation_rep = sum(
            (
                coefficient * basis.rep
                for coefficient, basis in zip(coefficients, bases)
            ),
            ImmutableDenseMatrix.zeros(first.dim, 1),
        )
        translation = point_type(
            rep=ImmutableDenseMatrix(translation_rep), space=first.space
        )
        for point in region:
            tiled[cast(OffsetType, point + translation)] = None

    return tuple(sorted(tiled))


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
    region : tuple[Offset, ...] | tuple[Momentum, ...]
        Spatial points defining the candidate corner set. All entries must
        share the same concrete type and affine space.

    Returns
    -------
    tuple[Offset, ...] | tuple[Momentum, ...]
        Interstitial centers, returned as the same concrete type as the inputs
        and ordered lexicographically by point coordinates.

    Raises
    ------
    TypeError
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
