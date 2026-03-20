import pytest
from sympy import ImmutableDenseMatrix

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.ops import nearest_sites
from qten.geometries.spatials import AffineSpace, Lattice, Offset


def test_nearest_sites_with_lattice_center_is_inclusive_at_cutoff():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(3, 3)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    region = nearest_sites(lattice, lattice.at("r", (0, 0)), n_nearest=2)

    assert tuple(tuple(site.rep) for site in region) == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
    )


def test_nearest_sites_with_affine_center_includes_all_tied_sites():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    center = Offset(
        rep=ImmutableDenseMatrix([0.5, 0.5]),
        space=AffineSpace(ImmutableDenseMatrix.eye(2)),
    )

    region = nearest_sites(lattice, center, n_nearest=1)

    assert tuple(tuple(site.rep) for site in region) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )


def test_nearest_sites_uses_distinct_distance_shells():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(3, 3)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    region = nearest_sites(lattice, lattice.at("r", (0, 0)), n_nearest=3)

    assert tuple(tuple(site.rep) for site in region) == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    )


def test_nearest_sites_with_zero_count_returns_empty_region():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(1),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )

    assert nearest_sites(lattice, lattice.at(), n_nearest=0) == ()


def test_nearest_sites_rejects_negative_counts():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(1),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )

    with pytest.raises(ValueError, match="n_nearest must be non-negative"):
        nearest_sites(lattice, lattice.at(), n_nearest=-1)
