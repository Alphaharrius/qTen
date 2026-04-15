import pytest
import sympy as sy
from sympy import ImmutableDenseMatrix

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.ops import (
    center_of_region,
    interstitial_centers,
    nearest_sites,
)
from qten.geometries.spatials import AffineSpace, Lattice, Momentum, Offset


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


def test_nearest_sites_excludes_off_lattice_center():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    center = Offset(
        rep=ImmutableDenseMatrix([sy.Rational(1, 2), sy.Rational(1, 2)]),
        space=AffineSpace(ImmutableDenseMatrix.eye(2)),
    )

    region = nearest_sites(lattice, center, n_nearest=1)

    assert center not in lattice
    assert center.rebase(lattice) not in region
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


def test_center_of_region_returns_offset_mean():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    center = center_of_region(
        (
            lattice.at("r", (0, 0)),
            lattice.at("r", (2, 0)),
            lattice.at("r", (0, 2)),
            lattice.at("r", (2, 2)),
        )
    )

    assert type(center) is Offset
    assert center.space == lattice
    assert center.rep == ImmutableDenseMatrix([1, 1])


def test_center_of_region_returns_momentum_mean():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(1),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k = lattice.dual

    center = center_of_region(
        (
            Momentum(rep=ImmutableDenseMatrix([0]), space=k),
            Momentum(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=k),
        )
    )

    assert type(center) is Momentum
    assert center.space == k
    assert center.rep == ImmutableDenseMatrix([sy.Rational(1, 4)])


def test_center_of_region_rejects_empty_region():
    with pytest.raises(ValueError, match="region must be non-empty"):
        center_of_region(())


def test_interstitial_centers_returns_shifted_square_lattice_centers():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(5, 5)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    region = tuple(lattice.at("r", (i, j)) for i in range(3) for j in range(3))

    centers = interstitial_centers(region)

    assert tuple(tuple(point.rep) for point in centers) == (
        (sy.Rational(1, 2), sy.Rational(1, 2)),
        (sy.Rational(1, 2), sy.Rational(3, 2)),
        (sy.Rational(3, 2), sy.Rational(1, 2)),
        (sy.Rational(3, 2), sy.Rational(3, 2)),
    )


def test_interstitial_centers_skips_cells_with_missing_corners():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(5, 5)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    region = (
        lattice.at("r", (0, 0)),
        lattice.at("r", (0, 1)),
        lattice.at("r", (1, 0)),
        lattice.at("r", (1, 1)),
        lattice.at("r", (1, 2)),
        lattice.at("r", (2, 1)),
    )

    centers = interstitial_centers(region)

    assert tuple(tuple(point.rep) for point in centers) == (
        (sy.Rational(1, 2), sy.Rational(1, 2)),
    )


def test_interstitial_centers_finds_nontrivial_voids_for_diamond_lattice():
    half = sy.Rational(1, 2)
    quarter = sy.Rational(1, 4)
    diamond = Lattice(
        basis=ImmutableDenseMatrix(
            [
                [0, half, half],
                [half, 0, half],
                [half, half, 0],
            ]
        ),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(3, 3, 3)),
        unit_cell={
            "A": ImmutableDenseMatrix([0, 0, 0]),
            "B": ImmutableDenseMatrix([quarter, quarter, quarter]),
        },
    )

    centers = interstitial_centers(diamond.cartes())

    assert centers
    assert (
        Offset(
            rep=ImmutableDenseMatrix([sy.Rational(1, 2)] * 3),
            space=diamond,
        )
        in centers
    )
