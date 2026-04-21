import pytest
import sympy as sy
from sympy import ImmutableDenseMatrix

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.ops import (
    center_of_region,
    get_strip_region_2d,
    interstitial_centers,
    nearest_sites,
    region_centering,
    region_tile,
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


def test_get_strip_region_2d_builds_axis_aligned_strip():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(8, 8)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    region = get_strip_region_2d(
        lattice.at("r", (1, 0)),
        length_step=3,
        width_step=2,
        side="lhs",
    )

    assert tuple(tuple(site.rep) for site in region) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    )


def test_get_strip_region_2d_trim_step_trims_the_tail():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(6, 6)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    region = get_strip_region_2d(
        lattice.at("r", (1, 0)),
        length_step=3,
        width_step=2,
        trim_step=1,
        side="lhs",
    )

    assert tuple(tuple(site.rep) for site in region) == (
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    )


def test_get_strip_region_2d_defaults_to_rhs_side():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(6, 6)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    region = get_strip_region_2d(
        lattice.at("r", (1, 0)),
        length_step=3,
        width_step=2,
    )

    assert tuple(tuple(site.rep) for site in region) == (
        (0, 0),
        (0, 5),
        (1, 0),
        (1, 5),
        (2, 0),
        (2, 5),
    )


def test_get_strip_region_2d_supports_custom_origin():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(8, 8)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    region = get_strip_region_2d(
        lattice.at("r", (1, 0)),
        length_step=3,
        width_step=2,
        side="lhs",
        origin=lattice.at("r", (2, 3)),
    )

    assert tuple(tuple(site.rep) for site in region) == (
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
    )


def test_get_strip_region_2d_supports_affine_origin():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(8, 8)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    affine_origin = Offset(
        rep=ImmutableDenseMatrix([2, 3]),
        space=AffineSpace(ImmutableDenseMatrix.eye(2)),
    )

    region = get_strip_region_2d(
        lattice.at("r", (1, 0)),
        length_step=2,
        width_step=1,
        side="lhs",
        origin=affine_origin,
    )

    assert tuple(tuple(site.rep) for site in region) == (
        (2, 3),
        (3, 3),
    )


def test_get_strip_region_2d_uses_primitive_direction():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(8, 8)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    region = get_strip_region_2d(
        lattice.at("r", (2, 2)),
        length_step=3,
        width_step=2,
        side="lhs",
    )

    assert tuple(tuple(site.rep) for site in region) == (
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
    )


def test_get_strip_region_2d_matches_diagonal_examples():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(16, 16)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    region = get_strip_region_2d(
        lattice.at("r", (1, 1)),
        length_step=3,
        width_step=2,
        side="lhs",
    )
    assert set(tuple(site.rep) for site in region) == {
        (0, 0),
        (1, 1),
        (2, 2),
        (0, 1),
        (1, 2),
    }

    region = get_strip_region_2d(
        lattice.at("r", (1, 1)),
        length_step=3,
        width_step=3,
        side="lhs",
    )
    assert set(tuple(site.rep) for site in region) == {
        (0, 0),
        (1, 1),
        (2, 2),
        (0, 1),
        (1, 2),
        (15, 1),
        (0, 2),
        (1, 3),
    }

    region = get_strip_region_2d(
        lattice.at("r", (1, 1)),
        length_step=3,
        width_step=3,
        trim_step=1,
        side="lhs",
    )
    assert set(tuple(site.rep) for site in region) == {
        (0, 2),
        (1, 1),
        (2, 2),
        (1, 2),
        (1, 3),
    }

    region = get_strip_region_2d(
        lattice.at("r", (1, 1)),
        length_step=3,
        width_step=2,
        side="rhs",
    )
    assert set(tuple(site.rep) for site in region) == {
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 0),
        (2, 1),
    }


def test_get_strip_region_2d_rejects_invalid_inputs():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    three_d_lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0, 0])},
    )

    with pytest.raises(ValueError, match="non-negative"):
        get_strip_region_2d(lattice.at(), length_step=-1, width_step=1)

    with pytest.raises(ValueError, match="trim_step must not exceed"):
        get_strip_region_2d(
            lattice.at("r", (1, 0)), length_step=1, width_step=1, trim_step=2
        )

    with pytest.raises(ValueError, match="non-zero"):
        get_strip_region_2d(lattice.at(), length_step=1, width_step=1)

    with pytest.raises(ValueError, match="side must be 'lhs' or 'rhs'"):
        get_strip_region_2d(
            lattice.at("r", (1, 0)),
            length_step=1,
            width_step=1,
            side="up",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="origin must have dimension"):
        get_strip_region_2d(
            lattice.at("r", (1, 0)),
            length_step=1,
            width_step=1,
            origin=Offset(
                rep=ImmutableDenseMatrix([0, 0, 0]),
                space=AffineSpace(ImmutableDenseMatrix.eye(3)),
            ),
        )

    with pytest.raises(ValueError, match="only 2D lattices"):
        get_strip_region_2d(
            three_d_lattice.at("r", (1, 0, 0)), length_step=1, width_step=1
        )


def test_get_strip_region_2d_with_zero_width_keeps_only_centerline():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(8, 8)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    region = get_strip_region_2d(
        lattice.at("r", (1, 0)),
        length_step=3,
        width_step=1,
        side="lhs",
    )

    assert tuple(tuple(site.rep) for site in region) == (
        (0, 0),
        (1, 0),
        (2, 0),
    )


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


def test_center_of_region_accounts_for_lattice_wrapping():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(1),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )

    center = center_of_region(
        (
            lattice.at("r", (0,)),
            lattice.at("r", (3,)),
        )
    )

    assert type(center) is Offset
    assert center.space == lattice
    assert center.rep == ImmutableDenseMatrix([sy.Rational(7, 2)])


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


def test_center_of_region_accounts_for_momentum_wrapping():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(1),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k = lattice.dual

    center = center_of_region(
        (
            Momentum(rep=ImmutableDenseMatrix([0]), space=k),
            Momentum(rep=ImmutableDenseMatrix([sy.Rational(3, 4)]), space=k),
        )
    )

    assert type(center) is Momentum
    assert center.space == k
    assert center.rep == ImmutableDenseMatrix([sy.Rational(7, 8)])


def test_center_of_region_rejects_empty_region():
    with pytest.raises(ValueError, match="region must be non-empty"):
        center_of_region(())


def test_region_centering_translates_region_to_target_offset_center():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(8, 8)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    region = (
        lattice.at("r", (0, 0)),
        lattice.at("r", (2, 0)),
        lattice.at("r", (0, 2)),
        lattice.at("r", (2, 2)),
    )

    centered = region_centering(region, lattice.at("r", (5, 6)))

    assert tuple(tuple(point.rep) for point in centered) == (
        (4, 5),
        (6, 5),
        (4, 7),
        (6, 7),
    )
    assert center_of_region(centered) == lattice.at("r", (5, 6))


def test_region_centering_handles_periodic_wrapping():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(1),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    region = (
        lattice.at("r", (0,)),
        lattice.at("r", (1,)),
    )

    centered = region_centering(region, lattice.at("r", (0,)))

    assert tuple(tuple(point.rep) for point in centered) == (
        (sy.Rational(7, 2),),
        (sy.Rational(1, 2),),
    )
    assert center_of_region(centered) == lattice.at("r", (0,))


def test_region_centering_rejects_center_with_wrong_space():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(1),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    region = (lattice.at("r", (0,)),)
    wrong_center = Offset(
        rep=ImmutableDenseMatrix([0]),
        space=AffineSpace(ImmutableDenseMatrix.eye(1)),
    )

    with pytest.raises(TypeError, match="same space as the region"):
        region_centering(region, wrong_center)


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


def test_region_tile_translates_region_by_integer_basis_combinations():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(8, 8)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    region = (
        lattice.at("r", (0, 0)),
        lattice.at("r", (0, 1)),
    )
    bases = (
        lattice.at("r", (1, 0)),
        lattice.at("r", (0, 2)),
    )

    tiled = region_tile(region, bases, counts=(2, 2))

    assert tuple(tuple(point.rep) for point in tiled) == (
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    )


def test_region_tile_deduplicates_after_periodic_wrapping():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(1),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(3)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )

    tiled = region_tile(
        region=(lattice.at("r", (0,)),),
        bases=(lattice.at("r", (1,)),),
        counts=(5,),
    )

    assert tuple(tuple(point.rep) for point in tiled) == ((0,), (1,), (2,))


def test_region_tile_rejects_invalid_counts_and_basis_space():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    region = (lattice.at("r", (0, 0)),)
    wrong_space_basis = (
        Offset(
            rep=ImmutableDenseMatrix([1, 0]),
            space=AffineSpace(ImmutableDenseMatrix.eye(2)),
        ),
    )

    with pytest.raises(ValueError, match="same length"):
        region_tile(region, (lattice.at("r", (1, 0)),), counts=(1, 2))

    with pytest.raises(ValueError, match="non-negative"):
        region_tile(region, (lattice.at("r", (1, 0)),), counts=(-1,))

    with pytest.raises(TypeError, match="same space as the region"):
        region_tile(region, wrong_space_basis, counts=(1,))
