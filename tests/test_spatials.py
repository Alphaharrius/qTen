import numpy as np
import pytest
import sympy as sy
import torch
from sympy import ImmutableDenseMatrix
from qten.geometries.spatials import (
    Lattice,
    ReciprocalLattice,
    Offset,
    Momentum,
    AffineSpace,
    AbstractLattice,
)
from qten.geometries.boundary import PeriodicBoundary
from qten.utils.collections_ext import FrozenDict


def test_lattice_creation_and_dual():
    # 2D square lattice
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    assert lattice.dim == 2
    assert lattice.shape == (2, 2)
    assert isinstance(lattice.affine, AffineSpace)
    assert isinstance(lattice, AbstractLattice)
    assert isinstance(lattice.unit_cell, FrozenDict)
    assert len(lattice.unit_cell) == 1
    assert lattice.unit_cell["r"].rep == ImmutableDenseMatrix([0, 0])

    # Check dual
    reciprocal = lattice.dual
    assert isinstance(reciprocal, ReciprocalLattice)
    assert isinstance(reciprocal, AbstractLattice)
    assert reciprocal.dim == 2
    assert not hasattr(reciprocal, "unit_cell")

    # Check double dual gives back original lattice (scaled by 1/4pi^2 in this implementation)
    orig_basis = lattice.basis
    round_trip_basis = reciprocal.dual.basis

    assert round_trip_basis == orig_basis


def test_lattice_with_unit_cell():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    unit_cell_input = {"a": (0, 0), "b": (0.5, 0.5)}
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={
            "a": ImmutableDenseMatrix(unit_cell_input["a"]),
            "b": ImmutableDenseMatrix(unit_cell_input["b"]),
        },
    )

    assert isinstance(lattice.unit_cell, FrozenDict)
    assert len(lattice.unit_cell) == 2
    assert isinstance(lattice.unit_cell["a"], Offset)
    assert lattice.unit_cell["a"].rep == ImmutableDenseMatrix([0, 0])
    assert isinstance(lattice.unit_cell["b"], Offset)
    assert lattice.unit_cell["b"].rep == ImmutableDenseMatrix([0.5, 0.5])

    # ReciprocalLattice should not accept unit_cell
    with pytest.raises(TypeError):
        ReciprocalLattice(basis=basis, lattice=lattice, unit_cell=unit_cell_input)


def test_affine_space_origin_returns_zero_offset():
    affine = AffineSpace(basis=ImmutableDenseMatrix([[2, 1], [0, 3]]))
    origin = affine.origin()

    assert isinstance(origin, Offset)
    assert origin.space == affine
    assert origin.rep == ImmutableDenseMatrix([0, 0])


def test_cartes_lattice():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    # cartes should return offsets for (0,0), (0,1), (1,0), (1,1)
    points = lattice.cartes()
    assert len(points) == 4
    assert isinstance(points[0], Offset)

    # Check content of points
    coords = set()
    for p in points:
        coords.add(tuple(p.rep))

    assert (0, 0) in coords
    assert (0, 1) in coords
    assert (1, 0) in coords
    assert (1, 1) in coords


def test_lattice_basis_vectors_return_primitive_vectors():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[2, 1], [0, 3]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    basis_vectors = lattice.basis_vectors()

    assert len(basis_vectors) == 2
    assert all(isinstance(v, Offset) for v in basis_vectors)
    assert basis_vectors[0].space == lattice
    assert basis_vectors[1].space == lattice
    assert basis_vectors[0].rep == ImmutableDenseMatrix([1, 0])
    assert basis_vectors[1].rep == ImmutableDenseMatrix([0, 1])
    assert basis_vectors[0].to_vec() == ImmutableDenseMatrix([2, 0])
    assert basis_vectors[1].to_vec() == ImmutableDenseMatrix([1, 3])


def test_lattice_basis_vectors_use_affine_space_when_not_lattice_sites():
    triangular = ImmutableDenseMatrix(
        [
            [sy.sqrt(3) / 2, 0],
            [-sy.Rational(1, 2), 1],
        ]
    )
    lattice = Lattice(
        basis=triangular,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={
            "a": triangular
            @ ImmutableDenseMatrix([sy.Rational(1, 3), sy.Rational(2, 3)]),
            "b": triangular
            @ ImmutableDenseMatrix([sy.Rational(2, 3), sy.Rational(1, 3)]),
        },
    )

    basis_vectors = lattice.basis_vectors()

    assert len(basis_vectors) == 2
    assert all(v.space == lattice.affine for v in basis_vectors)
    assert all(v not in lattice for v in basis_vectors)
    assert basis_vectors[0].rep == ImmutableDenseMatrix([1, 0])
    assert basis_vectors[1].rep == ImmutableDenseMatrix([0, 1])


def test_cartes_reciprocal_lattice():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    # shape (2, 2)
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    reciprocal = lattice.dual

    points = reciprocal.cartes()
    assert len(points) == 4

    coords = set()
    for p in points:
        # p.rep should be (n/2, m/2)
        coords.add(tuple(p.rep))

    assert (0, 0) in coords
    assert (sy.Rational(1, 2), 0) in coords
    assert (0, sy.Rational(1, 2)) in coords
    assert (sy.Rational(1, 2), sy.Rational(1, 2)) in coords

    explicit_points = reciprocal.cartes(Momentum)
    assert explicit_points == points


def test_reciprocal_basis_vectors_use_affine_space_when_not_sampled():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1, 0], [0, 1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    reciprocal = lattice.dual

    basis_vectors = reciprocal.basis_vectors()

    assert len(basis_vectors) == 2
    assert all(not isinstance(v, Momentum) for v in basis_vectors)
    assert all(v.space == reciprocal.affine for v in basis_vectors)
    assert basis_vectors[0].rep == ImmutableDenseMatrix([1, 0])
    assert basis_vectors[1].rep == ImmutableDenseMatrix([0, 1])


def test_cartes_tensor():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    # Default unit cell (empty -> one atom at origin)
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    coords = lattice.cartes(torch.Tensor)
    assert coords.shape == (4, 2)

    # Explicit unit cell
    unit_cell = {"a": (0.1, 0.1)}
    lattice_offset = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"a": ImmutableDenseMatrix(unit_cell["a"])},
    )
    coords_offset = lattice_offset.cartes(torch.Tensor)
    assert coords_offset.shape == (4, 2)

    # Check that (0.1, 0.1) is in the coordinates (corresponding to cell 0,0)
    expected = torch.tensor([0.1, 0.1], dtype=torch.float64)
    assert torch.any(torch.all(torch.isclose(coords_offset, expected), dim=1))


def test_lattice_contains_offset_by_unit_cell_mod_lattice_vectors():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1, 0], [0, 1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(3, 3)),
        unit_cell={
            "a": ImmutableDenseMatrix([0, 0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2), sy.Rational(1, 2)]),
        },
    )

    translated_unit_cell_site = Offset(
        rep=ImmutableDenseMatrix([sy.Rational(3, 2), sy.Rational(5, 2)]),
        space=lattice.affine,
    )
    off_unit_cell_site = Offset(
        rep=ImmutableDenseMatrix([sy.Rational(1, 4), sy.Rational(1, 2)]),
        space=lattice.affine,
    )

    assert translated_unit_cell_site in lattice
    assert off_unit_cell_site not in lattice


def test_offset_distance_in_affine_space_is_euclidean():
    affine = AffineSpace(basis=ImmutableDenseMatrix([[1, 0], [0, 1]]))
    a = Offset(rep=ImmutableDenseMatrix([0, 0]), space=affine)
    b = Offset(rep=ImmutableDenseMatrix([3, 4]), space=affine)

    assert a.distance(b) == pytest.approx(5.0)
    assert b.distance(a) == pytest.approx(5.0)


def test_offset_distance_respects_periodic_boundary():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    a = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    b = Offset(rep=ImmutableDenseMatrix([3]), space=lattice)
    c = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice.affine)
    d = Offset(rep=ImmutableDenseMatrix([sy.Rational(7, 2)]), space=lattice)

    assert a.distance(b) == pytest.approx(1.0)
    assert b.distance(a) == pytest.approx(1.0)
    assert c.distance(d) == pytest.approx(1.0)


def test_offset_distance_respects_periodic_boundary_with_skew_basis():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1, 1], [0, 1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    a = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice)
    b = Offset(rep=ImmutableDenseMatrix([3, 3]), space=lattice)

    assert a.distance(b) == pytest.approx(np.sqrt(5.0))
    assert b.distance(a) == pytest.approx(np.sqrt(5.0))


def test_offset_distance_to_affine_point_uses_lattice_boundary():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(8)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    lattice_site = Offset(rep=ImmutableDenseMatrix([7]), space=lattice)
    affine_point = Offset(rep=ImmutableDenseMatrix([1]), space=lattice.affine)

    assert lattice_site.distance(affine_point) == pytest.approx(2.0)


def test_offset_scalar_multiplication_preserves_space_and_wraps_lattice():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    offset = Offset(rep=ImmutableDenseMatrix([3]), space=lattice)

    assert offset * 2 == Offset(rep=ImmutableDenseMatrix([2]), space=lattice)
    assert 2 * offset == Offset(rep=ImmutableDenseMatrix([2]), space=lattice)
    assert sy.Rational(3, 2) * offset == Offset(
        rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice
    )


def test_momentum_scalar_multiplication_returns_momentum():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    momentum = Momentum(
        rep=ImmutableDenseMatrix([sy.Rational(1, 4)]), space=lattice.dual
    )

    doubled = momentum * 2
    scaled = sy.pi * momentum

    assert isinstance(doubled, Momentum)
    assert isinstance(scaled, Momentum)
    assert doubled == Momentum(
        rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice.dual
    )
    assert scaled == Momentum(rep=ImmutableDenseMatrix([sy.pi / 4]), space=lattice.dual)


def test_reciprocal_lattice_contains_only_valid_momentum_points_in_same_space():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice_a = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    lattice_b = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    reciprocal_a = lattice_a.dual
    reciprocal_b = lattice_b.dual

    momentum_same = Momentum(
        rep=ImmutableDenseMatrix([sy.Rational(1, 2), 0]),
        space=reciprocal_a,
    )
    momentum_equal_space = Momentum(
        rep=ImmutableDenseMatrix([sy.Rational(1, 2), 0]),
        space=reciprocal_b,
    )
    momentum_invalid = Momentum(
        rep=ImmutableDenseMatrix([sy.Rational(1, 4), 0]),
        space=reciprocal_a,
    )

    assert momentum_same in reciprocal_a
    assert momentum_equal_space in reciprocal_a
    assert momentum_invalid not in reciprocal_a


def test_offset_rejects_rep_with_wrong_dim():
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))

    with pytest.raises(ValueError, match="must have shape"):
        Offset(rep=ImmutableDenseMatrix([1, 2, 3]), space=space)


def test_offset_rejects_row_vector_rep():
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))

    with pytest.raises(ValueError, match="must have shape"):
        Offset(rep=ImmutableDenseMatrix([[1, 2]]), space=space)


def test_offset_rejects_non_numerical_rep():
    x = sy.symbols("x")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))

    with pytest.raises(ValueError, match="contain only numerical entries"):
        Offset(rep=ImmutableDenseMatrix([x, 1]), space=space)


def test_lattice_at_uses_unit_cell_site_in_origin_cell():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(3, 3)),
        unit_cell={
            "a": ImmutableDenseMatrix([0, 0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2), sy.Rational(1, 2)]),
        },
    )

    offset = lattice.at("b")

    assert isinstance(offset, Offset)
    assert offset.space == lattice
    assert offset.rep == ImmutableDenseMatrix([sy.Rational(1, 2), sy.Rational(1, 2)])


def test_lattice_at_adds_integer_cell_offset():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(5, 5)),
        unit_cell={"b": ImmutableDenseMatrix([sy.Rational(1, 2), 0])},
    )

    offset = lattice.at("b", (2, 3))

    assert offset.rep == ImmutableDenseMatrix([sy.Rational(5, 2), 3])


def test_lattice_at_rejects_unknown_unit_cell_site():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(1),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(3)),
    )

    with pytest.raises(KeyError, match="Unknown unit-cell site"):
        lattice.at("missing")


def test_lattice_at_rejects_wrong_cell_offset_dim():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(3, 3)),
    )

    with pytest.raises(ValueError, match="cell_offset must have length 2"):
        lattice.at("r", (1,))
