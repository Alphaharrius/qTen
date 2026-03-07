import pytest
from sympy import ImmutableDenseMatrix

from pyhilbert.boundary import PeriodicBoundary
from pyhilbert.spatials import Lattice, Offset


def col(*values: int) -> ImmutableDenseMatrix:
    return ImmutableDenseMatrix(list(values))


def as_tuple(vec: ImmutableDenseMatrix) -> tuple[int, ...]:
    return tuple(int(value) for value in vec)


def is_boundary_equivalent(
    basis: ImmutableDenseMatrix,
    left: ImmutableDenseMatrix,
    right: ImmutableDenseMatrix,
) -> bool:
    coeffs = basis.inv() @ (left - right)
    return all(value.is_integer for value in coeffs)

def test_periodic_boundary_rejects_non_square_basis():
    with pytest.raises(ValueError, match="boundary basis must be square"):
        PeriodicBoundary(ImmutableDenseMatrix([[1, 0, 0], [0, 1, 0]]))


def test_periodic_boundary_rejects_rank_deficient_basis():
    with pytest.raises(ValueError, match="boundary basis must be full-rank"):
        PeriodicBoundary(ImmutableDenseMatrix([[1, 2], [2, 4]]))


def test_periodic_boundary_wrap_rejects_wrong_shape():
    boundary = PeriodicBoundary(ImmutableDenseMatrix.diag(4, 3))

    with pytest.raises(ValueError, match="index shape"):
        boundary.wrap(ImmutableDenseMatrix([1, 2, 3]))


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (col(5, -1), col(1, 2)),
        (col(8, 6), col(0, 0)),
        (col(-1, -4), col(3, 2)),
    ],
)
def test_periodic_boundary_wrap_diagonal_basis(index: ImmutableDenseMatrix, expected: ImmutableDenseMatrix):
    boundary = PeriodicBoundary(ImmutableDenseMatrix.diag(4, 3))

    assert boundary.wrap(index) == expected


def test_periodic_boundary_representatives_diagonal_basis():
    boundary = PeriodicBoundary(ImmutableDenseMatrix.diag(2, 3))

    reps = boundary.representatives()

    assert len(reps) == 6
    assert {as_tuple(rep) for rep in reps} == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    }


def test_periodic_boundary_wrap_nondiagonal_basis_returns_representative():
    basis = ImmutableDenseMatrix([[2, 1], [0, 2]])
    boundary = PeriodicBoundary(basis)
    reps = boundary.representatives()
    rep_set = {as_tuple(rep) for rep in reps}

    wrapped = boundary.wrap(col(5, -1))

    assert as_tuple(wrapped) in rep_set
    assert is_boundary_equivalent(basis, col(5, -1), wrapped)


def test_periodic_boundary_representatives_nondiagonal_basis_cover_all_classes():
    basis = ImmutableDenseMatrix([[2, 1], [0, 2]])
    boundary = PeriodicBoundary(basis)
    reps = boundary.representatives()
    rep_set = {as_tuple(rep) for rep in reps}

    assert len(reps) == abs(int(basis.det()))
    assert len(rep_set) == len(reps)

    seen: set[tuple[int, ...]] = set()
    for x in range(-1, 3):
        for y in range(-1, 3):
            wrapped = boundary.wrap(col(x, y))
            wrapped_key = as_tuple(wrapped)
            assert wrapped_key in rep_set
            assert is_boundary_equivalent(basis, col(x, y), wrapped)
            seen.add(wrapped_key)

    assert seen == rep_set


def test_offset_constructor_wraps_using_boundary():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 3)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    offset = Offset(rep=col(5, -1), space=lattice)

    assert offset.rep == col(1, 2)


def test_offset_addition_wraps_on_diagonal_boundary():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 3)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    left = Offset(rep=col(3, 2), space=lattice)
    right = Offset(rep=col(2, 2), space=lattice)

    assert (left + right).rep == col(1, 1)


def test_offset_addition_wraps_to_representative_on_nondiagonal_boundary():
    basis = ImmutableDenseMatrix([[2, 1], [0, 2]])
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(basis),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    rep_set = {as_tuple(rep) for rep in lattice.boundaries.representatives()}

    left = Offset(rep=col(1, 0), space=lattice)
    right = Offset(rep=col(1, 1), space=lattice)
    summed = left + right

    assert as_tuple(summed.rep) in rep_set
    assert is_boundary_equivalent(basis, left.rep + right.rep, summed.rep)
