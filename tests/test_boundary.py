import pytest
from sympy import ImmutableDenseMatrix

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice, Offset, AffineSpace
from qten.affine_transform import AffineTransform, pointgroup
import sympy as sy
from qten.geometries.basis_transform import BasisTransform


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
def test_periodic_boundary_wrap_diagonal_basis(
    index: ImmutableDenseMatrix, expected: ImmutableDenseMatrix
):
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


def test_offset_with_affine_space():
    # Test that Offset can accept a pure AffineSpace and does not require a Lattice
    basis = sy.ImmutableDenseMatrix.eye(3)
    space = AffineSpace(basis=basis)
    rep = sy.ImmutableDenseMatrix([1, 2, 3])

    offset = Offset(rep=rep, space=space)
    assert offset.dim == 3
    assert offset.space == space
    assert offset.rep == rep


def test_pointgroup_with_affine_space():
    # Test that pointgroup uses AffineSpace instead of Lattice
    c3 = pointgroup("c3-xy:xy-o1")
    assert isinstance(c3.offset.space, AffineSpace)
    # Ensure it is not a Lattice (Lattice inherits from AffineSpace, so check type)
    assert type(c3.offset.space) is AffineSpace


def test_affine_group_element_rebase():
    # Test AffineTransform rebasing from AffineSpace to Lattice
    dim = 2
    affine_basis = sy.ImmutableDenseMatrix.eye(dim)
    affine_space = AffineSpace(basis=affine_basis)

    # Construct a 180-degree rotation operation in pure AffineSpace
    irrep = sy.ImmutableDenseMatrix([[-1, 0], [0, -1]])
    axes = (sy.Symbol("x"), sy.Symbol("y"))
    zero_offset = Offset(rep=sy.ImmutableDenseMatrix([0, 0]), space=affine_space)

    op = AffineTransform(
        irrep=irrep, axes=axes, offset=zero_offset, basis_function_order=1
    )

    # Construct a Lattice with boundaries
    lat_basis = sy.ImmutableDenseMatrix([[2, 0], [0, 2]])
    lattice = Lattice(
        basis=lat_basis,
        boundaries=PeriodicBoundary(lat_basis),
        unit_cell={"A": sy.ImmutableDenseMatrix([0, 0])},
    )

    # Rebase to Lattice
    op_lat = op.rebase(lattice)
    assert op_lat.offset.space == lattice
    # Check that the basis transformation is performed correctly
    # (2x2 identity rebase to 2x2 diagonal matrix should produce zero vector)
    assert op_lat.offset.rep == sy.ImmutableDenseMatrix([0, 0])


def test_potential_errors():
    # 1. Test dimension mismatch (Offset shape mismatch)
    basis = sy.ImmutableDenseMatrix.eye(3)
    space = AffineSpace(basis=basis)
    rep_2d = sy.ImmutableDenseMatrix([1, 2])

    with pytest.raises(ValueError, match="Invalid Shape"):
        Offset(rep=rep_2d, space=space)

    # 2. Test error during rebase when dimensions or spaces do not match
    lat_basis = sy.ImmutableDenseMatrix.eye(2)
    lattice_2d = Lattice(
        basis=lat_basis,
        boundaries=PeriodicBoundary(lat_basis),
        unit_cell={"A": sy.ImmutableDenseMatrix([0, 0])},
    )

    # Rebasing a 3D offset to a 2D lattice should raise an error
    offset_3d = Offset(rep=sy.ImmutableDenseMatrix([1, 1, 1]), space=space)
    with pytest.raises(
        Exception
    ):  # Sympy matrix multiplication should raise ShapeError
        offset_3d.rebase(lattice_2d)


def test_periodic_boundary_physical_invariance():
    """
    Test that wrapping a point always produces a physical shift
    that is exactly a linear combination of the physical boundary vectors.
    """
    # Create a boundary condition matrix N
    # Physical boundary vectors are B * N
    B = ImmutableDenseMatrix([[2, 1], [1, 2]])
    N = ImmutableDenseMatrix([[4, -1], [0, 4]])

    boundaries = PeriodicBoundary(N)

    # We want to check wrapping a point
    test_rep = col(5, -6)

    # Mocking Lattice-like behavior since boundary.wrap acts on relative indices,
    # and offset relies on Lattice basis. Here boundary.basis is N.
    # The actual physical shift is (B * N) * k, but inside boundary.py,
    # the shift applied is N * k
    wrapped_rep = boundaries.wrap(test_rep)

    # The shift in representation space is: test_rep - wrapped_rep
    rep_shift = test_rep - wrapped_rep

    # This shift must be an integer combination of the columns of N.
    # N * k = rep_shift => k = N^{-1} * rep_shift
    k = N.inv() @ rep_shift
    assert all(val.is_integer for val in k)

    # The physical shift is B * rep_shift.
    # The physical boundaries are B * N.
    # We can also verify that the physical shift is an integer combination of physical boundaries:
    # (B*N)^{-1} * physical_shift = N^{-1} * B^{-1} * B * rep_shift = N^{-1} * rep_shift = k
    physical_shift = B @ rep_shift
    physical_boundaries = B @ N
    k_phys = physical_boundaries.inv() @ physical_shift
    assert all(val.is_integer for val in k_phys)
    assert k == k_phys


def test_periodic_boundary_representatives_physical_domain():
    """
    Test that all generated representatives are properly wrapped within the
    fundamental domain defined by the wrapping rules.
    """
    N = ImmutableDenseMatrix([[4, -2], [1, 3]])
    boundaries = PeriodicBoundary(N)

    reps = boundaries.representatives()

    # By definition, wrapping a representative should return itself
    for rep in reps:
        assert boundaries.wrap(rep) == rep

    # And there should be exactly abs(det(N)) representatives
    assert len(reps) == abs(int(N.det()))


def test_lattice_rebase_physical_invariance():
    """
    Test that offsetting and rebasing an Offset between two lattices related by a basis
    transform correctly preserves the absolute physical space position and periodicity.
    """

    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    boundaries = PeriodicBoundary(basis * 4)
    lattice = Lattice(
        basis=basis,
        boundaries=boundaries,
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )

    # Basis transform M
    M = ImmutableDenseMatrix([[1, 4], [0, 1]])
    T = BasisTransform(M)
    lat_new = T(lattice)

    # Two points
    a = Offset(rep=ImmutableDenseMatrix([1, 1]), space=lattice)
    b = Offset(rep=ImmutableDenseMatrix([0, 4]), space=lattice)

    # In old lattice, a + b = [1, 5] which wraps to [1, 1] because period is 4
    c = a + b

    a_new = a.rebase(lat_new)
    b_new = b.rebase(lat_new)
    c_new = a_new + b_new

    # Even after mapping, the physical vector should be invariant
    # Note: b_new wraps to zero since it's exactly on a boundary vector mapping.
    # Therefore, c_new physical vector must be identically aligned in the physical boundary!

    phys_c = c.to_vec(ImmutableDenseMatrix)
    phys_c_new = c_new.to_vec(ImmutableDenseMatrix)

    # The physical vectors might not be strictly identical if they wrapped to different
    # cells, but their DIFFERENCE must be an integer multiple of the PHYSICAL boundaries.
    physical_boundaries = lattice.basis @ lattice.boundaries.basis
    diff = phys_c_new - phys_c
    k = physical_boundaries.inv() @ diff

    assert all(val.is_integer for val in k)


def assert_equivalent_mod_physical_boundaries(
    reference_vec: ImmutableDenseMatrix,
    candidate_vec: ImmutableDenseMatrix,
    physical_boundaries: ImmutableDenseMatrix,
) -> None:
    """
    Assert candidate_vec == reference_vec modulo integer combinations of
    physical boundary vectors.
    """
    diff = candidate_vec - reference_vec
    coeffs, params = physical_boundaries.gauss_jordan_solve(diff)

    # Full-rank boundary basis should yield a unique solution.
    assert params.shape == (0, 1)
    assert all(value.is_integer for value in coeffs)


def test_lattice_rebase_physical_invariance_3d():
    """
    Validate that a 3D basis transform preserves physical position modulo
    the transformed periodic boundaries.
    """
    basis = ImmutableDenseMatrix.eye(3)
    boundaries = PeriodicBoundary(ImmutableDenseMatrix.diag(4, 3, 5))
    lattice = Lattice(
        basis=basis,
        boundaries=boundaries,
        unit_cell={"r": ImmutableDenseMatrix([0, 0, 0])},
    )

    # Integer, unimodular transform to keep lattice indexing integral.
    M = ImmutableDenseMatrix([[1, 1, 0], [0, 1, 2], [0, 0, 1]])
    lat_new = BasisTransform(M)(lattice)

    a = Offset(rep=ImmutableDenseMatrix([1, 2, 3]), space=lattice)
    b = Offset(rep=ImmutableDenseMatrix([7, -1, 8]), space=lattice)
    c = a + b

    c_new = a.rebase(lat_new) + b.rebase(lat_new)

    phys_c = c.to_vec(ImmutableDenseMatrix)
    phys_c_new = c_new.to_vec(ImmutableDenseMatrix)

    physical_boundaries = lattice.basis @ lattice.boundaries.basis
    assert_equivalent_mod_physical_boundaries(phys_c, phys_c_new, physical_boundaries)


def test_affine_transform_boundary_condition_3d():
    """
    In 3D, affine-transformed offsets on a periodic lattice should be equivalent
    to the unwrapped affine image modulo physical boundary vectors.
    """
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 3, 5)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0, 0])},
    )

    x, y, z = sy.symbols("x y z")
    t = AffineTransform(
        irrep=ImmutableDenseMatrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        axes=(x, y, z),
        offset=Offset(rep=ImmutableDenseMatrix([5, -4, 7]), space=lattice),
        basis_function_order=1,
    )

    point = Offset(rep=ImmutableDenseMatrix([3, 2, 4]), space=lattice)
    transformed = t @ point

    # Unwrapped affine action in representation coordinates.
    unwrapped_rep = t.irrep @ point.rep + t.offset.rep
    unwrapped_phys = lattice.basis @ unwrapped_rep
    wrapped_phys = transformed.to_vec(ImmutableDenseMatrix)

    physical_boundaries = lattice.basis @ lattice.boundaries.basis
    assert_equivalent_mod_physical_boundaries(
        wrapped_phys, unwrapped_phys, physical_boundaries
    )
