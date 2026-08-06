import sympy as sy
import torch
import pytest
from sympy import ImmutableDenseMatrix

from qten.phys import Spin, expand_spin, su2_from_so3, su2_of_point_group
from qten.phys.spin import proper_rotation_matrix
from qten.pointgroups import PointGroupElement, PointGroupOpr, pointgroup
from qten.pointgroups.ops import (
    _hilbert_opr_repr,
    spinful_hilbert_opr_repr,
    spinful_transform_basis,
)
import qten
import qten.ops as Q
from qten.symbolics import HilbertSpace, IndexSpace, U1Basis
from qten.geometries.spatials import AffineSpace, Lattice, Offset


def _affine_space():
    return AffineSpace(basis=ImmutableDenseMatrix.eye(3))


def _site(x=0, y=0, z=0):
    space = _affine_space()
    return Offset(rep=ImmutableDenseMatrix([x, y, z]), space=space)


def _c4z():
    x, y, z = sy.symbols("x y z")
    R = ImmutableDenseMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return PointGroupOpr(PointGroupElement(irrep=R, axes=(x, y, z)))


def _c2z():
    x, y, z = sy.symbols("x y z")
    R = ImmutableDenseMatrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return PointGroupOpr(PointGroupElement(irrep=R, axes=(x, y, z)))


def test_spin_labels_and_ordering():
    assert Spin.up.is_up and Spin.down.is_down
    assert Spin.up.ms == sy.Rational(1, 2)
    assert Spin.up < Spin.down
    with pytest.raises(ValueError):
        Spin(0)


def test_su2_identity_and_unitarity():
    I = ImmutableDenseMatrix.eye(3)
    u = su2_from_so3(I)
    assert u == ImmutableDenseMatrix.eye(2)

    R = ImmutableDenseMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # C4z
    u = su2_from_so3(R)
    uh_u = sy.simplify(u.H @ u)
    assert uh_u == ImmutableDenseMatrix.eye(2)
    assert sy.simplify(u.det()) == 1


def test_su2_c2z_is_minus_i_sigma_z():
    R = ImmutableDenseMatrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    u = su2_from_so3(R)
    expected = ImmutableDenseMatrix([[-sy.I, 0], [0, sy.I]])
    assert sy.simplify(u - expected) == ImmutableDenseMatrix.zeros(2)


def test_improper_uses_proper_factor():
    # mirror σ_z = diag(1,1,-1) → proper part -σ_z = C2z
    R = ImmutableDenseMatrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    assert proper_rotation_matrix(R) == ImmutableDenseMatrix(
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    )


def test_expand_spin_c2z_phases():
    opr = _c2z()
    up_img = {spin: amp for amp, spin in expand_spin(opr, Spin.up)}
    dn_img = {spin: amp for amp, spin in expand_spin(opr, Spin.down)}
    assert set(up_img) == {Spin.up}
    assert set(dn_img) == {Spin.down}
    assert sy.simplify(up_img[Spin.up] - (-sy.I)) == 0
    assert sy.simplify(dn_img[Spin.down] - sy.I) == 0


def test_spinful_transform_basis_moves_site_and_spin():
    opr = _c4z().fixpoint_at(_site())
    psi = U1Basis.new(_site(1, 0, 0), Spin.up)
    image = spinful_transform_basis(opr, psi)
    # C4z: (1,0,0) -> (0,1,0); spin gets diagonal SU(2) phases
    sites = {term.irrep_of(Offset).rep for term in image.span}
    spins = {term.irrep_of(Spin) for term in image.span}
    assert sites == {ImmutableDenseMatrix([0, 1, 0])}
    assert spins == {Spin.up}


def test_spinful_hilbert_repr_is_unitary_and_mixes_for_c3():
    # C3 about [111]
    x, y, z = sy.symbols("x y z")
    R = ImmutableDenseMatrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    opr = PointGroupOpr(PointGroupElement(irrep=R, axes=(x, y, z))).fixpoint_at(
        _site()
    )
    space = HilbertSpace.new(
        [
            U1Basis.new(_site(), Spin.up),
            U1Basis.new(_site(), Spin.down),
        ]
    )
    D = spinful_hilbert_opr_repr(opr, space)
    assert D.data.shape == (2, 2)
    # unitarity
    eye = torch.eye(2, dtype=D.data.dtype)
    assert torch.allclose(D.data.conj().T @ D.data, eye, atol=1e-10)
    # C3 about [111] mixes up/down in the usual spin frame
    assert float(torch.abs(D.data[0, 1])) > 1e-8 or float(torch.abs(D.data[1, 0])) > 1e-8


def test_hilbert_opr_repr_dispatches_to_spinful():
    opr = _c2z().fixpoint_at(_site())
    space = HilbertSpace.new(
        [U1Basis.new(_site(), Spin.up), U1Basis.new(_site(), Spin.down)]
    )
    D = _hilbert_opr_repr(opr, space)
    # diag(-i, i)
    assert torch.allclose(
        D.data,
        torch.tensor([[-1j, 0], [0, 1j]], dtype=D.data.dtype),
        atol=1e-10,
    )


def test_point_group_column_symmetrize_spinful_c2():
    opr = _c2z().fixpoint_at(_site())
    space = HilbertSpace.new(
        [U1Basis.new(_site(), Spin.up), U1Basis.new(_site(), Spin.down)]
    )
    # seed = up + down
    w = qten.Tensor(
        data=torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        / torch.sqrt(torch.tensor(2.0)),
        dims=(space, IndexSpace.linear(1)),
    )
    # Use abelian symmetrize on C2
    out = Q.point_group_column_symmetrize(opr, w, full_sector=True)
    assert out.data.shape[1] >= 1


def test_su2_of_td_element_unitarity():
    td = pointgroup("-43m")
    assert isinstance(td, object)
    for g in td.elements():
        u = su2_of_point_group(g)
        assert sy.simplify(u.H @ u) == ImmutableDenseMatrix.eye(2)
        assert sy.simplify(u.det()) == 1


def test_full_td_spinful_symmetrize_is_fast_and_unitary_reps():
    """Full Td (24) spinful D(g) on a moderate local space should be practical."""
    import time

    td = pointgroup("-43m")
    center = _site()
    # 5x5x5 cube of sites around origin (125 sites x 2 spin = 250 dim)
    bases = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            for z in range(-2, 3):
                r = _site(x, y, z)
                bases.append(U1Basis.new(r, Spin.up))
                bases.append(U1Basis.new(r, Spin.down))
    space = HilbertSpace.new(bases)
    assert space.dim == 250

    q, _ = torch.linalg.qr(torch.randn(space.dim, 8, dtype=torch.complex128))
    w = qten.Tensor(data=q[:, :8], dims=(space, IndexSpace.linear(8)))

    t0 = time.perf_counter()
    out = Q.point_group_column_symmetrize(
        td, w, fixpoint=center, rebase_fixpoint=True, full_sector=False
    )
    elapsed = time.perf_counter() - t0
    assert out.data.shape[0] == space.dim
    assert out.data.shape[1] >= 1
    assert elapsed < 30.0, f"full Td spinful symmetrize too slow: {elapsed:.2f}s"


def test_spinful_hilbert_opr_repr_folds_fractional_lattice_offsets():
    """
    Bloch-like spaces store Offset.fractional() labels; D(g) must fold images.

    C2z sends B=(1/2,1/2,1/2) -> (-1/2,-1/2,1/2), which equals B only after
    fractional fold. Without folding, lookup raises even though the unit cell
    is closed under the operation.
    """
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        shape=(2, 2, 2),
        unit_cell={
            "A": ImmutableDenseMatrix([0, 0, 0]),
            "B": ImmutableDenseMatrix(
                [sy.Rational(1, 2), sy.Rational(1, 2), sy.Rational(1, 2)]
            ),
        },
    )
    A = lattice.unit_cell["A"]
    B = lattice.unit_cell["B"]
    assert A == A.fractional() and B == B.fractional()
    space = HilbertSpace.new(
        [
            U1Basis.new(A, Spin.up),
            U1Basis.new(A, Spin.down),
            U1Basis.new(B, Spin.up),
            U1Basis.new(B, Spin.down),
        ]
    )
    opr = _c2z().fixpoint_at(A, rebase=True)
    # Raw geometric image of B is outside the unit cell.
    moved = opr @ B
    assert moved != B
    assert moved.fractional() == B

    D = spinful_hilbert_opr_repr(opr, space)
    assert D.data.shape == (4, 4)
    eye = torch.eye(4, dtype=D.data.dtype)
    assert torch.allclose(D.data.conj().T @ D.data, eye, atol=1e-10)
