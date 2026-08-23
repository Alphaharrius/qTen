"""Temporary 1D / 2D / 3D smoke tests for the spin-first point-group lift.

Run:
    uv run pytest tests/test_pointgroup_spin_temp.py -q -s
"""

import sympy as sy
import torch

import qten
from qten.geometries.spatials import AffineSpace, Offset
from qten.phys import Spin, su2_from_so3, su2_of_point_group
from qten.phys.spin import proper_rotation_matrix
from qten.pointgroups import PointGroupOpr, SpinorIrrepSector, hilbert_repr, pointgroup
from qten.symbolics import HilbertSpace, IndexSpace, U1Basis


def _center(dim: int) -> Offset:
    return Offset(
        sy.ImmutableDenseMatrix.zeros(dim, 1),
        AffineSpace(basis=sy.ImmutableDenseMatrix.eye(dim)),
    )


def _spin_seed(center: Offset) -> qten.Tensor:
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    return qten.Tensor(
        data=torch.eye(2, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(2)),
    )


def _assert_su2(u: sy.ImmutableDenseMatrix) -> None:
    assert sy.simplify(u.H @ u) == sy.eye(2)
    assert sy.simplify(u.det()) == 1


def _assert_lift_of_rotation3(element) -> sy.ImmutableDenseMatrix:
    assert element.rotation3 is not None
    assert element.rotation3.shape == (3, 3)
    u = su2_of_point_group(element)
    expected = su2_from_so3(proper_rotation_matrix(element.rotation3))
    assert sy.simplify(u - expected) == sy.zeros(2)
    _assert_su2(u)
    return u


def _assert_hilbert_is_u(element, center: Offset, u: sy.ImmutableDenseMatrix) -> None:
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    D = hilbert_repr(PointGroupOpr(element).fixpoint_at(center), space).data
    expected = torch.tensor(
        [[complex(sy.N(u[i, j])) for j in range(2)] for i in range(2)],
        dtype=torch.complex128,
    )
    assert torch.allclose(D, expected, rtol=0, atol=1e-12)


def _assert_spinor_symmetrize(group, center: Offset) -> None:
    projected = qten.ops.point_group_column_symmetrize(
        group, _spin_seed(center), full_sector=True, fixpoint=center
    )
    assert projected.data.shape == (2, 2)
    gram = projected.data.conj().T @ projected.data
    assert torch.allclose(
        gram, torch.eye(2, dtype=torch.complex128), rtol=0, atol=1e-11
    )
    assert all(
        isinstance(label.irrep_of(SpinorIrrepSector), SpinorIrrepSector)
        for label in projected.dims[1].elements()
    )


def test_1d_mirror_lifts_c2_about_x_not_z():
    """1D m: spatial is [[-1]], spin uses σ_yz → C2 about x, not a padded C2z."""
    group = pointgroup("m", plane="x")
    assert group.axes == (sy.Symbol("x"),)
    assert group.order == 2
    mirror = next(
        element
        for element in group.elements()
        if sy.simplify(element.irrep[0, 0] + 1) == 0
    )
    assert mirror.irrep.shape == (1, 1)
    assert sy.simplify(mirror.rotation3 - sy.diag(-1, 1, 1)) == sy.zeros(3)

    u = _assert_lift_of_rotation3(mirror)
    # C2x is off-diagonal in Sz; a mistaken pad onto z would be diagonal.
    assert sy.simplify(u[0, 1]) != 0
    assert sy.simplify(u[1, 0]) != 0
    assert sy.simplify(u[0, 0]) == 0
    c2z = sy.ImmutableDenseMatrix([[-sy.I, 0], [0, sy.I]])
    assert sy.simplify(u - c2z) != sy.zeros(2)

    center = _center(1)
    _assert_hilbert_is_u(mirror, center, u)
    _assert_spinor_symmetrize(group, center)


def test_1d_affine_mirror_stores_rotation3():
    mirror = pointgroup("m-x:x")
    assert mirror.irrep.shape == (1, 1)
    u = _assert_lift_of_rotation3(mirror)
    _assert_hilbert_is_u(mirror, _center(1), u)


def test_2d_c2_is_rotation_about_z():
    """2D '2' is C2z, not the monoclinic C2y that a naive xy cut would keep."""
    group = pointgroup("2", plane="xy")
    assert group.axes == sy.symbols("x y")
    c2 = next(element for element in group.elements() if element.group_order() == 2)
    assert c2.irrep.shape == (2, 2)
    assert sy.simplify(c2.irrep + sy.eye(2)) == sy.zeros(2)
    assert sy.simplify(c2.rotation3 - sy.diag(-1, -1, 1)) == sy.zeros(3)

    u = _assert_lift_of_rotation3(c2)
    assert sy.simplify(u[0, 1]) == 0
    assert sy.simplify(u[1, 0]) == 0

    center = _center(2)
    _assert_hilbert_is_u(c2, center, u)
    _assert_spinor_symmetrize(group, center)


def test_2d_c4v_c4_is_e_minus_i_sz_pi_over_4():
    group = pointgroup("C4v", plane="xy")
    c4 = next(element for element in group.elements() if element.group_order() == 4)
    assert c4.irrep.shape == (2, 2)
    assert sy.simplify(c4.rotation3[2, 2] - 1) == 0
    u = _assert_lift_of_rotation3(c4)
    half = sy.pi / 4
    about_z = sy.ImmutableDenseMatrix(
        [
            [sy.cos(half) - sy.I * sy.sin(half), 0],
            [0, sy.cos(half) + sy.I * sy.sin(half)],
        ]
    )
    assert sy.simplify(u - about_z) == sy.zeros(2) or sy.simplify(
        u - about_z.H
    ) == sy.zeros(2)

    center = _center(2)
    _assert_hilbert_is_u(c4, center, u)
    _assert_spinor_symmetrize(group, center)


def test_2d_c3v_on_111_is_not_about_z():
    group = pointgroup("3m", plane=(1, 1, 1))
    assert group.axes == sy.symbols("x y")
    c3 = next(element for element in group.elements() if element.group_order() == 3)
    u = _assert_lift_of_rotation3(c3)
    assert sy.simplify(u[0, 1]) != 0 or sy.simplify(u[1, 0]) != 0
    _assert_spinor_symmetrize(group, _center(2))


def test_3d_c4v_spatial_equals_rotation3():
    group = pointgroup("4mm")
    c4 = next(element for element in group.elements() if element.group_order() == 4)
    assert c4.irrep.shape == (3, 3)
    assert sy.simplify(c4.irrep - c4.rotation3) == sy.zeros(3)
    u = _assert_lift_of_rotation3(c4)
    center = _center(3)
    _assert_hilbert_is_u(c4, center, u)
    _assert_spinor_symmetrize(group, center)


def test_3d_inversion_does_not_flip_spin():
    group = pointgroup("-1")
    inversion = next(
        element
        for element in group.elements()
        if sy.simplify(element.irrep + sy.eye(3)) == sy.zeros(3)
    )
    u = _assert_lift_of_rotation3(inversion)
    assert sy.simplify(u - sy.eye(2)) == sy.zeros(2)
    center = _center(3)
    _assert_hilbert_is_u(inversion, center, u)
    _assert_spinor_symmetrize(group, center)


def test_3d_td_spinor_projection_is_complete():
    group = pointgroup("-43m")
    center = _center(3)
    _assert_spinor_symmetrize(group, center)


def test_3d_c3v_axis_111_is_not_about_z():
    group = pointgroup("3m", axis=(1, 1, 1))
    assert group.axes == sy.symbols("x y z")
    c3 = next(element for element in group.elements() if element.group_order() == 3)
    u = _assert_lift_of_rotation3(c3)
    assert sy.simplify(u[0, 1]) != 0 or sy.simplify(u[1, 0]) != 0
    x = sy.I * (u[1, 0] + u[0, 1]) / 2
    y = (u[1, 0] - u[0, 1]) / 2
    z = sy.I * (u[0, 0] - u[1, 1]) / 2
    assert sy.simplify(x - y) == 0
    assert sy.simplify(y - z) == 0
    _assert_spinor_symmetrize(group, _center(3))
