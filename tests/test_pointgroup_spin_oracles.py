"""1D / 2D / 3D geometry oracles for spin-first point groups.

Role: construction-time geometry. Spatial matrices may be small; spin must
use the stored 3D ``rotation3``, never a later pad.

Other spin files:
- ``test_spin.py`` — Spin labels, SU(2) numerics, Hilbert D(g), projectors
- ``test_pointgroup_ops.py`` — column / joint / transform helpers
- ``test_pointgroup_registry.py`` — packaged catalog and class alignment
- ``test_spinful_lattices.py`` — Bloch models on real lattices
"""

import sympy as sy
import torch
import pytest

import qten
from qten.geometries.spatials import AffineSpace, Offset
from qten.phys import Spin, su2_from_so3, su2_of_point_group
from qten.phys.spin import proper_rotation_matrix
from qten.pointgroups import (
    FiniteIrrepSector,
    FinitePointGroup,
    PointGroupOpr,
    SpinorIrrepSector,
    hilbert_repr,
    pointgroup,
)
from qten.symbolics import HilbertSpace, IndexSpace, U1Basis


def _center(dim: int = 3) -> Offset:
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


def _assert_lift_of_rotation3(element) -> sy.ImmutableDenseMatrix:
    assert element.rotation3 is not None
    assert element.rotation3.shape == (3, 3)
    u = su2_of_point_group(element)
    expected = su2_from_so3(proper_rotation_matrix(element.rotation3))
    assert sy.simplify(u - expected) == sy.zeros(2)
    assert sy.simplify(u.H @ u) == sy.eye(2)
    assert sy.simplify(u.det()) == 1
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


# --- 1D ---


def test_1d_mirror_lifts_c2_about_x_not_z():
    """1D m: spatial [[-1]]; spin is C2 about x, not a pad onto z."""
    group = pointgroup("m", plane="x")
    assert group.axes == (sy.Symbol("x"),)
    mirror = next(
        element
        for element in group.elements()
        if sy.simplify(element.irrep[0, 0] + 1) == 0
    )
    assert mirror.irrep.shape == (1, 1)
    assert sy.simplify(mirror.rotation3 - sy.diag(-1, 1, 1)) == sy.zeros(3)

    u = _assert_lift_of_rotation3(mirror)
    assert sy.simplify(u[0, 0]) == 0
    assert sy.simplify(u[0, 1]) != 0
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


# --- 2D ---


def test_2d_c2_is_rotation_about_z():
    """Catalog 2D '2' is C2z, not monoclinic C2y."""
    group = pointgroup("2", plane="xy")
    c2 = next(element for element in group.elements() if element.group_order() == 2)
    assert c2.irrep.shape == (2, 2)
    assert sy.simplify(c2.rotation3 - sy.diag(-1, -1, 1)) == sy.zeros(3)
    u = _assert_lift_of_rotation3(c2)
    assert sy.simplify(u[0, 1]) == 0
    center = _center(2)
    _assert_hilbert_is_u(c2, center, u)
    _assert_spinor_symmetrize(group, center)


def test_c4v_xy_lift_is_rotation_about_z():
    group = pointgroup("C4v", plane="xy")
    assert group.axes == sy.symbols("x y")
    c4 = next(element for element in group.elements() if element.group_order() == 4)
    assert sy.simplify(c4.rotation3[2, 2] - 1) == 0
    u = _assert_lift_of_rotation3(c4)
    assert sy.simplify(u[0, 1]) == 0
    half = sy.pi / 4
    expected = sy.ImmutableDenseMatrix(
        [
            [sy.cos(half) - sy.I * sy.sin(half), 0],
            [0, sy.cos(half) + sy.I * sy.sin(half)],
        ]
    )
    assert sy.simplify(u - expected) == sy.zeros(2) or sy.simplify(
        u - expected.H
    ) == sy.zeros(2)
    center = _center(2)
    _assert_hilbert_is_u(c4, center, u)
    _assert_spinor_symmetrize(group, center)


def test_named_plane_cut_keeps_rotation3():
    group = pointgroup("C4v-xy")
    assert all(generator.rotation3 is not None for generator in group.generators)
    assert group.spinor_irreps is not None
    assert group.spinor_irreps["source"] == "qten-su2-principal-v1"


def test_c3v_plane_111_keeps_3d_rotation3():
    group = pointgroup("3m", plane=(1, 1, 1))
    assert group.axes == sy.symbols("x y")
    c3 = next(element for element in group.elements() if element.group_order() == 3)
    u = _assert_lift_of_rotation3(c3)
    assert sy.simplify(u[0, 1]) != 0 or sy.simplify(u[1, 0]) != 0
    _assert_spinor_symmetrize(group, _center(2))


# --- 3D ---


def test_3d_c4v_spatial_equals_rotation3():
    group = pointgroup("4mm")
    c4 = next(element for element in group.elements() if element.group_order() == 4)
    assert sy.simplify(c4.irrep - c4.rotation3) == sy.zeros(3)
    u = _assert_lift_of_rotation3(c4)
    center = _center(3)
    _assert_hilbert_is_u(c4, center, u)
    _assert_spinor_symmetrize(group, center)


def test_c3v_axis_111_lift_is_not_rotation_about_z():
    group = pointgroup("3m", axis=(1, 1, 1))
    c3 = next(element for element in group.elements() if element.group_order() == 3)
    u = _assert_lift_of_rotation3(c3)
    assert sy.simplify(u[0, 1]) != 0 or sy.simplify(u[1, 0]) != 0
    x = sy.I * (u[1, 0] + u[0, 1]) / 2
    y = (u[1, 0] - u[0, 1]) / 2
    z = sy.I * (u[0, 0] - u[1, 1]) / 2
    assert sy.simplify(x - y) == 0
    assert sy.simplify(y - z) == 0
    _assert_spinor_symmetrize(group, _center(3))


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


# --- Policy ---


def test_trivial_spin_td_uses_ordinary_sectors():
    center = _center(3)
    seed = _spin_seed(center)
    electron = qten.ops.point_group_column_symmetrize(
        pointgroup("-43m"), seed, full_sector=True, fixpoint=center
    )
    trivial = qten.ops.point_group_column_symmetrize(
        pointgroup("-43m", spin="trivial"),
        seed,
        full_sector=True,
        fixpoint=center,
    )
    electron_labels = {
        label.irrep_of(SpinorIrrepSector).irrep for label in electron.dims[1].elements()
    }
    trivial_labels = {
        label.irrep_of(FiniteIrrepSector).irrep for label in trivial.dims[1].elements()
    }
    assert electron_labels
    assert trivial_labels
    assert electron_labels != trivial_labels


def test_2d_custom_group_without_rotation3_cannot_lift_spin():
    x, y = sy.symbols("x y")
    custom = FinitePointGroup.from_matrices(
        (sy.ImmutableDenseMatrix([[0, -1], [1, 0]]),),
        axes=(x, y),
        symbol="custom-c4",
    )
    center = _center(2)
    with pytest.raises(ValueError, match="plane= or axis="):
        qten.ops.point_group_column_symmetrize(
            custom, _spin_seed(center), fixpoint=center
        )
