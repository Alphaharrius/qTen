"""Geometry oracles for spin-first point groups."""

import sympy as sy
import torch
import pytest

import qten
from qten.geometries.spatials import AffineSpace, Offset
from qten.phys import Spin, su2_of_point_group
from qten.pointgroups import (
    FiniteIrrepSector,
    FinitePointGroup,
    SpinorIrrepSector,
    pointgroup,
)
from qten.symbolics import HilbertSpace, IndexSpace, U1Basis


def _center():
    affine = AffineSpace(basis=sy.ImmutableDenseMatrix.eye(3))
    return Offset(sy.ImmutableDenseMatrix.zeros(3, 1), affine)


def _spin_seed(center):
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    return qten.Tensor(
        data=torch.eye(2, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(2)),
    )


def test_c4v_xy_lift_is_rotation_about_z():
    group = pointgroup("C4v", plane="xy")
    assert group.axes == sy.symbols("x y")
    c4 = next(element for element in group.elements() if element.group_order() == 4)
    assert c4.rotation3 is not None
    rotation3 = c4.rotation3
    assert sy.simplify(rotation3[2, 2] - 1) == 0
    assert sy.simplify(rotation3[0, 2]) == 0
    assert sy.simplify(rotation3[2, 0]) == 0
    u = su2_of_point_group(c4)
    assert sy.simplify(u[0, 1]) == 0
    assert sy.simplify(u[1, 0]) == 0
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


def test_c3v_axis_111_lift_is_not_rotation_about_z():
    group = pointgroup("3m", axis=(1, 1, 1))
    c3 = next(element for element in group.elements() if element.group_order() == 3)
    u = su2_of_point_group(c3)
    assert sy.simplify(u[0, 1]) != 0 or sy.simplify(u[1, 0]) != 0
    w = (u[0, 0] + u[1, 1]) / 2
    z_component = sy.I * (u[0, 0] - u[1, 1]) / 2
    x_component = sy.I * (u[1, 0] + u[0, 1]) / 2
    y_component = (u[1, 0] - u[0, 1]) / 2
    axis = sy.ImmutableDenseMatrix(
        [sy.simplify(x_component), sy.simplify(y_component), sy.simplify(z_component)]
    )
    assert sy.simplify(w) != 0
    assert sy.simplify(axis[0] - axis[1]) == 0
    assert sy.simplify(axis[1] - axis[2]) == 0
    assert sy.simplify(axis[0]) != 0


def test_trivial_spin_td_uses_ordinary_sectors():
    center = _center()
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


def test_custom_group_without_json_can_symmetrize():
    x, y, z = sy.symbols("x y z")
    custom = FinitePointGroup.from_matrices(
        (sy.ImmutableDenseMatrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),),
        axes=(x, y, z),
        symbol="custom-c2",
    )
    assert custom.irreps is None
    assert custom.spinor_irreps is None
    center = _center()
    projected = qten.ops.point_group_column_symmetrize(
        custom, _spin_seed(center), full_sector=True, fixpoint=center
    )
    assert projected.data.shape == (2, 2)
    assert all(
        isinstance(label.irrep_of(SpinorIrrepSector), SpinorIrrepSector)
        for label in projected.dims[1].elements()
    )


def test_2d_custom_group_without_rotation3_cannot_lift_spin():
    x, y = sy.symbols("x y")
    custom = FinitePointGroup.from_matrices(
        (sy.ImmutableDenseMatrix([[0, -1], [1, 0]]),),
        axes=(x, y),
        symbol="custom-c4",
    )
    center = Offset(
        sy.ImmutableDenseMatrix.zeros(2, 1),
        AffineSpace(basis=sy.ImmutableDenseMatrix.eye(2)),
    )
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    seed = qten.Tensor(
        data=torch.eye(2, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(2)),
    )
    with pytest.raises(ValueError, match="plane= or axis="):
        qten.ops.point_group_column_symmetrize(custom, seed, fixpoint=center)


def test_named_plane_cut_keeps_rotation3():
    group = pointgroup("C4v-xy")
    assert all(generator.rotation3 is not None for generator in group.generators)
    assert group.spinor_irreps is not None
    assert group.spinor_irreps["source"] == "qten-su2-principal-v1"


def test_c3v_plane_111_keeps_3d_rotation3():
    group = pointgroup("3m", plane=(1, 1, 1))
    assert group.axes == sy.symbols("x y")
    assert all(element.rotation3 is not None for element in group.elements())
    c3 = next(element for element in group.elements() if element.group_order() == 3)
    u = su2_of_point_group(c3)
    assert sy.simplify(u[0, 1]) != 0 or sy.simplify(u[1, 0]) != 0
