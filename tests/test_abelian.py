import sympy as sy
import torch
import pytest
from dataclasses import dataclass
from sympy import ImmutableDenseMatrix
from typing import cast

from qten.pointgroups import pointgroup
from qten.pointgroups.abelian import (
    AbelianBasis,
    AbelianGroup,
    AbelianOpr,
)
from qten.bands import bandtransform
from qten.geometries.fourier import fourier_transform
from qten.symbolics.state_space import MomentumSpace, brillouin_zone
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis, FuncOpr
from qten.geometries.spatials import AffineSpace, Momentum, Offset
from qten.geometries.spatials import Lattice
from qten.geometries.boundary import PeriodicBoundary
from qten.linalg.tensors import Tensor
from qten.symbolics import Multiple


@dataclass(frozen=True)
class Orb:
    name: str


def _state(*irreps, irrep: sy.Expr = sy.Integer(1)) -> U1Basis:
    return U1Basis(coef=irrep, base=tuple(irreps))


def _space_and_offset(dim: int):
    basis = ImmutableDenseMatrix.eye(dim)
    space = AffineSpace(basis=basis)
    offset = Offset(rep=ImmutableDenseMatrix([0] * dim), space=space)
    return space, offset


def _affine(
    *,
    irrep: ImmutableDenseMatrix,
    axes: tuple[sy.Symbol, ...],
    offset: Offset | None = None,
    basis_function_order: int | None = None,
) -> AbelianOpr:
    _ = basis_function_order
    g = AbelianGroup(irrep=irrep, axes=axes)
    if offset is None:
        return AbelianOpr(g=g)
    return AbelianOpr._from_parts(g=g, offset=offset)


def _transformed(op, obj):
    ret = op(obj)
    return ret.base if type(ret) is Multiple else ret


def _split_result(ret):
    if type(ret) is Multiple:
        return ret.coef, ret.base
    return None, ret


def test_affine_function_dim_and_str():
    x = sy.symbols("x")
    f = AbelianBasis(
        expr=x,
        axes=(x,),
        order=1,
        rep=ImmutableDenseMatrix([1]),
    )
    assert f.dim == 1
    assert str(f) == "x"
    assert repr(f) == "x"


def test_affine_group_full_rep_kronecker_power():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 2], [0, 1]])
    t = _affine(irrep=irrep, axes=(x, y), offset=offset, basis_function_order=2)
    expected = sy.kronecker_product(irrep, irrep)
    assert t.g._raw_euclidean_repr(2) == expected


def test_affine_group_rep_shape_for_order_two():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, 1]])
    t = _affine(irrep=irrep, axes=(x, y), offset=offset, basis_function_order=2)
    # Monomials: x^2, x*y, y^2 -> 3 basis terms.
    assert t.g.euclidean_repr(2).shape == (3, 3)


def test_affine_group_rejects_non_invertible_irrep():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)

    with pytest.raises(ValueError, match="non-zero determinant"):
        _affine(
            irrep=ImmutableDenseMatrix([[1, 0], [0, 0]]),
            axes=(x, y),
            offset=offset,
            basis_function_order=1,
        )


def test_affine_group_rejects_non_numerical_irrep():
    x, y = sy.symbols("x y")
    a = sy.symbols("a")
    _, offset = _space_and_offset(2)

    with pytest.raises(ValueError, match="contain only numerical entries"):
        _affine(
            irrep=ImmutableDenseMatrix([[1, a], [0, 1]]),
            axes=(x, y),
            offset=offset,
            basis_function_order=1,
        )


def test_affine_group_affine_rep_identity_basis():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(2)
    offset = Offset(rep=ImmutableDenseMatrix([1, 2]), space=space)
    irrep = ImmutableDenseMatrix([[2, 0], [0, 3]])
    expected = ImmutableDenseMatrix([[2, 0, 1], [0, 3, 2], [0, 0, 1]])
    actual = irrep.row_join(offset.rep).col_join(
        sy.zeros(1, irrep.cols).row_join(sy.ones(1, 1))
    )
    assert actual == expected


def test_affine_group_affine_rep_non_identity_basis():
    x, y = sy.symbols("x y")
    basis = ImmutableDenseMatrix([[2, 0], [0, 1]])
    space = AffineSpace(basis=basis)
    offset = Offset(rep=ImmutableDenseMatrix([1, 1]), space=space)
    irrep = ImmutableDenseMatrix([[1, 0], [0, 2]])
    expected = ImmutableDenseMatrix([[1, 0, 1], [0, 2, 1], [0, 0, 1]])
    actual = irrep.row_join(offset.rep).col_join(
        sy.zeros(1, irrep.cols).row_join(sy.ones(1, 1))
    )
    assert actual == expected


def test_funcopr_supports_multiple_u1basis():
    shift = FuncOpr(int, lambda x: x + 1)
    psi = U1Basis.new(1)
    weighted = Multiple(sy.Integer(3), psi)

    out = shift(weighted)

    assert type(out) is Multiple
    assert out.coef == sy.Integer(3)
    assert out.base == U1Basis.new(2)


def test_composedopr_supports_multiple_u1basis():
    shift = FuncOpr(int, lambda x: x + 1)
    scale = FuncOpr(int, lambda x: 2 * x)
    psi = U1Basis.new(1)
    weighted = Multiple(sy.Integer(5), psi)

    out = (shift @ scale)(weighted)

    assert type(out) is Multiple
    assert out.coef == sy.Integer(5)
    assert out.base == U1Basis.new(3)


def test_affine_group_rebase_changes_basis_consistently():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 2], [0, 1]])
    t = _affine(irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1)

    new_space = AffineSpace(basis=ImmutableDenseMatrix([[2, 0], [0, 2]]))
    new_t = t.rebase(new_space)

    assert new_t.g.irrep == irrep
    assert new_t.g.axes == t.g.axes
    assert new_t.offset.space == new_space


def test_affine_group_rebase_conjugates_irrep_for_nontrivial_basis_change():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[0, -1], [1, 0]])
    t = _affine(irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1)

    new_space = AffineSpace(basis=ImmutableDenseMatrix([[2, 0], [0, 1]]))
    new_t = t.rebase(new_space)

    change = new_space.basis.inv() @ space.basis
    expected = change @ irrep @ change.inv()
    assert new_t.g.irrep == ImmutableDenseMatrix(expected)


def test_abelian_group_matmul_returns_composed_group():
    x, y = sy.symbols("x y")
    left = AbelianGroup(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
    )
    right = AbelianGroup(
        irrep=ImmutableDenseMatrix([[1, 1], [0, 1]]),
        axes=(x, y),
    )

    composed = left @ right

    assert isinstance(composed, AbelianGroup)
    assert composed.axes == (x, y)
    assert composed.irrep == ImmutableDenseMatrix(left.irrep @ right.irrep)


def test_abelian_group_matmul_aligns_permuted_axes():
    x, y = sy.symbols("x y")
    left = AbelianGroup(
        irrep=ImmutableDenseMatrix([[2, 1], [3, 4]]),
        axes=(x, y),
    )
    right = AbelianGroup(
        irrep=ImmutableDenseMatrix([[5, 6], [7, 8]]),
        axes=(y, x),
    )

    composed = left @ right

    right_aligned = ImmutableDenseMatrix([[8, 7], [6, 5]])
    assert composed.axes == (x, y)
    assert composed.irrep == ImmutableDenseMatrix(left.irrep @ right_aligned)

    fx = AbelianBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = AbelianBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))

    assert composed.euclidean_repr(1) @ fx.rep == ImmutableDenseMatrix([[22], [48]])
    assert composed.euclidean_repr(1) @ fy.rep == ImmutableDenseMatrix([[19], [41]])


def test_abelian_group_matmul_extends_missing_axes_with_identity():
    x, y, z = sy.symbols("x y z")
    left = AbelianGroup(
        irrep=ImmutableDenseMatrix([[2, 3], [5, 7]]),
        axes=(x, y),
    )
    right = AbelianGroup(
        irrep=ImmutableDenseMatrix([[11, 13], [17, 19]]),
        axes=(y, z),
    )

    composed = left @ right

    left_embedded = ImmutableDenseMatrix([[2, 3, 0], [5, 7, 0], [0, 0, 1]])
    right_embedded = ImmutableDenseMatrix([[1, 0, 0], [0, 11, 13], [0, 17, 19]])
    assert composed.axes == (x, y, z)
    assert composed.irrep == ImmutableDenseMatrix(left_embedded @ right_embedded)

    fx = AbelianBasis(
        expr=x,
        axes=(x, y, z),
        order=1,
        rep=ImmutableDenseMatrix([1, 0, 0]),
    )
    fy = AbelianBasis(
        expr=y,
        axes=(x, y, z),
        order=1,
        rep=ImmutableDenseMatrix([0, 1, 0]),
    )
    fz = AbelianBasis(
        expr=z,
        axes=(x, y, z),
        order=1,
        rep=ImmutableDenseMatrix([0, 0, 1]),
    )

    expected_fx = left_embedded @ (right_embedded @ fx.rep)
    expected_fy = left_embedded @ (right_embedded @ fy.rep)
    expected_fz = left_embedded @ (right_embedded @ fz.rep)

    assert composed.euclidean_repr(1) @ fx.rep == expected_fx
    assert composed.euclidean_repr(1) @ fy.rep == expected_fy
    assert composed.euclidean_repr(1) @ fz.rep == expected_fz
    assert expected_fx == ImmutableDenseMatrix([[2], [5], [0]])
    assert expected_fy == ImmutableDenseMatrix([[33], [77], [17]])
    assert expected_fz == ImmutableDenseMatrix([[39], [91], [19]])


def test_abelian_group_matmul_rejects_duplicate_axes():
    x, y = sy.symbols("x y")
    left = AbelianGroup(
        irrep=ImmutableDenseMatrix.eye(2),
        axes=(x, x),
    )
    right = AbelianGroup(
        irrep=ImmutableDenseMatrix.eye(2),
        axes=(x, y),
    )

    with pytest.raises(ValueError, match="axes must be unique"):
        _ = left @ right


def test_affine_group_basis_keys_match_eigenvalues():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, -1]])
    t = _affine(irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1)

    basis = t.g.basis(1)
    assert set(basis.keys()) == {1, -1}
    for val, func in basis.items():
        assert isinstance(func, AbelianBasis)
        assert func.axes == (x, y)
        assert func.order == 1
        assert t.g.euclidean_repr(1) @ func.rep == val * func.rep
        gauge, _ = _split_result(t(func))
        assert gauge == val


def test_affine_transform_eigenfunction_phase():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[-1]])
    t = _affine(irrep=irrep, axes=(x,), offset=offset, basis_function_order=1)
    f = AbelianBasis(expr=x, axes=(x,), order=1, rep=ImmutableDenseMatrix([1]))
    gauge, out = _split_result(t(f))
    assert gauge == -1
    assert out == f


def test_affine_transform_abelian_basis_ignores_offset():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[-1]])
    t0 = _affine(
        irrep=irrep,
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([0]), space=space),
        basis_function_order=1,
    )
    t1 = _affine(
        irrep=irrep,
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([7]), space=space),
        basis_function_order=1,
    )
    f = AbelianBasis(expr=x, axes=(x,), order=1, rep=ImmutableDenseMatrix([1]))

    gauge0, out0 = _split_result(t0(f))
    gauge1, out1 = _split_result(t1(f))

    assert gauge0 == gauge1 == -1
    assert out0 == out1 == f


def test_affine_transform_non_eigenfunction_raises():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, -1]])
    t = _affine(irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1)
    f = AbelianBasis(
        expr=x + y,
        axes=(x, y),
        order=1,
        rep=ImmutableDenseMatrix([1, 1]),
    )
    try:
        t(f)
        assert False, "Expected ValueError for non-eigenfunction."
    except ValueError:
        pass


def test_affine_transform_axes_mismatch_raises():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[1]])
    t = _affine(irrep=irrep, axes=(x,), offset=offset, basis_function_order=1)
    f = AbelianBasis(expr=y, axes=(y,), order=1, rep=ImmutableDenseMatrix([1]))
    try:
        t(f)
        assert False, "Expected ValueError for axes mismatch."
    except ValueError:
        pass


def test_affine_transform_order_mismatch_rebuilds():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[2]])
    t = _affine(irrep=irrep, axes=(x,), offset=offset, basis_function_order=1)
    f = AbelianBasis(expr=x**2, axes=(x,), order=2, rep=ImmutableDenseMatrix([1]))
    gauge, out = _split_result(t(f))
    assert gauge == 4
    assert out == f


def test_affine_transform_zero_basis_vector_raises():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[1]])
    t = _affine(irrep=irrep, axes=(x,), offset=offset, basis_function_order=1)
    f = AbelianBasis(expr=0, axes=(x,), order=1, rep=ImmutableDenseMatrix([0]))
    try:
        t(f)
        assert False, "Expected ValueError for zero basis vector."
    except ValueError:
        pass


def test_affine_transform_offset_identity_same_space():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    irrep = ImmutableDenseMatrix.eye(2)
    t = _affine(
        irrep=irrep,
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
        basis_function_order=1,
    )
    offset = Offset(rep=ImmutableDenseMatrix([3, -4]), space=space)
    result = _transformed(t, offset)
    assert result.space == space
    assert result.rep == offset.rep


def test_affine_transform_offset_translation_only():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    irrep = ImmutableDenseMatrix.eye(2)
    t = _affine(
        irrep=irrep,
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([1, 2]), space=space),
        basis_function_order=1,
    )
    offset = Offset(rep=ImmutableDenseMatrix([3, 4]), space=space)
    result = _transformed(t, offset)
    assert result.rep == ImmutableDenseMatrix([4, 6])


def test_affine_transform_offset_linear_only():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[0, -1], [1, 0]])
    t = _affine(
        irrep=irrep,
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
        basis_function_order=1,
    )
    offset = Offset(rep=ImmutableDenseMatrix([2, 3]), space=space)
    result = _transformed(t, offset)
    assert result.rep == irrep @ offset.rep


def test_affine_transform_offset_linear_and_translation():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[2, 0], [0, 3]])
    t = _affine(
        irrep=irrep,
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([1, 2]), space=space),
        basis_function_order=1,
    )
    offset = Offset(rep=ImmutableDenseMatrix([3, 4]), space=space)
    result = _transformed(t, offset)
    expected = irrep @ offset.rep + ImmutableDenseMatrix([1, 2])
    assert result.rep == expected


def test_affine_transform_offset_rebase_transform_keeps_input_space():
    x, y = sy.symbols("x y")
    space_a = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    space_b = AffineSpace(basis=ImmutableDenseMatrix([[2, 0], [0, 1]]))

    t = _affine(
        irrep=ImmutableDenseMatrix([[2, 0], [0, 3]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([1, 2]), space=space_a),
        basis_function_order=1,
    )
    offset = Offset(rep=ImmutableDenseMatrix([1, 1]), space=space_b)
    result = _transformed(t, offset)

    native_offset = offset.rebase(space_a)
    expected_native = t.g.irrep @ native_offset.rep + t.offset.rep
    expected_rep = (
        Offset(rep=ImmutableDenseMatrix(expected_native), space=space_a)
        .rebase(space_b)
        .rep
    )

    assert result.space == space_b
    assert result.rep == ImmutableDenseMatrix(expected_rep)


def test_affine_transform_fixpoint_at_makes_target_offset_invariant():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    t = _affine(
        irrep=ImmutableDenseMatrix([[2, 1], [0, 3]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([5, -4]), space=space),
        basis_function_order=1,
    )
    fixed_point = Offset(rep=ImmutableDenseMatrix([2, -1]), space=space)

    t_fixed = t.fixpoint_at(fixed_point)
    result = _transformed(t_fixed, fixed_point)

    assert result.rep == fixed_point.rep
    assert result.space == fixed_point.space


def test_affine_transform_fixpoint_at_default_rebases_point_to_transform_base():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    other_space = AffineSpace(basis=ImmutableDenseMatrix.diag(2, 3))
    t = _affine(
        irrep=ImmutableDenseMatrix([[2, 1], [0, 3]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([5, -4]), space=space),
        basis_function_order=1,
    )
    fixed_point_other = Offset(rep=ImmutableDenseMatrix([1, -2]), space=other_space)

    t_fixed = t.fixpoint_at(fixed_point_other)
    fixed_point_base = fixed_point_other.rebase(space)
    result = _transformed(t_fixed, fixed_point_base)

    assert t_fixed.base() == space
    assert result.rep == fixed_point_base.rep
    assert result.space == space


def test_affine_transform_fixpoint_at_rebase_true_rebases_transform_to_point_space():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    other_space = AffineSpace(basis=ImmutableDenseMatrix.diag(2, 3))
    t = _affine(
        irrep=ImmutableDenseMatrix([[2, 1], [0, 3]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([5, -4]), space=space),
        basis_function_order=1,
    )
    fixed_point_other = Offset(rep=ImmutableDenseMatrix([1, -2]), space=other_space)

    t_fixed = t.fixpoint_at(fixed_point_other, rebase=True)
    result = _transformed(t_fixed, fixed_point_other)

    assert t_fixed.base() == other_space
    assert result.rep == fixed_point_other.rep
    assert result.space == other_space


def test_affine_transform_offset_one_dimensional():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)
    t = _affine(
        irrep=ImmutableDenseMatrix([[3]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([2]), space=space),
        basis_function_order=1,
    )
    offset = Offset(rep=ImmutableDenseMatrix([5]), space=space)
    result = _transformed(t, offset)
    assert result.rep == ImmutableDenseMatrix([17])


def test_affine_transform_offset_fixed_point_invariant():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[2, 0], [0, 3]])
    t_offset = ImmutableDenseMatrix([1, 2])
    t = _affine(
        irrep=irrep,
        axes=(x, y),
        offset=Offset(rep=t_offset, space=space),
        basis_function_order=1,
    )

    # Fixed point solves (R - I) * p = -t.
    R_minus_I = irrep - ImmutableDenseMatrix.eye(2)
    p = R_minus_I.inv() @ (-t_offset)
    fixed = Offset(rep=ImmutableDenseMatrix(p), space=space)

    result = _transformed(t, fixed)
    assert result.rep == fixed.rep


def test_affine_transform_direct_action_keeps_closed_values_and_transforms_open_ones():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)

    closed_op = _affine(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([0]), space=space),
        basis_function_order=1,
    )
    v = Offset(rep=ImmutableDenseMatrix([2]), space=space)
    assert closed_op(v) == v

    non_closed_op = _affine(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([1]), space=space),
        basis_function_order=1,
    )
    assert non_closed_op(v) == Offset(rep=ImmutableDenseMatrix([3]), space=space)


def test_affine_transform_u1basis_preserves_nontransformable_and_marks_nonclosure():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)
    t = _affine(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([1]), space=space),
        basis_function_order=1,
    )

    new_ket_offset = t(
        U1Basis(
            coef=sy.Integer(1),
            base=(Offset(rep=ImmutableDenseMatrix([0]), space=space),),
        )
    )
    assert new_ket_offset.coef == sy.Integer(1)
    assert cast(Offset, new_ket_offset.irrep_of(Offset)).rep == ImmutableDenseMatrix(
        [1]
    )

    orb_ket = U1Basis(coef=sy.Integer(1), base=(Orb("p"),))
    new_orb_ket = t(orb_ket)
    assert new_orb_ket == orb_ket


def test_affine_transform_u1state_updates_supported_irreps_and_preserves_coef():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)
    t = _affine(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([1]), space=space),
        basis_function_order=1,
    )

    psi = _state(
        Offset(rep=ImmutableDenseMatrix([0]), space=space),
        Orb("p"),
        irrep=sy.Integer(2),
    )
    new_psi = t(psi)
    assert new_psi.coef == sy.Integer(2)
    assert cast(Offset, new_psi.irrep_of(Offset)).rep == ImmutableDenseMatrix([1])
    assert cast(Orb, new_psi.irrep_of(Orb)).name == "p"


def test_u1span_gram_tracks_basis_order():
    a = _state(Orb("a"))
    b = _state(Orb("b"))
    span_ab = a | b
    span_ba = b | a

    gram = span_ab.cross_gram(span_ba)
    expected = ImmutableDenseMatrix([[0, 1], [1, 0]])
    assert gram == expected


def test_affine_transform_hilbert_matmul_matches_call_output_state():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)
    t = _affine(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([0]), space=space),
        basis_function_order=1,
    )
    h = HilbertSpace.new(
        [
            _state(Offset(rep=ImmutableDenseMatrix([0]), space=space), Orb("a")),
            _state(Offset(rep=ImmutableDenseMatrix([1]), space=space), Orb("b")),
        ]
    )

    out_call = t(h)
    out_matmul = t @ h
    assert out_call == out_matmul


def test_affine_transform_momentum_c4_ignores_translation_and_wraps_fractional():
    x, y = sy.symbols("x y")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    recip = lattice.dual

    # Include non-zero translation; momentum transform should use only linear part.
    t = _affine(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([2, -3]), space=lattice),
        basis_function_order=1,
    )
    k = Momentum(rep=ImmutableDenseMatrix([sy.Rational(1, 4), 0]), space=recip)

    out = t(k)
    assert isinstance(out, Momentum)
    assert out.space == recip
    assert out.rep == ImmutableDenseMatrix([0, sy.Rational(1, 4)])


def test_affine_transform_u1state_transforms_supported_irreps_only():
    x, y = sy.symbols("x y")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    t = _affine(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice),
        basis_function_order=1,
    )

    r = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine)
    m = _state(r, Orb("p"))
    out_state = t(m)
    assert out_state.irrep_of(Orb).name == "p"  # non-transformable irrep preserved
    assert out_state.irrep_of(Offset) == r


def test_affine_transform_hilbert_c4_u1state_mapping():
    x, y = sy.symbols("x y")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    t = _affine(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice),
        basis_function_order=1,
    )

    r_x = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine)
    r_y = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine)
    m_x = _state(r_x, Orb("p"))
    m_y = _state(r_y, Orb("d"))
    h = HilbertSpace.new([m_x, m_y])

    gh_expected = HilbertSpace.new(cast(U1Basis, _transformed(t, m)) for m in h)
    gh = t(h)
    assert gh == gh_expected

    tmat = h.cross_gram(gh)
    assert tmat.dims[0] == h
    assert tmat.dims[1] == gh_expected.rays()

    # mapping_matrix(h, gh_expected, mode_mapping) is identity for this construction.
    expected = torch.eye(2, dtype=torch.complex128)
    assert torch.allclose(tmat.data, expected)


def test_affine_transform_hilbert_applies_nontrivial_u1state_gauge_phase():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    t = _affine(
        irrep=ImmutableDenseMatrix([[-1]]),
        axes=(x,),
        offset=offset,
        basis_function_order=1,
    )

    gauge_basis = AbelianBasis(
        expr=x,
        axes=(x,),
        order=1,
        rep=ImmutableDenseMatrix([1]),
    )
    m = _state(gauge_basis)
    h = HilbertSpace.new([m])

    tmat = h.cross_gram(t @ h)
    expected = torch.tensor([[-1.0 + 0.0j]], dtype=tmat.data.dtype)
    assert torch.allclose(tmat.data, expected)


def test_affine_transform_hilbert_applies_matrix_gauge_block():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(2)
    t = _affine(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=offset,
        basis_function_order=1,
    )

    fx = AbelianBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = AbelianBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))

    h = HilbertSpace.new([_state(fx), _state(fy)])
    tmat = h.cross_gram(t @ h)

    expected = torch.diag(
        torch.tensor([-1.0 + 0.0j, 1.0 + 0.0j], dtype=tmat.data.dtype)
    )
    assert tmat.data.shape == (2, 2)
    assert torch.allclose(tmat.data, expected)


def test_bandtransform_both_preserves_c4_symmetric_momentum_tensor_up_to_alignment():
    x, y = sy.symbols("x y")

    # Square lattice with a 2x2 momentum grid: (0,0), (0,1/2), (1/2,0), (1/2,1/2).
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    k_space = brillouin_zone(lattice.dual)

    # Single-orbital Hilbert space in a square unit cell.
    r0 = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine)
    mode = _state(
        r0,
        AbelianBasis(
            expr=x - sy.I * y,
            axes=(x, y),
            order=1,
            rep=ImmutableDenseMatrix([1, -sy.I]),
        ),
    )
    h_space = HilbertSpace.new([mode])

    # C4-symmetric dispersion: e(k) = cos(2*pi*kx) + cos(2*pi*ky).
    energies = []
    for k in k_space.elements():
        kx = float(k.rep[0])
        ky = float(k.rep[1])
        e = sy.N(sy.cos(2 * sy.pi * kx) + sy.cos(2 * sy.pi * ky))
        energies.append(float(e))
    data = torch.tensor(energies, dtype=torch.complex128).reshape(k_space.dim, 1, 1)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    # C4 rotation in real space.
    c4 = _affine(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice),
        basis_function_order=1,
    )

    tensor_out = bandtransform(c4, tensor_in, opt="both")
    tensor_out = tensor_out.align(0, k_space).align(1, h_space).align(2, h_space)

    assert torch.allclose(tensor_out.data, tensor_in.data)


def test_bandtransform_both_matches_explicit_k_aligned_reference():
    x, y = sy.symbols("x y")

    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    k_space = brillouin_zone(lattice.dual)

    # Two orbitals exchanged by C4 around origin.
    r_x = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2), 0]), space=lattice.affine)
    r_y = Offset(rep=ImmutableDenseMatrix([0, sy.Rational(1, 2)]), space=lattice.affine)
    m_x = _state(r_x, Orb("p"))
    m_y = _state(r_y, Orb("p"))
    h_space = HilbertSpace.new([m_x, m_y])

    # k-dependent nontrivial 2x2 tensor so wrong k-block matching is visible.
    data = torch.zeros((k_space.dim, 2, 2), dtype=torch.complex128)
    for n, k in enumerate(k_space.elements()):
        kx = float(k.rep[0])
        ky = float(k.rep[1])
        phase_x = torch.exp(torch.tensor(-2j * torch.pi * kx, dtype=torch.complex128))
        phase_y = torch.exp(torch.tensor(-2j * torch.pi * ky, dtype=torch.complex128))
        data[n, 0, 0] = 0.3 + 0.1 * n
        data[n, 1, 1] = -0.2 + 0.05j * (n + 1)
        data[n, 0, 1] = 1.0 + 0.4 * phase_x + 0.2j * phase_y
        data[n, 1, 0] = -0.6 + 0.3j * phase_y.conj() + 0.1 * phase_x.conj()
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    c4 = _affine(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice),
        basis_function_order=1,
    )

    def _build_transform_ref(space: HilbertSpace, kspace: MomentumSpace) -> Tensor:
        fractional = FuncOpr(Offset, Offset.fractional)
        gspace = fractional @ (c4 @ space)
        bloch_transform = cast(Tensor, space.cross_gram(gspace)).h(-2, -1)  # (B', B)
        bloch_transform = bloch_transform.replace_dim(0, gspace)
        left_fourier = fourier_transform(kspace, space, gspace)
        right_fourier = fourier_transform(kspace, space, space)
        return cast(Tensor, left_fourier @ bloch_transform @ right_fourier.h(-2, -1))

    mapped_kspace = k_space.map(
        lambda k: cast(Momentum, _transformed(c4, k)).fractional()
    )
    left_ref = (
        _build_transform_ref(h_space, k_space)
        .replace_dim(0, mapped_kspace)
        .align(0, k_space)
    )
    right_ref = (
        _build_transform_ref(h_space, k_space)
        .replace_dim(0, mapped_kspace)
        .align(0, k_space)
    )
    tensor_ref = cast(Tensor, (left_ref @ tensor_in @ right_ref.h(-2, -1)))

    tensor_out = bandtransform(c4, tensor_in, opt="both")
    tensor_out = tensor_out.align(0, k_space).align(1, h_space).align(2, h_space)
    tensor_ref = tensor_ref.align(0, k_space).align(1, h_space).align(2, h_space)

    assert torch.allclose(tensor_out.data, tensor_ref.data)


def test_bandtransform_both_c4_fourfold_roundtrip_complex_tensor():
    x, y = sy.symbols("x y")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    k_space = brillouin_zone(lattice.dual)

    r_x = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2), 0]), space=lattice.affine)
    r_y = Offset(rep=ImmutableDenseMatrix([0, sy.Rational(1, 2)]), space=lattice.affine)
    p_minus = AbelianBasis(
        expr=x - sy.I * y,
        axes=(x, y),
        order=1,
        rep=ImmutableDenseMatrix([1, -sy.I]),
    )
    h_space = HilbertSpace.new([_state(r_x, p_minus), _state(r_y, p_minus)])

    c4 = _affine(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine),
        basis_function_order=1,
    )

    data = torch.zeros((k_space.dim, 2, 2), dtype=torch.complex128)
    for n, k in enumerate(k_space.elements()):
        kx = float(k.rep[0])
        ky = float(k.rep[1])
        phase = torch.exp(
            torch.tensor(2j * torch.pi * (kx - 2 * ky), dtype=torch.complex128)
        )
        data[n, 0, 0] = 0.11 + 0.09 * n
        data[n, 1, 1] = -0.21 + 0.06j * (n + 1)
        data[n, 0, 1] = 0.8 - 0.45j + 0.35 * phase
        data[n, 1, 0] = -0.5 + 0.2j - 0.15 * phase.conj()
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    out = tensor_in
    for _ in range(4):
        out = bandtransform(c4, out, opt="both")

    out = out.align(0, k_space).align(1, h_space).align(2, h_space)
    assert torch.allclose(out.data, tensor_in.data)


def test_affine_query_c3_xy_and_inverse_orientation():
    t = pointgroup("c3-xy:xy")
    t_inv = pointgroup("c3-xy:yx")

    assert t.axes == sy.symbols("x y")
    assert t.irrep * t_inv.irrep == ImmutableDenseMatrix.eye(2)


def test_affine_query_c6_rotates_honeycomb_a_to_b_sublattice():
    triangular = ImmutableDenseMatrix(
        [
            [sy.sqrt(3) / 2, 0],
            [-sy.Rational(1, 2), 1],
        ]
    )

    honeycomb = Lattice(
        basis=triangular,
        unit_cell={
            "a": ImmutableDenseMatrix([sy.Rational(1, 3), sy.Rational(2, 3)]),
            "b": ImmutableDenseMatrix([sy.Rational(2, 3), sy.Rational(1, 3)]),
        },
        shape=(12, 12),
    )

    c6 = pointgroup("c6-xy:xy")
    rotated = (AbelianOpr(g=c6) @ honeycomb.at("a")).fractional()

    assert rotated.rep.applyfunc(sy.simplify) == honeycomb.unit_cell["b"].rep


def test_affine_query_c3_xyz_on_yz_plane():
    t = pointgroup("c3-xyz:yz")
    x, y, z = sy.symbols("x y z")

    assert t.axes == (x, y, z)
    assert t.irrep[0, 0] == 1
    assert t.irrep[0, 1] == 0
    assert t.irrep[0, 2] == 0


def test_affine_query_cyclic_forbids_1d_rotation():
    try:
        pointgroup("c3-x:x")
        assert False, "Expected ValueError for 1D cyclic rotation."
    except ValueError:
        pass


def test_affine_query_mirror_2d_fixed_axis():
    t = pointgroup("m-xy:x")
    expected = ImmutableDenseMatrix([[1, 0], [0, -1]])
    assert t.irrep == expected


def test_affine_query_c6_2d_and_3d_examples():
    t2 = pointgroup("c6-xy:xy")
    assert t2.irrep.shape == (2, 2)

    t3 = pointgroup("c6-xyz:yz")
    assert t3.irrep.shape == (3, 3)
    assert t3.irrep[0, 0] == 1
    assert t3.irrep[0, 1] == 0
    assert t3.irrep[0, 2] == 0


def test_affine_query_mirror_1d_2d_3d_examples():
    t1 = pointgroup("m-x:x")
    assert t1.irrep == ImmutableDenseMatrix([[-1]])

    t2 = pointgroup("m-xy:y")
    assert t2.irrep == ImmutableDenseMatrix([[-1, 0], [0, 1]])

    t3 = pointgroup("m-xyz:yz")
    assert t3.irrep == ImmutableDenseMatrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
