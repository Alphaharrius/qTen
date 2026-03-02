import sympy as sy
import torch
import pytest
from dataclasses import dataclass
from sympy import ImmutableDenseMatrix
from typing import cast

from pyhilbert.affine_transform import (
    AbelianBasis,
    AffineGroupElement,
    pointgroup,
    bandtransform,
)
from pyhilbert.fourier import fourier_transform
from pyhilbert.state_space import MomentumSpace, brillouin_zone
from pyhilbert.hilbert_space import HilbertSpace, Ket, U1Basis, FuncOpr, hilbert
from pyhilbert.spatials import AffineSpace, Momentum, Offset
from pyhilbert.spatials import Lattice
from pyhilbert.tensors import Tensor


@dataclass(frozen=True)
class Orb:
    name: str


def _state(*irreps, irrep: sy.Expr = sy.Integer(1)) -> U1Basis:
    return U1Basis(irrep=irrep, kets=tuple(Ket(x) for x in irreps))


def _space_and_offset(dim: int):
    basis = ImmutableDenseMatrix.eye(dim)
    space = AffineSpace(basis=basis)
    offset = Offset(rep=ImmutableDenseMatrix([0] * dim), space=space)
    return space, offset


def _transformed(op, obj):
    _, out = op(obj)
    return out


def test_affine_function_dim_and_str():
    x = sy.symbols("x")
    f = AbelianBasis(
        expr=x,
        axes=(x,),
        order=1,
        rep=ImmutableDenseMatrix([1]),
    )
    assert f.dim == 1
    assert "AbelianBasis(x)" in str(f)
    assert "AbelianBasis(x)" in repr(f)


def test_affine_group_full_rep_kronecker_power():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 2], [0, 1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=2
    )
    expected = sy.kronecker_product(irrep, irrep)
    assert t.full_rep == expected


def test_affine_group_rep_shape_for_order_two():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, 1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=2
    )
    # Monomials: x^2, x*y, y^2 -> 3 basis terms.
    assert t.rep.shape == (3, 3)


def test_affine_group_affine_rep_identity_basis():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(2)
    offset = Offset(rep=ImmutableDenseMatrix([1, 2]), space=space)
    irrep = ImmutableDenseMatrix([[2, 0], [0, 3]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )
    expected = ImmutableDenseMatrix([[2, 0, 1], [0, 3, 2], [0, 0, 1]])
    assert t.affine_rep == expected


def test_affine_group_affine_rep_non_identity_basis():
    x, y = sy.symbols("x y")
    basis = ImmutableDenseMatrix([[2, 0], [0, 1]])
    space = AffineSpace(basis=basis)
    offset = Offset(rep=ImmutableDenseMatrix([1, 1]), space=space)
    irrep = ImmutableDenseMatrix([[1, 0], [0, 2]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )
    expected = ImmutableDenseMatrix([[1, 0, 2], [0, 2, 1], [0, 0, 1]])
    assert t.affine_rep == expected


def test_affine_group_rebase_changes_space_only():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, 1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )

    new_space = AffineSpace(basis=ImmutableDenseMatrix([[2, 0], [0, 2]]))
    new_t = t.rebase(new_space)

    assert new_t.irrep == t.irrep
    assert new_t.axes == t.axes
    assert new_t.basis_function_order == t.basis_function_order
    assert new_t.offset.space == new_space


def test_affine_group_basis_keys_match_eigenvalues():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, -1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )

    basis = t.basis
    assert set(basis.keys()) == {1, -1}
    for val, func in basis.items():
        assert isinstance(func, AbelianBasis)
        assert func.axes == (x, y)
        assert func.order == 1
        assert t.rep @ func.rep == val * func.rep
        gauge, _ = t(func)
        assert gauge == val


def test_affine_transform_eigenfunction_phase():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[-1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x,), offset=offset, basis_function_order=1
    )
    f = AbelianBasis(expr=x, axes=(x,), order=1, rep=ImmutableDenseMatrix([1]))
    gauge, _ = t(f)
    assert gauge == -1


def test_affine_transform_non_eigenfunction_raises():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, -1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )
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
    t = AffineGroupElement(
        irrep=irrep, axes=(x,), offset=offset, basis_function_order=1
    )
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
    t = AffineGroupElement(
        irrep=irrep, axes=(x,), offset=offset, basis_function_order=1
    )
    f = AbelianBasis(expr=x**2, axes=(x,), order=2, rep=ImmutableDenseMatrix([1]))
    gauge, _ = t(f)
    assert gauge == 4


def test_affine_transform_zero_basis_vector_raises():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x,), offset=offset, basis_function_order=1
    )
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
    t = AffineGroupElement(
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
    t = AffineGroupElement(
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
    t = AffineGroupElement(
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
    t = AffineGroupElement(
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

    t = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[2, 0], [0, 3]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([1, 2]), space=space_a),
        basis_function_order=1,
    )
    offset = Offset(rep=ImmutableDenseMatrix([1, 1]), space=space_b)
    result = _transformed(t, offset)

    t_b = t.rebase(space_b)
    hom = offset.rep.col_join(sy.ones(1, 1))
    expected_hom = t_b.affine_rep @ hom
    expected_rep = expected_hom[:-1, :]

    assert result.space == space_b
    assert result.rep == ImmutableDenseMatrix(expected_rep)


def test_affine_transform_with_nontrivial_origin_matches_original_action():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[2, 1], [0, 3]])
    t = AffineGroupElement(
        irrep=irrep,
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([1, -2]), space=space),
        basis_function_order=1,
    )
    origin = Offset(rep=ImmutableDenseMatrix([4, -1]), space=space)
    target = Offset(rep=ImmutableDenseMatrix([7, 5]), space=space)

    target_prime = Offset(rep=target.rep - origin.rep, space=space)
    t_prime = t.with_origin(origin)
    result_prime = _transformed(t_prime, target_prime)
    result = Offset(rep=result_prime.rep + origin.rep, space=space)

    expected = _transformed(t, target)
    assert result.rep == expected.rep


def test_affine_transform_with_origin_at_fixed_point_keeps_origin_fixed():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[2, 0], [0, 3]])
    t_offset = ImmutableDenseMatrix([1, -2])
    t = AffineGroupElement(
        irrep=irrep,
        axes=(x, y),
        offset=Offset(rep=t_offset, space=space),
        basis_function_order=1,
    )

    R_minus_I = irrep - ImmutableDenseMatrix.eye(2)
    origin_rep = R_minus_I.inv() @ (-t_offset)
    origin = Offset(rep=ImmutableDenseMatrix(origin_rep), space=space)
    t_prime = t.with_origin(origin)

    target_prime = Offset(rep=ImmutableDenseMatrix([0, 0]), space=space)
    result_prime = _transformed(t_prime, target_prime)
    result = Offset(rep=result_prime.rep + origin.rep, space=space)
    assert result.rep == origin.rep


def test_affine_group_element_order_c3_c4():
    x, y = sy.symbols("x y")
    space, _ = _space_and_offset(2)
    zero = Offset(rep=ImmutableDenseMatrix([0, 0]), space=space)

    # C4: 90-degree rotation.
    irrep_c4 = ImmutableDenseMatrix([[0, -1], [1, 0]])
    c4 = AffineGroupElement(
        irrep=irrep_c4,
        axes=(x, y),
        offset=zero,
        basis_function_order=1,
    )
    assert len(c4.group_elements(max_order=8)) == 4

    # C3: 120-degree rotation.
    cos = sy.Rational(-1, 2)
    sin = sy.sqrt(3) / 2
    irrep_c3 = ImmutableDenseMatrix([[cos, -sin], [sin, cos]])
    c3 = AffineGroupElement(
        irrep=irrep_c3,
        axes=(x, y),
        offset=zero,
        basis_function_order=1,
    )
    assert len(c3.group_elements(max_order=8)) == 3


def test_affine_transform_offset_one_dimensional():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)
    t = AffineGroupElement(
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
    t = AffineGroupElement(
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


def test_affine_transform_eigen_opr_enforces_closure():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)

    closed_op = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([0]), space=space),
        basis_function_order=1,
    ).eigen_opr()
    v = Offset(rep=ImmutableDenseMatrix([2]), space=space)
    gauge, out = closed_op(v)
    assert gauge == 1
    assert out == v

    non_closed_op = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([1]), space=space),
        basis_function_order=1,
    ).eigen_opr()
    with pytest.raises(ValueError, match="closure-preserving"):
        non_closed_op(v)


def test_affine_transform_ket_preserves_nontransformable_and_marks_nonclosure():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)
    t = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([1]), space=space),
        basis_function_order=1,
    )

    gauge_offset, new_ket_offset = t(
        Ket(Offset(rep=ImmutableDenseMatrix([0]), space=space))
    )
    assert gauge_offset is None
    assert cast(Offset, new_ket_offset.irrep).rep == ImmutableDenseMatrix([1])

    orb_ket = Ket(Orb("p"))
    gauge_orb, new_orb_ket = t(orb_ket)
    assert gauge_orb == 1
    assert new_orb_ket == orb_ket


def test_affine_transform_u1state_propagates_none_gauge_when_any_ket_nonclosed():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)
    t = AffineGroupElement(
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
    gauge, new_psi = t(psi)
    assert gauge is None
    assert new_psi.irrep == sy.Integer(2)
    assert cast(Offset, new_psi.irrep_of(Offset)).rep == ImmutableDenseMatrix([1])
    assert cast(Orb, new_psi.irrep_of(Orb)).name == "p"


def test_u1span_gram_tracks_basis_order():
    a = _state(Orb("a"))
    b = _state(Orb("b"))
    span_ab = a | b
    span_ba = b | a

    gram = span_ab.gram(span_ba)
    expected = ImmutableDenseMatrix([[0, 1], [1, 0]])
    assert gram == expected


def test_affine_transform_hilbert_matmul_matches_call_output_state():
    x = sy.symbols("x")
    space, _ = _space_and_offset(1)
    t = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([0]), space=space),
        basis_function_order=1,
    )
    h = hilbert(
        [
            _state(Offset(rep=ImmutableDenseMatrix([0]), space=space), Orb("a")),
            _state(Offset(rep=ImmutableDenseMatrix([1]), space=space), Orb("b")),
        ]
    )

    _, out_call = t(h)
    out_matmul = t @ h
    assert out_call == out_matmul


def test_affine_transform_momentum_c4_ignores_translation_and_wraps_fractional():
    x, y = sy.symbols("x y")
    lattice = Lattice(basis=ImmutableDenseMatrix.eye(2), shape=(4, 4))
    recip = lattice.dual

    # Include non-zero translation; momentum transform should use only linear part.
    t = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([2, -3]), space=lattice.affine),
        basis_function_order=1,
    )
    k = Momentum(rep=ImmutableDenseMatrix([sy.Rational(1, 4), 0]), space=recip)

    gauge, out = t(k)
    assert gauge is None
    assert isinstance(out, Momentum)
    assert out.space == recip
    assert out.rep == ImmutableDenseMatrix([0, sy.Rational(1, 4)])


def test_affine_transform_u1state_transforms_supported_irreps_only():
    x, y = sy.symbols("x y")
    lattice = Lattice(basis=ImmutableDenseMatrix.eye(2), shape=(2, 2))
    t = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine),
        basis_function_order=1,
    )

    r = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine)
    m = _state(r, Orb("p"))
    gauge, out_state = t(m)

    assert gauge == 1
    assert out_state.irrep_of(Orb).name == "p"  # non-transformable irrep preserved
    assert out_state.irrep_of(Offset) == r


def test_affine_transform_hilbert_c4_u1state_mapping():
    x, y = sy.symbols("x y")
    lattice = Lattice(basis=ImmutableDenseMatrix.eye(2), shape=(2, 2))
    t = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine),
        basis_function_order=1,
    )

    r_x = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine)
    r_y = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine)
    m_x = _state(r_x, Orb("p"))
    m_y = _state(r_y, Orb("d"))
    h = hilbert([m_x, m_y])

    tmat, gh = t(h)
    tmat = cast(Tensor, tmat)
    assert tmat.dims[0] == h

    gh_expected = hilbert(cast(U1Basis, _transformed(t, m)) for m in h)
    assert gh == gh_expected
    assert tmat.dims[1] == gh_expected.unit()

    # mapping_matrix(h, gh_expected, mode_mapping) is identity for this construction.
    expected = torch.eye(2, dtype=torch.complex128)
    assert torch.allclose(tmat.data, expected)


def test_affine_transform_hilbert_applies_nontrivial_u1state_gauge_phase():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    t = AffineGroupElement(
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
    h = hilbert([m])

    tmat, _ = t(h)
    tmat = cast(Tensor, tmat)
    expected = torch.tensor([[-1.0 + 0.0j]], dtype=tmat.data.dtype)
    assert torch.allclose(tmat.data, expected)


def test_affine_transform_hilbert_applies_matrix_gauge_block():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(2)
    t = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=offset,
        basis_function_order=1,
    )

    fx = AbelianBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = AbelianBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))

    h = hilbert([_state(fx), _state(fy)])
    tmat, _ = t(h)
    tmat = cast(Tensor, tmat)

    expected = torch.diag(
        torch.tensor([-1.0 + 0.0j, 1.0 + 0.0j], dtype=tmat.data.dtype)
    )
    assert tmat.data.shape == (2, 2)
    assert torch.allclose(tmat.data, expected)


def test_bandtransform_both_preserves_c4_symmetric_momentum_tensor_up_to_alignment():
    x, y = sy.symbols("x y")

    # Square lattice with a 2x2 momentum grid: (0,0), (0,1/2), (1/2,0), (1/2,1/2).
    lattice = Lattice(basis=ImmutableDenseMatrix.eye(2), shape=(2, 2))
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
    h_space = hilbert([mode])

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
    c4 = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine),
        basis_function_order=1,
    )

    tensor_out = bandtransform(c4, tensor_in, opt="both")
    tensor_out = tensor_out.align(0, k_space).align(1, h_space).align(2, h_space)

    assert torch.allclose(tensor_out.data, tensor_in.data)


def test_bandtransform_both_matches_explicit_k_aligned_reference():
    x, y = sy.symbols("x y")

    lattice = Lattice(basis=ImmutableDenseMatrix.eye(2), shape=(2, 2))
    k_space = brillouin_zone(lattice.dual)

    # Two orbitals exchanged by C4 around origin.
    r_x = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2), 0]), space=lattice.affine)
    r_y = Offset(rep=ImmutableDenseMatrix([0, sy.Rational(1, 2)]), space=lattice.affine)
    m_x = _state(r_x, Orb("p"))
    m_y = _state(r_y, Orb("p"))
    h_space = hilbert([m_x, m_y])

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

    c4 = AffineGroupElement(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine),
        basis_function_order=1,
    )

    def _build_transform_ref(space: HilbertSpace, kspace: MomentumSpace) -> Tensor:
        fractional = FuncOpr(Offset, Offset.fractional)
        gspace = fractional @ (c4 @ space)
        bloch_transform = cast(Tensor, space.gram(gspace)).h(-2, -1)  # (B', B)
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
    lattice = Lattice(basis=ImmutableDenseMatrix.eye(2), shape=(2, 2))
    k_space = brillouin_zone(lattice.dual)

    r_x = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2), 0]), space=lattice.affine)
    r_y = Offset(rep=ImmutableDenseMatrix([0, sy.Rational(1, 2)]), space=lattice.affine)
    p_minus = AbelianBasis(
        expr=x - sy.I * y,
        axes=(x, y),
        order=1,
        rep=ImmutableDenseMatrix([1, -sy.I]),
    )
    h_space = hilbert([_state(r_x, p_minus), _state(r_y, p_minus)])

    c4 = AffineGroupElement(
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
    t = pointgroup("c3-xy:xy-o2")
    t_inv = pointgroup("c3-xy:yx-o2")

    assert t.axes == sy.symbols("x y")
    assert t.basis_function_order == 2
    assert t.irrep * t_inv.irrep == ImmutableDenseMatrix.eye(2)


def test_affine_query_c3_xyz_on_yz_plane():
    t = pointgroup("c3-xyz:yz-o2")
    x, y, z = sy.symbols("x y z")

    assert t.axes == (x, y, z)
    assert t.basis_function_order == 2
    assert t.irrep[0, 0] == 1
    assert t.irrep[0, 1] == 0
    assert t.irrep[0, 2] == 0


def test_affine_query_cyclic_forbids_1d_rotation():
    try:
        pointgroup("c3-x:x-o2")
        assert False, "Expected ValueError for 1D cyclic rotation."
    except ValueError:
        pass


def test_affine_query_mirror_2d_fixed_axis():
    t = pointgroup("m-xy:x-o1")
    expected = ImmutableDenseMatrix([[1, 0], [0, -1]])
    assert t.irrep == expected


def test_affine_query_c6_2d_and_3d_examples():
    t2 = pointgroup("c6-xy:xy-o2")
    assert t2.irrep.shape == (2, 2)
    assert t2.basis_function_order == 2

    t3 = pointgroup("c6-xyz:yz-o2")
    assert t3.irrep.shape == (3, 3)
    assert t3.irrep[0, 0] == 1
    assert t3.irrep[0, 1] == 0
    assert t3.irrep[0, 2] == 0


def test_affine_query_mirror_1d_2d_3d_examples():
    t1 = pointgroup("m-x:x-o1")
    assert t1.irrep == ImmutableDenseMatrix([[-1]])

    t2 = pointgroup("m-xy:y-o1")
    assert t2.irrep == ImmutableDenseMatrix([[-1, 0], [0, 1]])

    t3 = pointgroup("m-xyz:yz-o1")
    assert t3.irrep == ImmutableDenseMatrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
