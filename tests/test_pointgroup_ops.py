import sympy as sy
import torch
from sympy import ImmutableDenseMatrix

from qten.geometries.spatials import AffineSpace, Offset
from qten.linalg.tensors import Tensor
from qten.pointgroups.abelian import AbelianBasis, AbelianGroup, AbelianOpr
from qten.pointgroups.ops import abelian_column_symmetrize
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.ops import hilbert_opr_repr
from qten.symbolics.state_space import IndexSpace


def _state(*irreps, irrep: sy.Expr = sy.Integer(1)) -> U1Basis:
    return U1Basis(coef=irrep, base=tuple(irreps))


def _opr_with_offset(
    irrep: ImmutableDenseMatrix,
    axes: tuple[sy.Symbol, ...],
    offset: Offset,
) -> AbelianOpr:
    opr = AbelianOpr(g=AbelianGroup(irrep=irrep, axes=axes))
    object.__setattr__(opr, "offset", offset)
    return opr


def test_abelian_column_symmetrize_projects_indexspace_columns_to_sector_labels():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = AbelianBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = AbelianBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    h = HilbertSpace.new([_state(fx), _state(fy)])

    w = Tensor(
        data=torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        / torch.sqrt(torch.tensor(2.0, dtype=torch.float64)),
        dims=(h, IndexSpace.linear(1)),
    )
    w_sym = abelian_column_symmetrize(mirror, w, full_sector=True)

    assert isinstance(w_sym.dims[1], HilbertSpace)
    labels = list(w_sym.dims[1].elements())
    sector_phases = torch.tensor(
        [complex(sy.N(mirror(label.irrep_of(AbelianBasis)).coef)) for label in labels],
        dtype=torch.complex128,
    )

    g_full = hilbert_opr_repr(mirror, h)
    expected = torch.diag(sector_phases)
    assert torch.allclose((w_sym.h(-2, -1) @ g_full @ w_sym).data, expected)
    assert set(sector_phases.tolist()) == {1.0 + 0.0j, -1.0 + 0.0j}


def test_abelian_column_symmetrize_defaults_to_one_sector_per_input_column():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = AbelianBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = AbelianBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    h = HilbertSpace.new([_state(fx), _state(fy)])

    w = Tensor(
        data=torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        / torch.sqrt(torch.tensor(2.0, dtype=torch.float64)),
        dims=(h, IndexSpace.linear(1)),
    )
    w_sym = abelian_column_symmetrize(mirror, w)

    assert w_sym.data.shape == (2, 1)
    label = next(iter(w_sym.dims[1].elements()))
    phase = complex(sy.N(mirror(label.irrep_of(AbelianBasis)).coef))
    assert phase in {1.0 + 0.0j, -1.0 + 0.0j}


def test_abelian_column_symmetrize_appends_basis_for_hilbertspace():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = AbelianBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = AbelianBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    row_space = HilbertSpace.new([_state(fx), _state(fy)])
    seed_space = HilbertSpace.new([_state("seed_a"), _state("seed_b")])

    w = Tensor(
        data=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.complex128),
        dims=(row_space, seed_space),
    )
    w_sym = abelian_column_symmetrize(mirror, w)

    labels = list(w_sym.dims[1].elements())
    assert len(labels) == 2
    assert {label.irrep_of(str) for label in labels} == {"seed_a", "seed_b"}
    assert torch.allclose(w_sym.data[:, 0], w_sym.data[:, 1])
    assert all(
        complex(sy.N(mirror(label.irrep_of(AbelianBasis)).coef)) == 1.0 + 0.0j
        for label in labels
    )


def test_abelian_column_symmetrize_adds_degeneracy_tag_for_duplicate_labels():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = AbelianBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = AbelianBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    row_space = HilbertSpace.new([_state(fx), _state(fy)])

    w = Tensor(
        data=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.complex128),
        dims=(row_space, IndexSpace.linear(2)),
    )
    w_sym = abelian_column_symmetrize(mirror, w)

    labels = list(w_sym.dims[1].elements())
    assert len(labels) == 2
    assert all(label.irrep_of(int) in (0, 1) for label in labels)


def test_abelian_column_symmetrize_full_sector_expands_mixed_column():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = AbelianBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = AbelianBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    h = HilbertSpace.new([_state(fx), _state(fy)])

    w = Tensor(
        data=torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        / torch.sqrt(torch.tensor(2.0, dtype=torch.float64)),
        dims=(h, IndexSpace.linear(1)),
    )
    w_sym = abelian_column_symmetrize(mirror, w, full_sector=True)

    assert w_sym.data.shape == (2, 2)
