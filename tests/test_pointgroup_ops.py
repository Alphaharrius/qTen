import sympy as sy
import pytest
import torch
from sympy import ImmutableDenseMatrix

from qten.geometries.spatials import AffineSpace, Offset
from qten.linalg.tensors import Tensor
from qten.pointgroups.abelian import (
    AbelianBasis,
    AbelianGroup,
    AbelianOpr,
)
from qten.pointgroups.ops import (
    abelian_column_symmetrize,
    joint_abelian_basis,
    joint_abelian_column_symmetrize,
)
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.ops import hilbert_opr_repr
from qten.symbolics.state_space import IndexSpace
from qten.utils.devices import Device


def _has_gpu() -> bool:
    try:
        Device("gpu").torch_device()
        return True
    except RuntimeError:
        return False


def _has_complex_gpu() -> bool:
    try:
        device = Device("gpu").torch_device()
        torch.zeros(1, dtype=torch.complex128, device=device)
        return True
    except (RuntimeError, TypeError, NotImplementedError):
        return False


HAS_GPU = _has_gpu()
HAS_COMPLEX_GPU = _has_complex_gpu()


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


def test_joint_abelian_column_symmetrize_projects_diagonal_mirrors():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror_45 = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[0, 1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )
    mirror_135 = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[0, -1], [-1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    points = [
        Offset(rep=ImmutableDenseMatrix([1, 0]), space=space),
        Offset(rep=ImmutableDenseMatrix([0, 1]), space=space),
        Offset(rep=ImmutableDenseMatrix([-1, 0]), space=space),
        Offset(rep=ImmutableDenseMatrix([0, -1]), space=space),
    ]
    h = HilbertSpace.new([_state(point) for point in points])

    w = Tensor(
        data=torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.complex128),
        dims=(h, IndexSpace.linear(1)),
    )
    w_sym = joint_abelian_column_symmetrize(
        [mirror_45, mirror_135], w, full_sector=True
    )

    assert w_sym.data.shape == (4, 4)
    labels = list(w_sym.dims[1].elements())
    assert all(
        isinstance(label.irrep_of(AbelianBasis), AbelianBasis) for label in labels
    )

    for opr in (mirror_45, mirror_135):
        g_full = hilbert_opr_repr(opr, h)
        projected_repr = w_sym.h(-2, -1) @ g_full @ w_sym
        assert torch.allclose(
            projected_repr.data,
            torch.diag(torch.diagonal(projected_repr.data)),
            atol=1e-10,
        )
    assert {
        tuple(
            sy.simplify(opr(label.irrep_of(AbelianBasis)).coef)
            for opr in (mirror_45, mirror_135)
        )
        for label in labels
    } == {
        (sy.Integer(1), sy.Integer(1)),
        (sy.Integer(1), sy.Integer(-1)),
        (sy.Integer(-1), sy.Integer(1)),
        (sy.Integer(-1), sy.Integer(-1)),
    }


def test_joint_abelian_basis_returns_common_diagonal_mirror_eigenfunctions():
    x, y = sy.symbols("x y")
    mirror_45 = AbelianGroup(
        irrep=ImmutableDenseMatrix([[0, 1], [1, 0]]),
        axes=(x, y),
    )
    mirror_135 = AbelianGroup(
        irrep=ImmutableDenseMatrix([[0, -1], [-1, 0]]),
        axes=(x, y),
    )

    common = joint_abelian_basis([mirror_45, mirror_135], order=1)

    assert set(common) == {
        (sy.Integer(1), sy.Integer(-1)),
        (sy.Integer(-1), sy.Integer(1)),
    }
    assert {
        sy.expand(bases[0].expr) for bases in common.values() if len(bases) == 1
    } == {x + y, x - y}


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(
                not HAS_COMPLEX_GPU,
                reason="requires GPU backend with complex tensor support",
            ),
        ),
    ],
)
def test_abelian_column_symmetrize_preserves_device_for_empty_output(
    device_name: str,
):
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
        data=torch.zeros((2, 1), dtype=torch.complex128),
        dims=(h, IndexSpace.linear(1)),
    ).to_device(Device(device_name))
    w_sym = abelian_column_symmetrize(mirror, w, full_sector=True)

    assert w_sym.device == w.device
    assert w_sym.data.shape == (2, 0)
