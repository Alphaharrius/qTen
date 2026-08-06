import sympy as sy
import pytest
import torch
from sympy import ImmutableDenseMatrix

from qten.geometries.spatials import AffineSpace, Lattice, Offset
from qten.linalg.tensors import Tensor
from qten.phys import Spin
from qten.pointgroups import (
    FinitePointGroup,
    PointGroupBasis,
    PointGroupElement,
    PointGroupOpr,
    SpinorIrrepSector,
    SymmetryDegeneracy,
    pointgroup,
)
from qten.pointgroups.ops import (
    point_group_column_symmetrize,
    get_direct_transform,
    _hilbert_opr_repr,
    joint_point_group_basis,
    joint_point_group_column_symmetrize,
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
) -> PointGroupOpr:
    opr = PointGroupOpr(g=PointGroupElement(irrep=irrep, axes=axes))
    object.__setattr__(opr, "offset", offset)
    return opr


def test_point_group_column_symmetrize_projects_indexspace_columns_to_sector_labels():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = PointGroupBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = PointGroupBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    h = HilbertSpace.new([_state(fx), _state(fy)])

    w = Tensor(
        data=torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        / torch.sqrt(torch.tensor(2.0, dtype=torch.float64)),
        dims=(h, IndexSpace.linear(1)),
    )
    w_sym = point_group_column_symmetrize(mirror, w, full_sector=True)

    assert isinstance(w_sym.dims[1], HilbertSpace)
    labels = list(w_sym.dims[1].elements())
    sector_phases = torch.tensor(
        [complex(sy.N(label.irrep_of(PointGroupBasis).irrep)) for label in labels],
        dtype=torch.complex128,
    )

    g_full = _hilbert_opr_repr(mirror, h)
    expected = torch.diag(sector_phases)
    assert torch.allclose((w_sym.h(-2, -1) @ g_full @ w_sym).data, expected)
    assert set(sector_phases.tolist()) == {1.0 + 0.0j, -1.0 + 0.0j}


def test_point_group_basis_representation_respects_target_gauge():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )
    fx = PointGroupBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    gauged_space = HilbertSpace.new([_state(fx, irrep=sy.I)])

    representation = _hilbert_opr_repr(mirror, gauged_space)

    assert torch.allclose(
        representation.data,
        torch.tensor([[-1]], dtype=torch.complex128),
        rtol=0,
        atol=1e-12,
    )
    assert torch.allclose(
        representation.data @ representation.data,
        torch.eye(1, dtype=torch.complex128),
        rtol=0,
        atol=1e-12,
    )


def test_point_group_column_symmetrize_defaults_to_one_sector_per_input_column():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = PointGroupBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = PointGroupBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    h = HilbertSpace.new([_state(fx), _state(fy)])

    w = Tensor(
        data=torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        / torch.sqrt(torch.tensor(2.0, dtype=torch.float64)),
        dims=(h, IndexSpace.linear(1)),
    )
    w_sym = point_group_column_symmetrize(mirror, w)

    assert w_sym.data.shape == (2, 1)
    label = next(iter(w_sym.dims[1].elements()))
    phase = complex(sy.N(label.irrep_of(PointGroupBasis).irrep))
    assert phase in {1.0 + 0.0j, -1.0 + 0.0j}


def test_point_group_column_symmetrize_appends_basis_for_hilbertspace():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = PointGroupBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = PointGroupBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    row_space = HilbertSpace.new([_state(fx), _state(fy)])
    seed_space = HilbertSpace.new([_state("seed_a"), _state("seed_b")])

    w = Tensor(
        data=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.complex128),
        dims=(row_space, seed_space),
    )
    w_sym = point_group_column_symmetrize(mirror, w)

    labels = list(w_sym.dims[1].elements())
    assert len(labels) == 2
    assert {label.irrep_of(str) for label in labels} == {"seed_a", "seed_b"}
    assert torch.allclose(w_sym.data[:, 0], w_sym.data[:, 1])
    assert all(
        complex(sy.N(label.irrep_of(PointGroupBasis).irrep)) == 1.0 + 0.0j
        for label in labels
    )


def test_point_group_column_symmetrize_adds_degeneracy_tag_for_duplicate_labels():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = PointGroupBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = PointGroupBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    row_space = HilbertSpace.new([_state(fx), _state(fy)])

    w = Tensor(
        data=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.complex128),
        dims=(row_space, IndexSpace.linear(2)),
    )
    w_sym = point_group_column_symmetrize(mirror, w)

    labels = list(w_sym.dims[1].elements())
    assert len(labels) == 2
    assert {label.irrep_of(SymmetryDegeneracy).index for label in labels} == {0, 1}


def test_degeneracy_tag_preserves_existing_integer_metadata():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )
    fx = PointGroupBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = PointGroupBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    row_space = HilbertSpace.new([_state(fx), _state(fy)])
    seed_space = HilbertSpace.new([U1Basis.new(7, fx), U1Basis.new(7, fy)])
    w = Tensor(
        data=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.complex128),
        dims=(row_space, seed_space),
    )

    projected = point_group_column_symmetrize(mirror, w)
    labels = projected.dims[1].elements()

    assert all(label.irrep_of(int) == 7 for label in labels)
    assert {label.irrep_of(SymmetryDegeneracy).index for label in labels} == {0, 1}


def test_point_group_column_symmetrize_full_sector_expands_mixed_column():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = PointGroupBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = PointGroupBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    h = HilbertSpace.new([_state(fx), _state(fy)])

    w = Tensor(
        data=torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        / torch.sqrt(torch.tensor(2.0, dtype=torch.float64)),
        dims=(h, IndexSpace.linear(1)),
    )
    w_sym = point_group_column_symmetrize(mirror, w, full_sector=True)

    assert w_sym.data.shape == (2, 2)


def test_point_group_column_symmetrize_accepts_point_group_basis_rows():
    x, y = sy.symbols("x y")
    c4v = pointgroup("C4v-xy")

    assert isinstance(c4v, FinitePointGroup)
    mirror = PointGroupOpr(c4v.generators[1])
    e_basis = {basis.expr: basis for basis in c4v.irrep_basis(order=1, irrep="E")}
    h = HilbertSpace.new([_state(e_basis[x]), _state(e_basis[y])])

    w = Tensor(
        data=torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        / torch.sqrt(torch.tensor(2.0, dtype=torch.float64)),
        dims=(h, IndexSpace.linear(1)),
    )
    w_sym = point_group_column_symmetrize(mirror, w, full_sector=True)

    labels = list(w_sym.dims[1].elements())
    sector_phases = {
        sy.simplify(label.irrep_of(PointGroupBasis).irrep) for label in labels
    }

    assert w_sym.data.shape == (2, 2)
    assert torch.allclose(torch.abs(w_sym.data), torch.eye(2, dtype=torch.float64))
    assert sector_phases == {sy.Integer(1), sy.Integer(-1)}


def test_finite_point_group_projection_promotes_integer_seed_data():
    x, y = sy.symbols("x y")
    c4v = pointgroup("C4v-xy")
    e_basis = {basis.expr: basis for basis in c4v.irrep_basis(order=1, irrep="E")}
    space = HilbertSpace.new([_state(e_basis[x]), _state(e_basis[y])])
    seed = Tensor(
        data=torch.tensor([[1], [0]], dtype=torch.int64),
        dims=(space, IndexSpace.linear(1)),
    )

    projected = point_group_column_symmetrize(c4v, seed, full_sector=True)

    assert projected.data.is_complex()
    assert projected.data.shape == (2, 1)
    assert torch.allclose(
        torch.linalg.vector_norm(projected.data, dim=0),
        torch.ones(1, dtype=projected.data.real.dtype),
    )


def test_finite_spinful_projection_resolves_tetrahedral_orbital_multiplets():
    td = pointgroup("-43m")
    affine = AffineSpace(basis=ImmutableDenseMatrix.eye(3))
    center = Offset(ImmutableDenseMatrix.zeros(3, 1), affine)
    vertices = (
        Offset(ImmutableDenseMatrix([1, 1, 1]), affine),
        Offset(ImmutableDenseMatrix([1, -1, -1]), affine),
        Offset(ImmutableDenseMatrix([-1, 1, -1]), affine),
        Offset(ImmutableDenseMatrix([-1, -1, 1]), affine),
    )
    space = HilbertSpace.new(
        U1Basis.new(vertex, spin)
        for vertex in vertices
        for spin in (Spin.up, Spin.down)
    )
    seed = Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )

    projected = point_group_column_symmetrize(
        td,
        seed,
        full_sector=True,
        fixpoint=center,
    )

    assert projected.data.shape == (space.dim, space.dim)
    assert torch.allclose(
        projected.data.conj().T @ projected.data,
        torch.eye(space.dim, dtype=torch.complex128),
        rtol=0,
        atol=1e-12,
    )
    sectors = [
        label.irrep_of(SpinorIrrepSector) for label in projected.dims[1].elements()
    ]
    assert {sector.source for sector in sectors} == {"spgrep"}
    counts: dict[str, int] = {}
    dimensions: dict[str, int] = {}
    for sector in sectors:
        counts[sector.irrep] = counts.get(sector.irrep, 0) + 1
        dimensions[sector.irrep] = sector.dim
    assert all(count % dimensions[irrep] == 0 for irrep, count in counts.items())


def test_joint_point_group_column_symmetrize_projects_diagonal_mirrors():
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
    w_sym = joint_point_group_column_symmetrize(
        [mirror_45, mirror_135], w, full_sector=True
    )

    assert w_sym.data.shape == (4, 4)
    labels = list(w_sym.dims[1].elements())
    assert all(
        isinstance(label.irrep_of(PointGroupBasis), PointGroupBasis) for label in labels
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
        tuple(sy.simplify(phase) for phase in label.irrep_of(PointGroupBasis).irrep)
        for label in labels
    } == {
        (sy.Integer(1), sy.Integer(1)),
        (sy.Integer(1), sy.Integer(-1)),
        (sy.Integer(-1), sy.Integer(1)),
        (sy.Integer(-1), sy.Integer(-1)),
    }


def test_joint_projection_rejects_noncommuting_affine_actions():
    x = sy.Symbol("x")
    lattice = Lattice(basis=ImmutableDenseMatrix([[1]]), shape=(4,))
    space = HilbertSpace.new(U1Basis.new(site) for site in lattice.cartes())
    reflection = PointGroupElement(
        irrep=ImmutableDenseMatrix([[-1]]),
        axes=(x,),
    )
    about_zero = PointGroupOpr(reflection).fixpoint_at(
        Offset(ImmutableDenseMatrix([0]), lattice),
        rebase=True,
    )
    about_half = PointGroupOpr(reflection).fixpoint_at(
        Offset(ImmutableDenseMatrix([sy.Rational(1, 2)]), lattice),
        rebase=True,
    )
    seed = Tensor(
        data=torch.tensor([[1.0], [2.0], [3.0], [5.0]], dtype=torch.complex128),
        dims=(space, IndexSpace.linear(1)),
    )

    with pytest.raises(ValueError, match="commuting Hilbert-space representations"):
        joint_point_group_column_symmetrize(
            (about_zero, about_half),
            seed,
            full_sector=True,
        )


def test_joint_point_group_basis_returns_common_diagonal_mirror_eigenfunctions():
    x, y = sy.symbols("x y")
    mirror_45 = PointGroupElement(
        irrep=ImmutableDenseMatrix([[0, 1], [1, 0]]),
        axes=(x, y),
    )
    mirror_135 = PointGroupElement(
        irrep=ImmutableDenseMatrix([[0, -1], [-1, 0]]),
        axes=(x, y),
    )

    common = joint_point_group_basis([mirror_45, mirror_135], order=1)

    assert set(common) == {
        (sy.Integer(1), sy.Integer(-1)),
        (sy.Integer(-1), sy.Integer(1)),
    }
    assert {
        sy.expand(bases[0].expr) for bases in common.values() if len(bases) == 1
    } == {x + y, x - y}


def test_get_direct_transform_builds_transformed_output_space_for_affine_offsets():
    x = sy.symbols("x")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(1))
    shift = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[1]]),
        axes=(x,),
        offset=Offset(rep=ImmutableDenseMatrix([1]), space=space),
    )

    src = HilbertSpace.new([_state(Offset(rep=ImmutableDenseMatrix([0]), space=space))])
    transform = get_direct_transform(shift, src)

    assert transform.dims[0] == src
    assert isinstance(transform.dims[1], HilbertSpace)
    assert transform.dims[1].elements() == (
        _state(Offset(rep=ImmutableDenseMatrix([1]), space=space)),
    )
    assert torch.equal(
        transform.data,
        torch.tensor([[1.0 + 0.0j]], dtype=torch.complex128),
    )


def test_get_direct_transform_rotates_point_group_basis_directly_without_phase_in_data():
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    c4 = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[0, -1], [1, 0]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = PointGroupBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = PointGroupBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    src = HilbertSpace.new([_state(fx), _state(fy)])

    transform = get_direct_transform(c4, src)
    out_basis = transform.dims[1].elements()

    assert out_basis[0].coef == sy.Integer(1)
    assert out_basis[0].irrep_of(PointGroupBasis).expr == y
    assert out_basis[1].coef == sy.Integer(1)
    assert out_basis[1].irrep_of(PointGroupBasis).expr == -x
    assert torch.equal(
        transform.data,
        torch.tensor(
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            dtype=torch.complex128,
        ),
    )


def test_get_direct_transform_rotates_point_group_basis_for_band_space():
    x, y = sy.symbols("x y")
    c4v = pointgroup("C4v-xy")

    assert isinstance(c4v, FinitePointGroup)
    rotation = PointGroupOpr(c4v.generators[0])
    e_basis = {basis.expr: basis for basis in c4v.irrep_basis(order=1, irrep="E")}
    src = HilbertSpace.new([_state(e_basis[x]), _state(e_basis[y])])

    transform = get_direct_transform(rotation, src)
    out_basis = transform.dims[1].elements()

    assert isinstance(out_basis[0].irrep_of(PointGroupBasis), PointGroupBasis)
    assert sy.simplify(out_basis[0].irrep_of(PointGroupBasis).expr - y) == 0
    assert sy.simplify(out_basis[1].irrep_of(PointGroupBasis).expr + x) == 0
    assert torch.equal(
        transform.data,
        torch.tensor(
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            dtype=torch.complex128,
        ),
    )


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
def test_point_group_column_symmetrize_preserves_device_for_empty_output(
    device_name: str,
):
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = PointGroupBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = PointGroupBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))
    h = HilbertSpace.new([_state(fx), _state(fy)])

    w = Tensor(
        data=torch.zeros((2, 1), dtype=torch.complex128),
        dims=(h, IndexSpace.linear(1)),
    ).to_device(Device(device_name))
    w_sym = point_group_column_symmetrize(mirror, w, full_sector=True)

    assert w_sym.device == w.device
    assert w_sym.data.shape == (2, 0)
