import pytest
import torch
import sympy as sy
from sympy import ImmutableDenseMatrix

from qten.utils.devices import Device

from qten.linalg.tensors import (
    zeros,
    ones,
    eye,
    Tensor,
    kernel_tensor,
    mapping_matrix,
    one_hot,
)
from qten.symbolics import IndexSpace, HilbertSpace, U1Basis
from qten.geometries.spatials import Lattice, PeriodicBoundary, AffineSpace, Offset
from qten.phys._ff_observables import FFObservable
from qten.linalg.decompose import svd, eigh
from qten.pointgroups.abelian import AbelianBasis, AbelianGroup, AbelianOpr
from qten.pointgroups.ops import abelian_column_symmetrize
from qten.geometries.fourier import fourier_transform
from qten.symbolics.ops import hilbert_opr_repr
from qten.symbolics.hilbert_space import FuncOpr
from qten.symbolics.state_space import brillouin_zone, IndexSpace
from qten.bands import bandfold
from qten.geometries.basis_transform import BasisTransform
from qten.bands import bandfillings


has_gpu = torch.cuda.is_available() or (
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)

pytestmark = pytest.mark.skipif(not has_gpu, reason="No GPU (CUDA or MPS) available")


@pytest.fixture
def device():
    d = Device.new("gpu")
    td = d.torch_device()
    if td.type == "cuda":
        return Device("gpu", td.index)
    return d


def test_tensor_factories(device):
    s1 = IndexSpace.linear(3)
    s2 = IndexSpace.linear(4)

    t_zeros = zeros((s1, s2), device=device)
    assert t_zeros.device == device

    t_ones = ones((s1, s2), device=device)
    assert t_ones.device == device

    t_eye = eye((s1, s1), device=device)
    assert t_eye.device == device

    t_scalar = Tensor.scalar(4.2, device=device)
    assert t_scalar.device == device


def test_kernel_and_mapping(device):
    s1 = IndexSpace.linear(2)
    s2 = IndexSpace.linear(2)

    def ker(i, j):
        return i + j

    t_ker = kernel_tensor(ker, (s1, s2), device=device)
    assert t_ker.device == device

    mapping = {0: 1, 1: 0}
    t_map = mapping_matrix(s1, s2, mapping, device=device)
    assert t_map.device == device


def test_tensor_math(device):
    s1 = IndexSpace.linear(3)

    t1 = ones((s1, s1), device=device)
    t2 = eye((s1, s1), device=device)

    # Matrix multiplication
    t3 = t1 @ t2
    assert t3.device == device

    # Addition
    t4 = t1 + t2
    assert t4.device == device

    # to_device functionality
    t_cpu = ones((s1, s1))
    t_gpu = t_cpu.to_device(device)
    assert t_gpu.device == device


def test_decompositions(device):
    s1 = IndexSpace.linear(3)

    # Create symmetric matrix on GPU
    data = torch.randn(3, 3, device=device.torch_device(), dtype=torch.float64)
    data = data + data.T
    t = Tensor(data=data, dims=(s1, s1))

    # SVD
    u, s, vh = svd(t)
    assert u.device == device
    assert s.device == device
    assert vh.device == device

    # EIGH
    vals, vecs = eigh(t)
    assert vals.device == device
    assert vecs.device == device


def test_spatials_lattice(device):
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    coords = lattice.cartes(torch.Tensor, device=device)
    assert coords.device.type == device.torch_device().type


def test_free_fermions(device):
    ff = FFObservable()
    b1 = U1Basis(coef=sy.Integer(1), base=("A",))
    b2 = U1Basis(coef=sy.Integer(1), base=("B",))
    ff.add_bond(sy.Integer(1), b1, b2)

    t_ten = ff.to_tensor(device=device)
    assert t_ten.device == device


def _opr_with_offset(
    irrep: ImmutableDenseMatrix,
    axes: tuple[sy.Symbol, ...],
    offset: Offset,
) -> AbelianOpr:
    opr = AbelianOpr(g=AbelianGroup(irrep=irrep, axes=axes))
    object.__setattr__(opr, "offset", offset)
    return opr


def test_pointgroup_ops(device):
    x, y = sy.symbols("x y")
    space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))

    mirror = _opr_with_offset(
        irrep=ImmutableDenseMatrix([[-1, 0], [0, 1]]),
        axes=(x, y),
        offset=Offset(rep=ImmutableDenseMatrix([0, 0]), space=space),
    )

    fx = AbelianBasis(expr=x, axes=(x, y), order=1, rep=ImmutableDenseMatrix([1, 0]))
    fy = AbelianBasis(expr=y, axes=(x, y), order=1, rep=ImmutableDenseMatrix([0, 1]))

    def _state(tag) -> U1Basis:
        return U1Basis(coef=sy.Integer(1), base=(tag,))

    row_space = HilbertSpace.new([_state(fx), _state(fy)])
    seed_space = HilbertSpace.new([_state("seed_a"), _state("seed_b")])

    w = Tensor(
        data=torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.complex128,
            device=device.torch_device(),
        ),
        dims=(row_space, seed_space),
    )

    # This invokes projected spaces which previously had hardcoded CPU allocations
    w_sym = abelian_column_symmetrize(mirror, w)

    assert w_sym.device == device


def test_cross_gram_device(device):
    r0 = Offset(
        rep=ImmutableDenseMatrix([0]),
        space=AffineSpace(basis=ImmutableDenseMatrix.eye(1)),
    )
    b0 = U1Basis(coef=sy.Integer(1), base=(r0, "s"))
    space1 = HilbertSpace.new([b0])
    space2 = HilbertSpace.new([b0])

    tensor = space1.cross_gram(space2, device=device)
    assert tensor.device == device


def test_hilbert_opr_repr_device(device):
    r0 = Offset(
        rep=ImmutableDenseMatrix([0]),
        space=AffineSpace(basis=ImmutableDenseMatrix.eye(1)),
    )
    b0 = U1Basis(coef=sy.Integer(1), base=(r0, "s"))
    space = HilbertSpace.new([b0])

    identity_opr = FuncOpr(Offset, lambda x: x)

    tensor = hilbert_opr_repr(identity_opr, space, device=device)
    assert tensor.device == device


def test_fourier_tensor_device(device):
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    recip = lat.dual

    k_space = brillouin_zone(recip)

    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat)

    m0 = U1Basis(coef=sy.Integer(1), base=(r0, "s"))
    m1 = U1Basis(coef=sy.Integer(1), base=(r1, "s"))
    region_space = HilbertSpace.new([m0, m1])

    b0 = U1Basis(coef=sy.Integer(1), base=(r0, "s"))
    bloch_space = HilbertSpace.new([b0])

    ft_tensor_gpu = fourier_transform(k_space, bloch_space, region_space, device=device)

    assert ft_tensor_gpu.device == device


def test_one_hot_device(device):
    # Create integer tensor on GPU
    idx_space = IndexSpace.linear(3)
    t = Tensor(
        data=torch.tensor([0, 1, 2], device=device.torch_device()), dims=(idx_space,)
    )

    # One-hot encode with new dimension
    out_dim = IndexSpace.linear(4)
    t_oh = one_hot(t, out_dim)

    assert t_oh.device == device


def test_bandfillings_device(device):
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    recip = lat.dual

    k_space = brillouin_zone(recip)

    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat)
    m0 = U1Basis(coef=sy.Integer(1), base=(r0, "s"))
    bloch_space = HilbertSpace.new([m0])

    H = ones((k_space, bloch_space, bloch_space), device=device)

    # Fill exactly half the band
    filled = bandfillings(H, 0.5)

    assert filled.device == device


def test_device_bounded_cpu(device):
    s1 = IndexSpace.linear(3)
    t_gpu = ones((s1, s1), device=device)
    t_cpu = t_gpu.cpu()

    assert t_gpu.device == device
    assert t_cpu.device.name == "cpu"

    t_gpu_back = t_cpu.gpu()
    assert t_gpu_back.device == device


def test_bandfold_device(device):
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    recip = lat.dual

    k_space = brillouin_zone(recip)

    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat)
    m0 = U1Basis(coef=sy.Integer(1), base=(r0, "s"))
    bloch_space = HilbertSpace.new([m0])

    # create a mock Hamiltonian tensor (K, B, B)
    # on the given device
    H = ones((k_space, bloch_space, bloch_space), device=device)

    # Scale by factor 2 to fold the band
    T = BasisTransform(M=ImmutableDenseMatrix([[2]]))

    # Run bandfold, it should operate completely on device
    H_folded = bandfold(T, H, opt="both")

    # Assert output device matches input device
    assert H_folded.device == device
