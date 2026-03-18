import torch
import sympy as sy
from sympy import ImmutableDenseMatrix

from qten.bands import bandselect
from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice
from qten.linalg.tensors import Tensor
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import IndexSpace, brillouin_zone


def _space(name: str, n: int) -> HilbertSpace:
    return HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=((name, i),)) for i in range(n)
    )


def _band_tensor() -> tuple[Tensor, HilbertSpace]:
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 4)

    energies = torch.tensor(
        [
            [-3.0, -1.0, 2.0, 4.0],
            [-2.0, 1.0, 3.0, 5.0],
        ],
        dtype=torch.float64,
    )
    data = torch.diag_embed(energies).to(torch.complex128)
    tensor = Tensor(data=data, dims=(k_space, band_space, band_space))
    return tensor, band_space


def test_bandselect_supports_slice_criterion():
    tensor, band_space = _band_tensor()

    selected = bandselect(tensor, lowest_two=slice(0, 2))["lowest_two"]

    assert selected.dims[:2] == (tensor.dims[0], band_space)
    assert isinstance(selected.dims[2], IndexSpace)
    assert selected.dims[2].dim == 2
    expected = torch.eye(4, dtype=torch.complex128)[:, :2].expand(2, -1, -1)
    assert torch.allclose(selected.data, expected)


def test_bandselect_supports_index_tuple_criterion():
    tensor, band_space = _band_tensor()

    selected = bandselect(tensor, picked=(0, 2))["picked"]

    assert selected.dims[:2] == (tensor.dims[0], band_space)
    assert isinstance(selected.dims[2], IndexSpace)
    assert selected.dims[2].dim == 2
    expected = torch.eye(4, dtype=torch.complex128)[:, (0, 2)].expand(2, -1, -1)
    assert torch.allclose(selected.data, expected)


def test_bandselect_supports_energy_window_criterion_with_padding():
    tensor, band_space = _band_tensor()

    selected = bandselect(tensor, window=(-1.5, 2.5))["window"]

    assert selected.dims[:2] == (tensor.dims[0], band_space)
    assert isinstance(selected.dims[2], IndexSpace)
    assert selected.dims[2].dim == 2
    expected = torch.zeros((2, 4, 2), dtype=torch.complex128)
    expected[0, :, 0] = torch.eye(4, dtype=torch.complex128)[:, 1]
    expected[0, :, 1] = torch.eye(4, dtype=torch.complex128)[:, 2]
    expected[1, :, 0] = torch.eye(4, dtype=torch.complex128)[:, 1]
    assert torch.allclose(selected.data, expected)


def test_bandselect_supports_callable_criterion_with_padding():
    tensor, band_space = _band_tensor()

    selected = bandselect(tensor, negative=lambda e: e < 0)["negative"]

    assert selected.dims[:2] == (tensor.dims[0], band_space)
    assert isinstance(selected.dims[2], IndexSpace)
    assert selected.dims[2].dim == 2
    expected = torch.zeros((2, 4, 2), dtype=torch.complex128)
    expected[0, :, 0] = torch.eye(4, dtype=torch.complex128)[:, 0]
    expected[0, :, 1] = torch.eye(4, dtype=torch.complex128)[:, 1]
    expected[1, :, 0] = torch.eye(4, dtype=torch.complex128)[:, 0]
    assert torch.allclose(selected.data, expected)
