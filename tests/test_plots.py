from dataclasses import dataclass

import numpy as np
import sympy as sy
import torch

import plotly.graph_objects as go

from pyhilbert.hilbert_space import U1Basis, hilbert
from pyhilbert.spatials import Lattice
from pyhilbert.state_space import brillouin_zone
from pyhilbert.tensors import Tensor


@dataclass(frozen=True)
class Orb:
    name: str


def _basis_state(name: str) -> U1Basis:
    return U1Basis(u1=sy.Integer(1), rep=(Orb(name),))


def _space(size: int, prefix: str):
    return hilbert(_basis_state(f"{prefix}{i}") for i in range(size))


def create_dummy_tensor(data_like):
    """Create a Tensor with simple HilbertSpace dims for tests."""
    if isinstance(data_like, np.ndarray):
        data = torch.from_numpy(data_like)
    else:
        data = data_like

    dims = tuple(_space(dim, f"d{axis}_") for axis, dim in enumerate(data.shape))

    return Tensor(data=data, dims=dims)


def test_plot_heatmap_complex_matrix_returns_two_traces():
    mat = np.array(
        [
            [1 + 2j, 2 - 1j, 0 + 1j],
            [-1 + 0j, 0 + 0j, 3 - 2j],
            [4 + 1j, -2 + 2j, 1 - 3j],
        ],
        dtype=np.complex128,
    )
    tensor_obj = create_dummy_tensor(mat)

    fig = tensor_obj.plot("heatmap", title="Complex Heatmap", show=False)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Complex Heatmap"


def test_plot_heatmap_high_rank_requires_and_accepts_fixed_indices():
    # Shape (2, 2, 2): choosing axes=(1, 2) requires one fixed index.
    data = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    tensor_obj = create_dummy_tensor(data)

    fig = tensor_obj.plot(
        "heatmap",
        title="Rank-3 Slice",
        show=False,
        axes=(1, 2),
        fixed_indices=(0,),
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_plot_spectrum_hermitian_and_nonhermitian():
    hermitian = np.array(
        [[2.0, 1.0 - 1.0j, 0.0], [1.0 + 1.0j, 3.0, -2.0j], [0.0, 2.0j, 1.0]],
        dtype=np.complex128,
    )
    nonhermitian = np.array([[0.0, 1.0], [-2.0, 0.5]], dtype=np.float64)

    h_tensor = create_dummy_tensor(hermitian)
    nh_tensor = create_dummy_tensor(nonhermitian)

    h_fig = h_tensor.plot("spectrum", title="Hermitian Spectrum", show=False)
    nh_fig = nh_tensor.plot("spectrum", title="Non-Hermitian Spectrum", show=False)

    assert isinstance(h_fig, go.Figure)
    assert isinstance(nh_fig, go.Figure)
    assert len(h_fig.data) == 1
    assert len(nh_fig.data) == 1
    assert h_fig.layout.title.text == "Hermitian Spectrum"
    assert nh_fig.layout.title.text == "Non-Hermitian Spectrum"


def test_plot_structure_2d_and_3d():
    a = sy.Symbol("a")

    basis_2d = sy.ImmutableDenseMatrix([[a, 0], [0, a]])
    lattice_2d = Lattice(basis=basis_2d, shape=(3, 3))
    fig_2d = lattice_2d.plot("structure", subs={a: 1.5}, show=False)

    basis_3d = sy.ImmutableDenseMatrix([[a, 0, 0], [0, a, 0], [0, 0, a]])
    lattice_3d = Lattice(basis=basis_3d, shape=(2, 2, 2))
    fig_3d = lattice_3d.plot("structure", subs={a: 1.0}, show=False)

    assert isinstance(fig_2d, go.Figure)
    assert isinstance(fig_3d, go.Figure)
    assert len(fig_2d.data) >= 1
    assert len(fig_3d.data) >= 1


def test_bandstructure_plot():
    a = sy.Symbol("a")
    basis = sy.ImmutableDenseMatrix([[a, 0], [0, a]])
    lat = Lattice(basis=basis, shape=(4, 4))
    k_space = brillouin_zone(lat.dual)
    bloch_space = _space(2, "orb_")

    hk_data = torch.zeros((k_space.dim, 2, 2), dtype=torch.complex128)
    for i, k in enumerate(k_space.elements()):
        kvec = np.array(k.rep.subs({a: 1.0})).astype(float).flatten()
        knorm = float(np.linalg.norm(kvec))
        e0 = -2.0 * np.cos(knorm)
        e1 = 1.0 + 2.0 * np.cos(knorm)
        v = 0.1 + 0.2 * (i / max(1, k_space.dim - 1))
        hk_data[i] = torch.tensor([[e0, v], [v, e1]], dtype=torch.complex128)

    h_k = Tensor(data=hk_data, dims=(k_space, bloch_space, bloch_space))

    fig = h_k.plot(
        "bandstructure",
        backend="plotly",
        title="Test Bandstructure",
        show=False,
        subs={a: 1.0},
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert fig.layout.title.text == "Test Bandstructure"
