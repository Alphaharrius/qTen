from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import sympy as sy
import torch

from qten.symbolics.hilbert_space import U1Basis, HilbertSpace
from qten.geometries.spatials import Lattice, Offset
from qten.geometries.boundary import PeriodicBoundary
from qten.symbolics.state_space import brillouin_zone
from qten.linalg.tensors import Tensor
from qten.geometries.fourier import fourier_transform
from qten.plottings import Plottable
from qten_plots.plottables import PointCloud


@dataclass(frozen=True)
class Orb:
    name: str


def _basis_state(name: str) -> U1Basis:
    return U1Basis(coef=sy.Integer(1), base=(Orb(name),))


def _space(size: int, prefix: str):
    return HilbertSpace.new(_basis_state(f"{prefix}{i}") for i in range(size))


def create_dummy_tensor(data_like):
    """Create a Tensor with simple HilbertSpace dims for tests."""
    if isinstance(data_like, np.ndarray):
        data = torch.from_numpy(data_like)
    else:
        data = data_like

    dims = tuple(_space(dim, f"d{axis}_") for axis, dim in enumerate(data.shape))

    return Tensor(data=data, dims=dims)


def test_plottable_loads_plot_extensions_from_entry_points(monkeypatch):
    old_registry = dict(Plottable._registry)
    old_loaded = Plottable._backends_loaded
    old_errors = list(Plottable._backend_load_errors)

    Plottable._registry.clear()
    Plottable._backends_loaded = False
    Plottable._backend_load_errors.clear()

    loaded = []

    class FakeEntryPoint:
        name = "plotly"

        def load(self):
            loaded.append("plotly")

    entry_points = [FakeEntryPoint()]

    monkeypatch.setattr(
        "qten.plottings._plottings.metadata.entry_points",
        lambda **kwargs: entry_points
        if kwargs.get("group") == "qten.plottings"
        else [],
    )
    try:
        Plottable._ensure_backends_loaded()

        assert loaded == ["plotly"]
        assert Plottable._backends_loaded is True
    finally:
        Plottable._registry = old_registry
        Plottable._backends_loaded = old_loaded
        Plottable._backend_load_errors = old_errors


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
    basis_2d = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice_2d = Lattice(
        basis=basis_2d,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(3, 3)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )
    fig_2d = lattice_2d.plot("structure", show=False)

    basis_3d = sy.ImmutableDenseMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lattice_3d = Lattice(
        basis=basis_3d,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2, 2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0, 0])},
    )
    fig_3d = lattice_3d.plot("structure", show=False)

    assert isinstance(fig_2d, go.Figure)
    assert isinstance(fig_3d, go.Figure)
    assert len(fig_2d.data) >= 1
    assert len(fig_3d.data) >= 1


def test_plot_structure_accepts_pointcloud_highlights():
    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(3, 3)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )
    highlights = [
        PointCloud.of(
            [lattice.at(cell_offset=(0, 0)), lattice.at(cell_offset=(1, 1))],
            color="#ff0000",
        )
    ]

    fig = lattice.plot("structure", show=False, highlights=highlights)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_pointcloud_scatter_plot():
    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )
    cloud = PointCloud.of(
        [lattice.at(cell_offset=(0, 0)), lattice.at(cell_offset=(1, 0))],
        color="#00aa88",
    )

    fig = cloud.plot("scatter", show=False)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_bandstructure_plot():
    # 1. Define Lattice (2D Square)
    # Basis: [[a, 0], [0, a]]
    basis = sy.ImmutableDenseMatrix([[1, 0.0], [0.0, 1]])
    # Small shape for test speed
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )

    # 2. Define Unit Cell (Bloch Space)
    # Single s-orbital at origin (0,0)
    r_0 = Offset(rep=sy.ImmutableDenseMatrix([[0.0], [0.0]]), space=lat.affine)
    basis_s = U1Basis.new(r_0, "s")
    bloch_space = HilbertSpace.new([basis_s])

    # 3. Define Region Space (Real Space Neighbors)
    neighbor_offsets = [
        (0, 0),  # Center
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]

    region_modes = []
    offset_to_idx = {}

    for i, (dx, dy) in enumerate(neighbor_offsets):
        r_vec = sy.ImmutableDenseMatrix([[dx], [dy]])
        r_off = Offset(rep=r_vec, space=lat.affine)
        # Create a mode at this position based on the unit cell mode
        m = basis_s.replace(r_off)
        region_modes.append(m)
        offset_to_idx[(dx, dy)] = i

    region_space = HilbertSpace.new(region_modes)

    # 4. Construct Hamiltonian H_real
    t_n = -1.0 + 0j

    h_data = torch.zeros((region_space.dim, region_space.dim), dtype=torch.complex128)

    origin_idx = offset_to_idx[(0, 0)]

    nn_coords = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dx, dy in nn_coords:
        idx = offset_to_idx[(dx, dy)]
        h_data[origin_idx, idx] = t_n
        h_data[idx, origin_idx] = np.conjugate(t_n)

    h_real = Tensor(data=h_data, dims=(region_space, region_space))

    # 5. Define Momentum Space (Grid)
    k_space = brillouin_zone(lat.dual)

    # 6. Compute H(k)
    F = fourier_transform(k_space, bloch_space, region_space)
    F_dag = F.h(1, 2)
    h_k = F @ h_real @ F_dag

    # 7. Visualization
    fig = h_k.plot(
        "bandstructure",
        backend="plotly",
        title="Test Bandstructure",
        show=False,
    )

    assert isinstance(fig, go.Figure)
    # Check if we have traces (surfaces for 2D or lines for 1D/path)
    # Since we used brillouin_zone on a 2D lattice, it should produce a 2D grid plot (Surface)
    assert len(fig.data) >= 1
    # Check title
    assert fig.layout.title.text == "Test Bandstructure"
