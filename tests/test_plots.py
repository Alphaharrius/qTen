from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import pytest
import sympy as sy
import torch

from qten.symbolics import interpolate_reciprocal_path
from qten.symbolics.hilbert_space import U1Basis, HilbertSpace
from qten.geometries.spatials import Lattice, Offset
from qten.geometries.boundary import PeriodicBoundary
from qten.symbolics.state_space import brillouin_zone
from qten.symbolics.state_space import IndexSpace
from qten.linalg.tensors import Tensor
from qten.geometries.fourier import fourier_transform
from qten.plottings import Plottable
from qten_plots.plottables import PointCloud
from qten_plots._utils import band_path_positions


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


def _make_square_lattice_band_tensor(shape: tuple[int, int]) -> Tensor:
    basis = sy.ImmutableDenseMatrix([[1, 0.0], [0.0, 1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(*shape)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )

    r_0 = Offset(rep=sy.ImmutableDenseMatrix([[0.0], [0.0]]), space=lat.affine)
    basis_s = U1Basis.new(r_0, "s")
    bloch_space = HilbertSpace.new([basis_s])

    neighbor_offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    region_modes = []
    offset_to_idx = {}
    for i, (dx, dy) in enumerate(neighbor_offsets):
        r_vec = sy.ImmutableDenseMatrix([[dx], [dy]])
        r_off = Offset(rep=r_vec, space=lat.affine)
        region_modes.append(basis_s.replace(r_off))
        offset_to_idx[(dx, dy)] = i

    region_space = HilbertSpace.new(region_modes)
    h_data = torch.zeros((region_space.dim, region_space.dim), dtype=torch.complex128)

    origin_idx = offset_to_idx[(0, 0)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        idx = offset_to_idx[(dx, dy)]
        h_data[origin_idx, idx] = -1.0 + 0j
        h_data[idx, origin_idx] = -1.0 + 0j

    h_real = Tensor(data=h_data, dims=(region_space, region_space))
    k_space = brillouin_zone(lat.dual)
    F = fourier_transform(k_space, bloch_space, region_space)
    return F @ h_real @ F.h(1, 2)


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


def test_plot_column_scatter_uses_offset_positions_and_complex_encoding():
    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )

    row_space = HilbertSpace.new(
        [
            U1Basis.new(
                Offset(rep=sy.ImmutableDenseMatrix([[0], [0]]), space=lattice.affine)
            ),
            U1Basis.new(
                Offset(rep=sy.ImmutableDenseMatrix([[1], [0]]), space=lattice.affine)
            ),
            U1Basis.new(
                Offset(rep=sy.ImmutableDenseMatrix([[0], [1]]), space=lattice.affine)
            ),
        ]
    )
    col_space = IndexSpace.linear(2)
    tensor = Tensor(
        data=torch.tensor(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [2.0 + 0.0j, -1.0 + 0.0j],
                [0.5 - 0.5j, 1.0 - 1.0j],
            ],
            dtype=torch.complex128,
        ),
        dims=(row_space, col_space),
    )

    fig = tensor.plot(
        "column_scatter",
        title="Column Scatter",
        show=False,
        default_size=20.0,
        ncols=2,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Column Scatter"
    assert [annotation.text for annotation in fig.layout.annotations] == ["0", "1"]
    assert list(fig.data[0].x) == [0.0, 1.0, 0.0]
    assert list(fig.data[0].y) == [0.0, 0.0, 1.0]
    assert fig.data[0].name == "0"
    assert max(fig.data[0].marker.size) == 20.0
    assert all(str(color).startswith("rgb(") for color in fig.data[0].marker.color)


def test_plot_column_scatter_requires_offset_on_first_hilbert_space():
    row_space = _space(3, "row_")
    col_space = _space(2, "col_")
    tensor = Tensor(
        data=torch.ones((3, 2), dtype=torch.complex128),
        dims=(row_space, col_space),
    )

    with pytest.raises(ValueError, match="Offset"):
        tensor.plot("column_scatter", show=False)


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


def test_plot_structure_uses_highlight_coordinates_without_wrapping():
    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(3, 3)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )
    outside_cell = Offset(rep=sy.ImmutableDenseMatrix([4, 1]), space=lattice.affine)

    fig = lattice.plot(
        "structure",
        show=False,
        highlights=[PointCloud.of([outside_cell], color="#ff0000")],
    )

    assert isinstance(fig, go.Figure)
    highlight = fig.data[-1]
    assert list(highlight.x) == [4.0]
    assert list(highlight.y) == [1.0]


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


def test_pointcloud_scatter_uses_marker_opacity_size_and_border():
    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )
    cloud = PointCloud.of(
        [lattice.at(cell_offset=(0, 0)), lattice.at(cell_offset=(1, 0))],
        color="#00aa88",
        marker="square",
        opacity=0.4,
        size=17,
        border_color="#112233",
        border_width=2.5,
    )

    fig = cloud.plot("scatter", show=False)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].marker.symbol == "square"
    assert fig.data[0].marker.opacity == 0.4
    assert fig.data[0].marker.size == 17
    assert fig.data[0].marker.line.color == "#112233"
    assert fig.data[0].marker.line.width == 2.5


def test_pointcloud_scatter_uses_custom_name_for_legend():
    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )
    cloud = PointCloud.of(
        [lattice.at(cell_offset=(0, 0))],
        name="My Region",
    )

    fig = cloud.plot("scatter", show=False)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].name == "My Region"


def test_pointcloud_scatter_normalizes_shared_marker_aliases():
    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )
    cloud = PointCloud.of(
        [lattice.at(cell_offset=(0, 0))],
        marker="s",
    )

    fig = cloud.plot("scatter", show=False)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].marker.symbol == "square"


def test_pointcloud_rejects_unsupported_marker_name():
    basis = sy.ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0])},
    )

    with pytest.raises(ValueError, match="Unsupported PointCloud marker"):
        PointCloud.of([lattice.at(cell_offset=(0,))], marker="triangle")


def test_plot_structure_highlights_use_pointcloud_marker_opacity_size_and_border():
    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(3, 3)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )

    fig = lattice.plot(
        "structure",
        show=False,
        highlights=[
            PointCloud.of(
                [lattice.at(cell_offset=(0, 0))],
                color="#ff0000",
                marker="square",
                opacity=0.25,
                size=21,
                border_color="#223344",
                border_width=3,
            )
        ],
    )

    assert isinstance(fig, go.Figure)
    highlight = fig.data[-1]
    assert highlight.marker.symbol == "square"
    assert highlight.marker.opacity == 0.25
    assert highlight.marker.size == 21
    assert highlight.marker.line.color == "#223344"
    assert highlight.marker.line.width == 3


def test_plot_structure_highlights_use_pointcloud_name_for_legend():
    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(3, 3)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )

    fig = lattice.plot(
        "structure",
        show=False,
        highlights=[PointCloud.of([lattice.at(cell_offset=(0, 0))], name="Edge Sites")],
    )

    assert isinstance(fig, go.Figure)
    assert fig.data[-1].name == "Edge Sites"


def test_plot_structure_hover_can_use_lattice_coords():
    basis = sy.ImmutableDenseMatrix([[2, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 1)),
        unit_cell=OrderedDict(
            [
                ("a", sy.ImmutableDenseMatrix([0, 0])),
                ("b", sy.ImmutableDenseMatrix([sy.Rational(1, 2), 0])),
            ]
        ),
    )

    fig = lattice.plot("structure", show=False, use_lattice_coords=True)

    assert isinstance(fig, go.Figure)
    basis_b_trace = next(trace for trace in fig.data if trace.name == "Basis 1")
    assert basis_b_trace.hovertemplate == "%{text}<extra></extra>"
    assert "(1/2, 0)" == basis_b_trace.text[0]


def test_pointcloud_scatter_hover_can_use_lattice_coords():
    basis = sy.ImmutableDenseMatrix([[2, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )
    cloud = PointCloud.of(
        [
            Offset(
                rep=sy.ImmutableDenseMatrix([sy.Rational(1, 2), 0]),
                space=lattice.affine,
            )
        ]
    )

    fig = cloud.plot("scatter", show=False, use_lattice_coords=True)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].hovertemplate == "%{text}<extra></extra>"
    assert "(1/2, 0)" == fig.data[0].text[0]


def test_pointcloud_scatter_matplotlib_interprets_size_as_linear_extent():
    import matplotlib

    matplotlib.use("Agg")

    basis = sy.ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )
    cloud = PointCloud.of(
        [lattice.at(cell_offset=(0, 0))],
        size=5,
    )

    fig = cloud.plot("scatter", backend="matplotlib")
    ax = fig.axes[0]
    collection = ax.collections[0]

    assert collection.get_sizes().tolist() == [25]


def test_plot_column_scatter_hover_can_use_lattice_coords():
    basis = sy.ImmutableDenseMatrix([[2, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )

    row_space = HilbertSpace.new(
        [
            U1Basis.new(
                Offset(
                    rep=sy.ImmutableDenseMatrix([[sy.Rational(1, 2)], [0]]),
                    space=lattice.affine,
                )
            ),
            U1Basis.new(
                Offset(rep=sy.ImmutableDenseMatrix([[1], [0]]), space=lattice.affine)
            ),
        ]
    )
    col_space = IndexSpace.linear(1)
    tensor = Tensor(
        data=torch.tensor([[1.0 + 0.0j], [0.5 + 0.5j]], dtype=torch.complex128),
        dims=(row_space, col_space),
    )

    fig = tensor.plot("column_scatter", show=False, use_lattice_coords=True)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].hovertemplate == "%{text}<extra></extra>"
    assert "|value|=1" in fig.data[0].text[0]
    assert "(1/2, 0)" in fig.data[0].text[0]


def test_bandstructure_plot():
    h_k = _make_square_lattice_band_tensor((4, 4))

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


def test_bandstructure_plot_auto_falls_back_to_path_for_effectively_1d_k_mesh():
    h_k = _make_square_lattice_band_tensor((4, 1))

    fig = h_k.plot(
        "bandstructure",
        backend="plotly",
        title="Degenerate 2D Bandstructure",
        show=False,
    )

    assert len(fig.data) >= 1
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)


def test_bandstructure_surface_mode_rejects_effectively_1d_k_mesh():
    h_k = _make_square_lattice_band_tensor((4, 1))

    with pytest.raises(ValueError, match="requires two varying momentum directions"):
        h_k.plot(
            "bandstructure",
            backend="plotly",
            mode="surface",
            show=False,
        )


def test_band_path_positions_wraps_across_periodic_boundary():
    basis = sy.ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(16)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lat.dual)
    k_points = list(k_space.elements())
    reordered = [k_points[0], *k_points[:0:-1]]

    wrapped_space = type(k_space)(
        structure=OrderedDict((k, i) for i, k in enumerate(reordered))
    )
    k_cart = np.array([[float(k.rep[0, 0])] for k in reordered], dtype=float)

    x_vals = band_path_positions(wrapped_space, k_cart)

    step = float(lat.dual.basis[0, 0]) / 16.0
    expected = np.arange(len(reordered), dtype=float) * step
    assert np.allclose(x_vals, expected)


def _make_2d_lattice_and_spaces():
    """Build a 2D square lattice with one orbital, return (lattice, bloch_space, region_space, h_real)."""
    basis = sy.ImmutableDenseMatrix([[1, 0.0], [0.0, 1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
    )

    r_0 = Offset(rep=sy.ImmutableDenseMatrix([[0.0], [0.0]]), space=lat.affine)
    basis_s = U1Basis.new(r_0, "s")
    bloch_space = HilbertSpace.new([basis_s])

    neighbor_offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    region_modes = []
    offset_to_idx = {}
    for i, (dx, dy) in enumerate(neighbor_offsets):
        r_vec = sy.ImmutableDenseMatrix([[dx], [dy]])
        r_off = Offset(rep=r_vec, space=lat.affine)
        region_modes.append(basis_s.replace(r_off))
        offset_to_idx[(dx, dy)] = i

    region_space = HilbertSpace.new(region_modes)
    h_data = torch.zeros((region_space.dim, region_space.dim), dtype=torch.complex128)
    origin_idx = offset_to_idx[(0, 0)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        idx = offset_to_idx[(dx, dy)]
        h_data[origin_idx, idx] = -1.0 + 0j
        h_data[idx, origin_idx] = -1.0 + 0j

    h_real = Tensor(data=h_data, dims=(region_space, region_space))
    return lat, bloch_space, region_space, h_real


def test_bandstructure_plot_with_bz_path_plotly():
    lat, bloch_space, region_space, h_real = _make_2d_lattice_and_spaces()
    path = interpolate_reciprocal_path(
        lat.dual,
        [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0)],
        n_points=30,
        labels=["Gamma", "X", "M", "Gamma"],
    )

    F = fourier_transform(path.k_space, bloch_space, region_space)
    h_k = F @ h_real @ F.h(1, 2)

    fig = h_k.plot(
        "bandstructure",
        backend="plotly",
        title="Path Bandstructure",
        show=False,
        bz_path=path,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)
    assert fig.layout.xaxis.ticktext is not None
    tick_labels = list(fig.layout.xaxis.ticktext)
    assert tick_labels == ["Gamma", "X", "M", "Gamma"]


def test_bandstructure_plot_with_bz_path_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat, bloch_space, region_space, h_real = _make_2d_lattice_and_spaces()
    path = interpolate_reciprocal_path(
        lat.dual,
        [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0)],
        n_points=30,
        labels=["Gamma", "X", "M", "Gamma"],
    )

    F = fourier_transform(path.k_space, bloch_space, region_space)
    h_k = F @ h_real @ F.h(1, 2)

    fig = h_k.plot(
        "bandstructure",
        backend="matplotlib",
        title="Path Bandstructure",
        bz_path=path,
    )

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["Gamma", "X", "M", "Gamma"]
    plt.close(fig)
