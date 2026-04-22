import numpy as np
import pytest
import torch
import sympy as sy
from sympy import ImmutableDenseMatrix

from qten.bands import (
    nearest_bands,
    bandselect,
    interpolate_path,
)
from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice
from qten.linalg.tensors import Tensor
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import (
    BzPath,
    IndexSpace,
    MomentumSpace,
    brillouin_zone,
)
from qten.symbolics import interpolate_reciprocal_path


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


# --- interpolate_path tests ---


def _recip_2d():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1, 0], [0, 1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    return lattice.dual


def _recip_3d():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0, 0])},
    )
    return lattice.dual


def test_interpolate_path_returns_bzpath_with_correct_n_points():
    recip = _recip_2d()
    path = interpolate_path(recip, [(0, 0), (0.5, 0), (0.5, 0.5)], n_points=50)

    assert isinstance(path, BzPath)
    assert isinstance(path.k_space, MomentumSpace)
    assert len(path.path_order) == 50
    assert path.k_space.dim == 50


def test_interpolate_path_waypoint_indices_match_waypoints():
    recip = _recip_2d()
    waypoints = [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0)]
    path = interpolate_path(recip, waypoints, n_points=100)

    assert len(path.waypoint_indices) == len(waypoints)
    assert len(path.path_order) == 100
    assert path.waypoint_indices[0] == 0
    assert path.waypoint_indices[-1] == 99

    elements = path.k_space.elements()
    for wp_pos, wp in zip(path.waypoint_indices, waypoints):
        k_idx = path.path_order[wp_pos]
        k = elements[k_idx]
        frac = np.array([float(k.rep[j, 0]) for j in range(recip.dim)])
        expected = np.array(wp, dtype=float)
        assert np.allclose(frac, expected, atol=1e-9)


def test_interpolate_path_auto_labels():
    recip = _recip_2d()
    path = interpolate_path(recip, [(0, 0), (0.5, 0)], n_points=10)

    assert len(path.labels) == 2
    assert path.labels[0] == str((0, 0))
    assert path.labels[1] == str((0.5, 0))


def test_interpolate_path_custom_labels():
    recip = _recip_2d()
    path = interpolate_path(
        recip, [(0, 0), (0.5, 0)], n_points=10, labels=["Gamma", "X"]
    )

    assert path.labels == ("Gamma", "X")


def test_interpolate_path_label_count_mismatch_raises():
    recip = _recip_2d()
    with pytest.raises(ValueError, match="labels"):
        interpolate_path(recip, [(0, 0), (0.5, 0)], n_points=10, labels=["Gamma"])


def test_interpolate_path_too_few_waypoints_raises():
    recip = _recip_2d()
    with pytest.raises(ValueError, match="two waypoints"):
        interpolate_path(recip, [(0, 0)], n_points=10)


def test_interpolate_path_wrong_dim_raises():
    recip = _recip_2d()
    with pytest.raises(ValueError, match="components"):
        interpolate_path(recip, [(0, 0, 0), (0.5, 0, 0)], n_points=10)


def test_interpolate_path_3d_lattice():
    recip = _recip_3d()
    waypoints = [(0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0, 0, 0)]
    path = interpolate_path(recip, waypoints, n_points=80)

    assert len(path.path_order) == 80
    # Closed loop: first and last waypoint share one k-point, so k_space has
    # one fewer unique element than path length.
    assert path.k_space.dim == 79
    assert len(path.waypoint_indices) == 4
    assert len(path.labels) == 4


def test_interpolate_path_distributes_proportionally():
    recip = _recip_2d()
    path = interpolate_path(recip, [(0, 0), (1, 0), (1, 0.5)], n_points=100)

    idx0, idx1, idx2 = path.waypoint_indices
    seg1_count = idx1 - idx0
    seg2_count = idx2 - idx1

    # Segment 1 has length 1, segment 2 has length 0.5 (in Cartesian space
    # with unit-basis reciprocal lattice), so ~2:1 ratio.
    assert seg1_count > seg2_count
    assert abs(seg1_count / max(seg2_count, 1) - 2.0) < 0.5


def test_interpolate_path_path_positions_are_monotonic():
    recip = _recip_2d()
    path = interpolate_path(recip, [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0)], n_points=60)

    positions = np.array(path.path_positions)
    assert positions[0] == 0.0
    assert np.all(np.diff(positions) >= -1e-15)
    assert len(positions) == len(path.path_order)


def test_interpolate_path_closed_loop_deduplicates():
    recip = _recip_2d()
    path = interpolate_path(recip, [(0, 0), (0.5, 0), (0, 0)], n_points=20)

    assert len(path.path_order) == 20
    # First and last path positions map to the same k_space index.
    assert path.path_order[0] == path.path_order[-1]
    assert path.k_space.dim == 19


def test_interpolate_path_named_route_with_points_dict():
    recip = _recip_2d()
    pts = {"Gamma": (0, 0), "X": (0.5, 0), "M": (0.5, 0.5)}
    path = interpolate_path(
        recip, ["Gamma", "X", "M", "Gamma"], n_points=40, points=pts
    )

    assert len(path.path_order) == 40
    assert path.labels == ("Gamma", "X", "M", "Gamma")

    elements = path.k_space.elements()
    first_k = elements[path.path_order[0]]
    frac = np.array([float(first_k.rep[j, 0]) for j in range(recip.dim)])
    assert np.allclose(frac, [0, 0], atol=1e-9)


def test_interpolate_path_named_route_missing_point_raises():
    recip = _recip_2d()
    with pytest.raises(ValueError, match="not found"):
        interpolate_path(recip, ["Gamma", "Z"], n_points=10, points={"Gamma": (0, 0)})


def test_interpolate_path_mixed_names_and_tuples():
    recip = _recip_2d()
    pts = {"Gamma": (0, 0)}
    path = interpolate_path(recip, ["Gamma", (0.5, 0)], n_points=10, points=pts)

    assert path.labels == ("Gamma", "(0.5, 0)")
    assert len(path.path_order) == 10


def test_interpolate_path_accessible_via_ops():
    from qten.ops import interpolate_path as ip

    recip = _recip_2d()
    path = ip(recip, [(0, 0), (0.5, 0)], n_points=10)
    assert isinstance(path, BzPath)


def test_interpolate_reciprocal_path_accessible_via_geometries():
    recip = _recip_2d()
    path = interpolate_reciprocal_path(recip, [(0, 0), (0.5, 0)], n_points=10)
    assert isinstance(path, BzPath)


# --- bands_near_value_as_tensor_KHH tests ---


def _nondiag_band_tensor() -> Tensor:
    """Build a (2, 2, 2) Hamiltonian with non-diagonal anchor eigenbasis.

    H(k=Gamma) = [[0, 1], [1, 0]] has eigvecs (1,-1)/sqrt(2) (eigval -1)
    and (1, 1)/sqrt(2) (eigval +1).
    H(k=X)     = [[1, 2], [2, 3]] is used to exercise the projection math.
    """
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 2)

    h_gamma = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    h_x = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.complex128)
    data = torch.stack([h_gamma, h_x], dim=0)
    return Tensor(data=data, dims=(k_space, band_space, band_space))


def test_bands_near_value_selects_single_band_at_gamma():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, point="Gamma", close_to=-1.0, tol=1e-6)

    assert result.dims[0] == tensor.dims[0]
    assert isinstance(result.dims[1], IndexSpace)
    assert isinstance(result.dims[2], IndexSpace)
    assert result.dims[1].dim == 1
    assert result.dims[2].dim == 1
    # Diagonal H: eigenvector for eigenvalue -1 at Gamma is e_1, so the
    # projection just picks the (1, 1) diagonal entry at every k.
    expected = torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected)


def test_bands_near_value_selects_multiple_bands_in_tolerance_window():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, point="Gamma", close_to=0.5, tol=1.6)

    assert result.dims[1].dim == 2
    assert result.dims[2].dim == 2
    expected = torch.zeros((2, 2, 2), dtype=torch.complex128)
    expected[0] = torch.diag(torch.tensor([-1.0, 2.0], dtype=torch.complex128))
    expected[1] = torch.diag(torch.tensor([1.0, 3.0], dtype=torch.complex128))
    assert torch.allclose(result.data, expected)


def test_bands_near_value_with_points_dict_non_gamma():
    tensor, _ = _band_tensor()

    result = nearest_bands(
        tensor,
        point="X",
        close_to=1.0,
        tol=1e-6,
        points={"X": (0.5,)},
    )

    assert result.dims[1].dim == 1
    # At X the eigenvalue 1.0 belongs to band 1; the projection picks (1, 1).
    expected = torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected)


def test_bands_near_value_with_explicit_fractional_tuple():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, point=(0.5,), close_to=3.0, tol=1e-6)

    assert result.dims[1].dim == 1
    # Band 2 (eigvalue 3 at X) projects to (2, 2) diagonal entries.
    expected = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected)


def test_bands_near_value_empty_subspace_when_no_match():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, close_to=1000.0, tol=1e-6)

    assert result.dims[1].dim == 0
    assert result.dims[2].dim == 0
    assert result.data.shape == (2, 0, 0)


def test_bands_near_value_non_diagonal_projection_math():
    tensor = _nondiag_band_tensor()

    result = nearest_bands(tensor, point="Gamma", close_to=-1.0, tol=1e-6)

    # Selected eigenvector at Gamma is v = (1, -1)/sqrt(2).
    # v^H H(Gamma) v = -1, v^H H(X) v = 0.5*(1 - 2 - 2 + 3) = 0.
    assert result.dims[1].dim == 1
    expected = torch.tensor([[[-1.0]], [[0.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected, atol=1e-10)


def test_bands_near_value_wraps_fractional_coordinates():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, point=(1.0,), close_to=-1.0, tol=1e-6)

    # (1.0,) wraps to Gamma = (0,), so this matches the Gamma selection.
    expected = torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected)


def test_bands_near_value_unknown_label_without_points_raises():
    tensor, _ = _band_tensor()

    with pytest.raises(KeyError, match="not found"):
        nearest_bands(tensor, point="Z")


def test_bands_near_value_dimension_mismatch_raises():
    tensor, _ = _band_tensor()

    with pytest.raises(ValueError, match="coordinates"):
        nearest_bands(tensor, point=(0.0, 0.0))


def test_bands_near_value_rejects_wrong_rank():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 2)
    data = torch.zeros((2, 2), dtype=torch.complex128)
    rank2 = Tensor(data=data, dims=(k_space, band_space))

    with pytest.raises(ValueError, match="rank 3"):
        nearest_bands(rank2)
