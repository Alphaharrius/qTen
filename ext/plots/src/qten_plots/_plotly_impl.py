from typing import Optional, Union, Tuple, Sequence
import colorsys

import torch
import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.figure_factory as ff  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from qten.geometries.spatials import Lattice, ReciprocalLattice, Offset
from qten.linalg.tensors import Tensor
from qten.symbolics.state_space import brillouin_zone
from ._utils import compute_bonds
from .plottables import PointCloud


def _pointcloud_coords(obj: PointCloud) -> torch.Tensor:
    if not obj.offsets:
        return torch.empty((0, 0), dtype=torch.float64)

    coords = np.stack([offset.to_vec(np.ndarray) for offset in obj.offsets])
    return torch.tensor(coords, dtype=torch.float64)


def _canonical_lattice_offset(lattice: Lattice, offset: Offset) -> Offset:
    rebased = offset.rebase(lattice)
    fractional = rebased.fractional().rep
    cell = lattice.boundaries.wrap(rebased.rep - fractional)
    return Offset(rep=cell + fractional, space=lattice)


# --- Registered Plot Methods ---


@Lattice.register_plot_method("structure", backend="plotly")
def plot_structure(
    obj: Lattice,
    spin_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    plot_type: str = "edge-and-node",
    show: bool = True,
    fig: Optional[go.Figure] = None,
    color_by: str = "basis",
    highlights: Sequence[PointCloud] | None = None,
    **kwargs,
) -> go.Figure:
    """
    Visualize the lattice structure using Plotly.

    This function creates an interactive 3D or 2D plot of the lattice structure,
    including sites, bonds, and optional spin vectors.

    Parameters
    ----------
    obj : Lattice
        The lattice instance to visualize.
    spin_data : array-like, optional
        (N_sites, 3) array of spin vectors.
    plot_type : {'edge-and-node', 'scatter'}, default 'edge-and-node'
        Visualization style.
    show : bool, default True
        If True, calls `fig.show()` to display the plot immediately.
    fig : plotly.graph_objects.Figure, optional
        Existing figure to add traces to.
    color_by : {'basis', 'unit_cell'}, default 'basis'
        How to color the sites.
    **kwargs
        Additional keyword arguments passed to `go.Figure`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    valid_types = ["edge-and-node", "scatter"]
    if plot_type not in valid_types:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Options: {valid_types}")

    valid_color_by = ["basis", "unit_cell"]
    if color_by not in valid_color_by:
        raise ValueError(f"Invalid color_by '{color_by}'. Options: {valid_color_by}")

    coords = obj.cartes(torch.Tensor)
    coords_np = coords.numpy()

    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2] if obj.dim == 3 else None

    if fig is None:
        fig = go.Figure()

    # Bonds (Only for 'edge-and-node')
    if plot_type == "edge-and-node":
        x_lines, y_lines, z_lines = compute_bonds(coords, obj.dim)
        if x_lines:
            if obj.dim == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_lines,
                        y=y_lines,
                        z=z_lines,
                        mode="lines",
                        line=dict(color="black", width=1),
                        name="Bonds",
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_lines,
                        y=y_lines,
                        mode="lines",
                        line=dict(color="black", width=2),
                        name="Bonds",
                        showlegend=False,
                    )
                )

    # Sites
    num_basis = len(obj.unit_cell) if obj.unit_cell else 1
    num_cells = coords.shape[0] // num_basis

    n_colors = num_basis if color_by == "basis" else num_cells

    basis_colors = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in (
            colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.8, 0.9) for i in range(n_colors)
        )
    ]
    if color_by == "basis":
        for b in range(num_basis):
            indices = [c * num_basis + b for c in range(num_cells)]
            x_group = x[indices]
            y_group = y[indices]
            z_group = z[indices] if z is not None else None

            trace_name = f"Basis {b}"
            trace_color = basis_colors[b]

            if obj.dim == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_group,
                        y=y_group,
                        z=z_group,
                        mode="markers",
                        marker=dict(size=5, color=trace_color),
                        name=trace_name,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_group,
                        y=y_group,
                        mode="markers",
                        marker=dict(size=10, color=trace_color, symbol="circle"),
                        name=trace_name,
                    )
                )

    else:  # color_by == "unit_cell"
        for c in range(num_cells):
            start_idx = c * num_basis
            end_idx = start_idx + num_basis

            x_group = x[start_idx:end_idx]
            y_group = y[start_idx:end_idx]
            z_group = z[start_idx:end_idx] if z is not None else None

            trace_name = f"Cell {c}"
            trace_color = basis_colors[c]

            if obj.dim == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_group,
                        y=y_group,
                        z=z_group,
                        mode="markers",
                        marker=dict(size=5, color=trace_color),
                        name=trace_name,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_group,
                        y=y_group,
                        mode="markers",
                        marker=dict(size=10, color=trace_color, symbol="circle"),
                        name=trace_name,
                    )
                )

    # Spins (Optional)
    if spin_data is not None:
        if isinstance(spin_data, np.ndarray):
            spin_data = torch.from_numpy(spin_data)

        if spin_data.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Spin data shape {spin_data.shape} does not match sites {coords.shape[0]}. "
                f"Did you forget to provide spin data for all basis atoms?"
            )

        spin_np = spin_data.numpy()

        if obj.dim == 3:
            fig.add_trace(
                go.Cone(
                    x=x,
                    y=y,
                    z=z,
                    u=spin_np[:, 0],
                    v=spin_np[:, 1],
                    w=spin_np[:, 2],
                    sizemode="absolute",
                    sizeref=0.5,
                    anchor="tail",
                    colorscale="Viridis",
                    name="Spins",
                )
            )
        else:
            quiver = ff.create_quiver(
                x,
                y,
                spin_np[:, 0],
                spin_np[:, 1],
                scale=0.2,
                arrow_scale=0.3,
                name="Spins",
                line=dict(color="red"),
            )
            fig.add_traces(quiver.data)

    # Layout
    if highlights:
        site_to_index = {site: idx for idx, site in enumerate(obj.cartes())}
        fallback_colors = [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            for r, g, b in (
                colorsys.hsv_to_rgb((i * 0.38197 + 0.17) % 1.0, 0.95, 0.85)
                for i in range(len(highlights))
            )
        ]
        for idx, cloud in enumerate(highlights):
            trace_indices = []
            for offset in cloud.offsets:
                canonical = _canonical_lattice_offset(obj, offset)
                # if canonical not in site_to_index:
                #     raise ValueError(
                #         f"Highlighted offset {offset} is not a site in the plotted lattice."
                #     )
                trace_indices.append(site_to_index[canonical])

            if not trace_indices:
                continue

            trace_color = cloud.color or fallback_colors[idx]
            x_group = x[trace_indices]
            y_group = y[trace_indices]
            z_group = z[trace_indices] if z is not None else None
            if obj.dim == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_group,
                        y=y_group,
                        z=z_group,
                        mode="markers",
                        marker=dict(size=8, color=trace_color, symbol="diamond"),
                        name=f"Highlight {idx}",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_group,
                        y=y_group,
                        mode="markers",
                        marker=dict(size=13, color=trace_color, symbol="diamond"),
                        name=f"Highlight {idx}",
                    )
                )

    if obj.dim == 3:
        fig.update_layout(title="3D Lattice System", scene=dict(aspectmode="data"))
    else:
        fig.update_layout(
            title="2D Lattice System", yaxis=dict(scaleanchor="x", scaleratio=1)
        )

    if show:
        fig.show()
    return fig


@PointCloud.register_plot_method("scatter", backend="plotly")
def plot_pointcloud(
    obj: PointCloud,
    show: bool = True,
    fig: Optional[go.Figure] = None,
    **kwargs,
) -> go.Figure:
    coords = _pointcloud_coords(obj)
    if coords.shape[1] not in (2, 3):
        raise ValueError(
            f"PointCloud scatter supports only 2D or 3D points, got dimension {coords.shape[1]}."
        )

    coords_np = coords.numpy()
    if fig is None:
        fig = go.Figure()

    trace_color = obj.color or kwargs.pop("color", "#d1495b")
    if coords.shape[1] == 3:
        fig.add_trace(
            go.Scatter3d(
                x=coords_np[:, 0],
                y=coords_np[:, 1],
                z=coords_np[:, 2],
                mode="markers",
                marker=dict(size=6, color=trace_color),
                name="PointCloud",
            )
        )
        fig.update_layout(title="3D Point Cloud", scene=dict(aspectmode="data"))
    else:
        fig.add_trace(
            go.Scatter(
                x=coords_np[:, 0],
                y=coords_np[:, 1],
                mode="markers",
                marker=dict(size=10, color=trace_color, symbol="circle"),
                name="PointCloud",
            )
        )
        fig.update_layout(
            title="2D Point Cloud", yaxis=dict(scaleanchor="x", scaleratio=1)
        )

    if show:
        fig.show()
    return fig


@Tensor.register_plot_method("heatmap", backend="plotly")
def plot_heatmap(
    obj: Tensor,
    title: str = "Matrix Visualization",
    show: bool = True,
    fixed_indices: Optional[Tuple[int, ...]] = None,
    axes: Tuple[int, int] = (-2, -1),
    **kwargs,
) -> go.Figure:
    """
    Plot a heatmap of a matrix using Plotly.

    Handles complex matrices by showing Real and Imaginary parts side-by-side
    with a shared symmetric color scale.

    Parameters
    ----------
    obj : Tensor
        Tensor to visualize as a 2D heatmap.
    title : str, default "Matrix Visualization"
        Title of the plot.
    show : bool, default True
        Whether to show the plot immediately.
    fixed_indices : tuple of int, optional
        Indices used to fix non-heatmap dimensions. For an N-dimensional tensor,
        this must provide N-2 indices after selecting `axes`.
    axes : tuple of int, default (-2, -1)
        Pair of dimensions used as (row_axis, col_axis) in the heatmap.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure.
    """
    tensor = obj.data.detach().cpu()
    rank = tensor.ndim
    if rank < 2:
        raise ValueError(
            f"Heatmap requires rank >= 2 tensor, got shape {tuple(tensor.shape)}"
        )

    if len(axes) != 2:
        raise ValueError(f"`axes` must have length 2, got {axes}")

    normalized_axes = []
    for axis in axes:
        ax_norm = axis + rank if axis < 0 else axis
        if not (0 <= ax_norm < rank):
            raise ValueError(
                f"Axis {axis} is out of bounds for tensor with rank {rank}"
            )
        normalized_axes.append(ax_norm)
    row_axis, col_axis = normalized_axes
    if row_axis == col_axis:
        raise ValueError(f"`axes` must reference two different dimensions, got {axes}")

    permute_order = [i for i in range(rank) if i not in (row_axis, col_axis)] + [
        row_axis,
        col_axis,
    ]
    tensor = tensor.permute(*permute_order)

    expected_fixed = rank - 2
    fixed_indices_resolved: Tuple[int, ...]
    if fixed_indices is None:
        if expected_fixed == 0:
            fixed_indices_resolved = ()
        else:
            raise ValueError(
                f"Heatmap for shape {tuple(obj.data.shape)} with axes={axes} requires "
                f"`fixed_indices` of length {expected_fixed}."
            )
    else:
        if len(fixed_indices) != expected_fixed:
            raise ValueError(
                f"`fixed_indices` length must be {expected_fixed} for shape "
                f"{tuple(obj.data.shape)} with axes={axes}, got {len(fixed_indices)}."
            )
        fixed_indices_resolved = fixed_indices

    indexer: Tuple[Union[int, slice], ...] = (
        *fixed_indices_resolved,
        slice(None),
        slice(None),
    )
    try:
        tensor = tensor[indexer]
    except IndexError as exc:
        raise IndexError(
            f"`fixed_indices` {fixed_indices_resolved} is out of bounds for shape "
            f"{tuple(obj.data.shape)} with axes={axes}."
        ) from exc

    is_complex = tensor.is_complex()

    if is_complex:
        real_part = tensor.real
        imag_part = tensor.imag

        # Calculate global range using torch
        limit = max(torch.abs(real_part).max(), torch.abs(imag_part).max()).item()

        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Real Part", "Imaginary Part")
        )

        # Convert to numpy only for Plotly
        fig.add_trace(
            go.Heatmap(
                z=real_part.numpy(),
                colorscale="RdBu",
                zmin=-limit,
                zmax=limit,
                showscale=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=imag_part.numpy(),
                colorscale="RdBu",
                zmin=-limit,
                zmax=limit,
                showscale=True,
            ),
            row=1,
            col=2,
        )
    else:
        # Real matrix
        limit = torch.abs(tensor).max().item()
        fig = go.Figure(
            data=go.Heatmap(
                z=tensor.numpy(),
                colorscale="RdBu",
                zmin=-limit,
                zmax=limit,
                showscale=True,
            )
        )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)

    if show:
        fig.show()
    return fig


@Tensor.register_plot_method("spectrum", backend="plotly")
def plot_spectrum(
    obj: Tensor,
    title: str = "Spectrum Visualization",
    show: bool = True,
    fixed_indices: Optional[Tuple[int, ...]] = None,
    axes: Tuple[int, int] = (-2, -1),
    **kwargs,
) -> go.Figure:
    """
    Plot the eigenvalue spectrum using Plotly.

    Parameters
    ----------
    obj : Tensor
        Matrix/tensor to analyze as a 2D operator.
    title : str, default "Spectrum Visualization"
        Title of the plot.
    show : bool, default True
        Whether to show the plot immediately.
    fixed_indices : tuple of int, optional
        Indices used to fix non-matrix dimensions. For an N-dimensional tensor,
        this must provide N-2 indices after selecting `axes`.
    axes : tuple of int, default (-2, -1)
        Pair of dimensions used as (row_axis, col_axis) for spectrum analysis.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure.
    """
    tensor = obj.data.detach().cpu()

    rank = tensor.ndim
    if rank < 2:
        raise ValueError(
            f"Spectrum requires rank >= 2 tensor, got shape {tuple(tensor.shape)}"
        )

    if len(axes) != 2:
        raise ValueError(f"`axes` must have length 2, got {axes}")

    normalized_axes = []
    for axis in axes:
        ax_norm = axis + rank if axis < 0 else axis
        if not (0 <= ax_norm < rank):
            raise ValueError(
                f"Axis {axis} is out of bounds for tensor with rank {rank}"
            )
        normalized_axes.append(ax_norm)
    row_axis, col_axis = normalized_axes
    if row_axis == col_axis:
        raise ValueError(f"`axes` must reference two different dimensions, got {axes}")

    permute_order = [i for i in range(rank) if i not in (row_axis, col_axis)] + [
        row_axis,
        col_axis,
    ]
    tensor = tensor.permute(*permute_order)

    expected_fixed = rank - 2
    fixed_indices_resolved: Tuple[int, ...]
    if fixed_indices is None:
        if expected_fixed == 0:
            fixed_indices_resolved = ()
        else:
            raise ValueError(
                f"Spectrum for shape {tuple(tensor.shape)} with axes={axes} requires "
                f"`fixed_indices` of length {expected_fixed}."
            )
    else:
        if len(fixed_indices) != expected_fixed:
            raise ValueError(
                f"`fixed_indices` length must be {expected_fixed} for shape "
                f"{tuple(tensor.shape)} with axes={axes}, got {len(fixed_indices)}."
            )
        fixed_indices_resolved = fixed_indices

    indexer: Tuple[Union[int, slice], ...] = (
        *fixed_indices_resolved,
        slice(None),
        slice(None),
    )
    try:
        tensor = tensor[indexer]
    except IndexError as exc:
        raise IndexError(
            f"`fixed_indices` {fixed_indices_resolved} is out of bounds for shape "
            f"{tuple(tensor.shape)} with axes={axes}."
        ) from exc

    if tensor.shape[-2] != tensor.shape[-1]:
        raise ValueError(
            f"Spectrum requires a square matrix after slicing, got shape {tuple(tensor.shape)}"
        )

    # 2. Check for Hermiticity (M == M.H)
    is_complex = tensor.is_complex()
    is_hermitian = False

    # Calculate Frobenius norm once
    norm = torch.norm(tensor)
    if norm > 0:
        if is_complex:
            diff = torch.norm(tensor - tensor.conj().T)
        else:
            diff = torch.norm(tensor - tensor.T)

        if diff / norm < 1e-5:
            is_hermitian = True
    else:
        is_hermitian = True  # Zero matrix is Hermitian

    fig = go.Figure()

    # 3. Calculate and Plot
    if is_hermitian:
        # Returns real values sorted in ascending order
        evals = torch.linalg.eigvalsh(tensor)
        y_vals = evals.numpy()
        x_vals = np.arange(len(y_vals))

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+lines",
                marker=dict(size=6, color="blue"),
                name="Eigenvalues",
            )
        )
        fig.update_layout(xaxis_title="Index", yaxis_title="Eigenvalue")
    else:
        # General case (Complex plane)
        evals = torch.linalg.eigvals(tensor)
        real_parts = evals.real.numpy()
        imag_parts = evals.imag.numpy()

        fig.add_trace(
            go.Scatter(
                x=real_parts,
                y=imag_parts,
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Eigenvalues",
            )
        )
        fig.update_layout(xaxis_title="Real Part", yaxis_title="Imaginary Part")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(title=title)

    if show:
        fig.show()
    return fig


@Tensor.register_plot_method("bandstructure", backend="plotly")
def plot_bandstructure(
    obj: Tensor,
    title: str = "Band Structure",
    show: bool = True,
    fig: Optional[go.Figure] = None,
    mode: str = "auto",
    hide_nullspace: bool = False,
    nullspace_tol: float = 1e-9,
    **kwargs,
) -> go.Figure:
    """
    Plot the band structure using Real Physical Reciprocal Coordinates.

    Parameters
    ----------
    obj : Tensor
        Tensor to visualize as a band structure.
    title : str, default "Band Structure"
        Title of the plot.
    show : bool, default True
        Whether to show the plot immediately.
    fig : plotly.graph_objects.Figure, optional
        Existing figure to add traces to.
    hide_nullspace : bool, default False
        If True, mask surface points with |E| <= nullspace_tol so the
        band surface opens around the nullspace.
    nullspace_tol : float, default 1e-9
        Energy tolerance used when hide_nullspace is enabled.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure.
    """
    if obj.rank() != 3:
        raise ValueError(f"Tensor must be rank 3, got {obj.rank()}")
    if mode not in ("auto", "path", "surface"):
        raise ValueError(f"Invalid mode '{mode}'. Options: ('auto', 'path', 'surface')")
    if nullspace_tol < 0:
        raise ValueError(f"nullspace_tol must be non-negative, got {nullspace_tol}")

    k_space = obj.dims[0]
    k_points = list(k_space)

    # 2. Diagonalize
    eigvals = torch.linalg.eigvalsh(obj.data)  # (K, N_bands)
    eigvals_np = eigvals.detach().cpu().numpy()
    n_bands = eigvals_np.shape[1]

    if fig is None:
        fig = go.Figure()

    recip = None
    is_canonical_2d_bz = False
    if len(k_points) > 0:
        recip = k_points[0].space
        basis_sym = recip.basis

        basis_mat = np.array(basis_sym.evalf()).astype(float)

        k_fracs = [np.array(k.rep).astype(float).flatten() for k in k_points]
        k_fracs_arr = np.stack(k_fracs)  # Shape: (K, 2)

        # `Momentum.rep` is stored as a column vector in the reciprocal basis,
        # so batched row-wise conversion to Cartesian coordinates uses `B^T`.
        k_cart = k_fracs_arr @ basis_mat.T

        if (
            isinstance(recip, ReciprocalLattice)
            and recip.dim == 2
            and all(k.space == recip for k in k_points)
        ):
            canonical_k_space = brillouin_zone(recip)
            is_canonical_2d_bz = tuple(k_space.elements()) == tuple(
                canonical_k_space.elements()
            )
    else:
        k_cart = np.array([])

    is_surface = mode == "surface" or (mode == "auto" and is_canonical_2d_bz)
    if is_surface and not is_canonical_2d_bz:
        raise ValueError(
            "Surface bandstructure plotting requires the momentum axis to be the "
            "canonical 2D Brillouin-zone mesh returned by brillouin_zone(recip)."
        )

    if is_surface and recip is not None:
        # === 3D Surface Plot ===
        nx, ny = recip.shape

        KX = k_cart[:, 0].reshape(nx, ny)
        KY = k_cart[:, 1].reshape(nx, ny)
        evals_grid = eigvals_np.reshape(nx, ny, n_bands)
        if hide_nullspace:
            evals_grid = evals_grid.copy()
            evals_grid[np.abs(evals_grid) <= nullspace_tol] = np.nan

        for b in range(n_bands):
            fig.add_trace(
                go.Surface(
                    x=KX,
                    y=KY,
                    z=evals_grid[:, :, b],
                    name=f"Band {b}",
                    showscale=(b == 0),
                    colorscale="Viridis",
                    opacity=0.9,
                    hovertemplate="kx: %{x:.2f}<br>ky: %{y:.2f}<br>E: %{z:.3f}<extra></extra>",
                )
            )

    else:
        # === 1D Line Plot ===
        x_vals = np.array([0.0])
        if len(k_points) > 1:
            diffs = k_cart[1:] - k_cart[:-1]
            dists = np.linalg.norm(diffs, axis=1)
            x_vals = np.concatenate(([0.0], np.cumsum(dists)))

        for b in range(n_bands):
            fig.add_trace(
                go.Scatter(x=x_vals, y=eigvals_np[:, b], mode="lines", name=f"Band {b}")
            )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="kx (1/Å)",
            yaxis_title="ky (1/Å)",
            zaxis_title="Energy (eV)",
            aspectmode="data",
        )
        if is_surface
        else None,
    )

    if show:
        fig.show()

    return fig
