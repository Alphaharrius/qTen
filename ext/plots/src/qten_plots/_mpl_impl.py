import colorsys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Any, cast, Tuple, Sequence
from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice, Offset
from qten.linalg.tensors import Tensor
from qten.symbolics.state_space import (
    BzPath,
    StateSpace,
)
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from ._utils import (
    analyze_bandstructure_sampling,
    band_path_positions,
    compute_bonds,
    interpolate_path_on_grid,
    pointcloud_marker_for_mpl,
    pointcloud_size_for_mpl,
    unwrap_periodic_offsets,
)
from .plottables import PointCloud


# --- Registered Plot Methods (Matplotlib Backend) ---


def _pointcloud_coords(obj: PointCloud) -> torch.Tensor:
    ordered_offsets = tuple(sorted(obj.offsets))
    if not ordered_offsets:
        return torch.empty((0, 0), dtype=torch.float64)

    coords = np.stack([offset.to_vec(np.ndarray) for offset in ordered_offsets])
    return torch.tensor(coords, dtype=torch.float64)


def _minimal_periodic_highlight_groups(
    lattice: Lattice, cloud: PointCloud
) -> list[np.ndarray]:
    ordered_offsets = tuple(sorted(cloud.offsets))
    if len(ordered_offsets) <= 1:
        return []

    reps = np.stack(
        [
            np.array(offset.rebase(lattice.affine).rep.evalf(), dtype=float).reshape(-1)
            for offset in ordered_offsets
        ]
    )
    cart = np.stack([offset.to_vec(np.ndarray) for offset in ordered_offsets])
    boundary_basis = np.array(lattice.boundaries.basis.evalf(), dtype=float)
    boundary_basis_inv = np.linalg.inv(boundary_basis)
    lattice_basis = np.array(lattice.basis.evalf(), dtype=float)

    anchor = reps[0]
    grouped: dict[tuple[float, ...], list[np.ndarray]] = {}
    for rep, cart_point in zip(reps, cart):
        diff = rep - anchor
        coeffs = diff @ boundary_basis_inv.T
        wrapped_coeffs = coeffs - np.round(coeffs)
        unwrapped_rep = anchor + wrapped_coeffs @ boundary_basis.T
        shift_rep = unwrapped_rep - rep
        if np.allclose(shift_rep, 0.0, atol=1e-10):
            continue
        shift_cart = shift_rep @ lattice_basis.T
        shift_key = tuple(np.round(shift_cart, 12))
        grouped.setdefault(shift_key, []).append(cart_point + shift_cart)

    return [np.stack(points) for points in grouped.values()]


def _complex_phase_colors(values: np.ndarray) -> list[tuple[float, float, float]]:
    colors: list[tuple[float, float, float]] = []
    for value in values:
        phase = np.angle(value)
        hue = float((phase + np.pi) / (2 * np.pi))
        colors.append(colorsys.hsv_to_rgb(hue % 1.0, 0.95, 0.95))
    return colors


def _subplot_grid(n_items: int, ncols: int) -> tuple[int, int]:
    cols = min(ncols, max(1, n_items))
    rows = math.ceil(n_items / cols)
    return rows, cols


def _column_label(col_dim: StateSpace, index: int) -> str:
    return str(col_dim.elements()[index])


@Lattice.register_plot_method("structure", backend="matplotlib")
def plot_structure_mpl(
    obj: Lattice,
    spin_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    plot_type: str = "edge-and-node",
    elev: float = 30,
    azim: float = -60,
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
    color_by: str = "basis",
    highlights: Sequence[PointCloud] | None = None,
    use_lattice_coords: bool = False,
    periodic_image_opacity: float = 0.5,
    **kwargs,
) -> plt.Figure:
    """
    Visualize the lattice structure (sites, bonds, spins) using Matplotlib.

    Parameters
    ----------
    obj : Lattice
        The lattice instance to plot.
    spin_data : array-like, optional
        (N_sites, 3) array containing spin vectors for each site.
    plot_type : {'edge-and-node', 'scatter'}, default 'edge-and-node'
        Visualization style. 'edge-and-node' draws bonds between nearest neighbors;
        'scatter' draws only the sites.
    elev : float, default 30
        Elevation angle (in degrees) for 3D plots.
    azim : float, default -60
        Azimuth angle (in degrees) for 3D plots.
    save_path : str, optional
        If provided, saves the figure to this path. File format is inferred from the extension.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    color_by : {'basis', 'unit_cell'}, default 'basis'
        How to color the sites.
    **kwargs
        Additional keyword arguments passed to `plt.figure` (e.g., `figsize`).

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """
    valid_types = ["edge-and-node", "scatter"]
    if plot_type not in valid_types:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Options: {valid_types}")

    valid_color_by = ["basis", "unit_cell"]
    if color_by not in valid_color_by:
        raise ValueError(f"Invalid color_by '{color_by}'. Options: {valid_color_by}")
    if not (0.0 <= periodic_image_opacity <= 1.0):
        raise ValueError(
            f"periodic_image_opacity must lie in [0, 1], got {periodic_image_opacity}."
        )

    coords = obj.cartes(torch.Tensor)
    coords_np = coords.numpy()

    x = coords_np[:, 0]
    y = coords_np[:, 1]

    is_3d = obj.dim == 3
    z = coords_np[:, 2] if is_3d else None

    if ax is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title("3D Lattice System")
        else:
            ax = fig.add_subplot(111)
            ax.set_title("2D Lattice System")
            ax.set_aspect("equal")
    else:
        fig = ax.get_figure()

    # Bonds
    if plot_type == "edge-and-node":
        x_lines, y_lines, z_lines = compute_bonds(coords, obj.dim)
        if len(x_lines) > 0:
            if is_3d and z_lines is not None:
                ax.plot(
                    x_lines, y_lines, z_lines, color="black", linewidth=1, label="Bonds"
                )
            else:
                ax.plot(x_lines, y_lines, color="black", linewidth=1.5, label="Bonds")

    # Sites
    num_basis = len(obj.unit_cell) if obj.unit_cell else 1
    num_cells = coords.shape[0] // num_basis

    n_colors = num_basis if color_by == "basis" else num_cells

    basis_colors = [
        colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.8, 0.9) for i in range(n_colors)
    ]
    colors = []
    if color_by == "basis":
        for _ in range(num_cells):
            for b in range(num_basis):
                colors.append(basis_colors[b % len(basis_colors)])
    else:  # color_by == "unit_cell"
        for c in range(num_cells):
            color = basis_colors[c % len(basis_colors)]
            for _ in range(num_basis):
                colors.append(color)

    if is_3d:
        cast(Any, ax).scatter(x, y, z, c=colors, s=20, label="Sites")
    else:
        ax.scatter(x, y, c=colors, s=50, zorder=5, label="Sites")

    if highlights:
        fallback_colors = [
            colorsys.hsv_to_rgb((i * 0.38197 + 0.17) % 1.0, 0.95, 0.85)
            for i in range(len(highlights))
        ]
        for idx, cloud in enumerate(highlights):
            highlight_coords = _pointcloud_coords(cloud)
            if highlight_coords.shape[1] != obj.dim:
                raise ValueError(
                    "Highlight PointCloud dimension does not match plotted lattice "
                    f"dimension {obj.dim}."
                )
            if highlight_coords.shape[0] == 0:
                continue

            highlight_np = highlight_coords.numpy()
            x_group = highlight_np[:, 0]
            y_group = highlight_np[:, 1]
            trace_color = cloud.color or fallback_colors[idx]
            trace_marker = pointcloud_marker_for_mpl(cloud.marker, default="D")
            trace_alpha = cloud.opacity
            trace_size = pointcloud_size_for_mpl(
                cloud.size, default_area=55 if is_3d else 110
            )
            trace_edgecolor = cloud.border_color
            trace_linewidth = cloud.border_width
            trace_label = cloud.name or f"Highlight {idx}"
            if is_3d:
                z_group = highlight_np[:, 2]
                cast(Any, ax).scatter(
                    x_group,
                    y_group,
                    z_group,
                    c=[trace_color],
                    s=trace_size,
                    marker=trace_marker,
                    alpha=trace_alpha,
                    edgecolors=trace_edgecolor,
                    linewidths=trace_linewidth,
                    label=trace_label,
                )
                if isinstance(obj.boundaries, PeriodicBoundary):
                    for ghost_points in _minimal_periodic_highlight_groups(obj, cloud):
                        cast(Any, ax).scatter(
                            ghost_points[:, 0],
                            ghost_points[:, 1],
                            ghost_points[:, 2],
                            c=[trace_color],
                            s=trace_size,
                            marker=trace_marker,
                            alpha=periodic_image_opacity,
                            edgecolors=trace_edgecolor,
                            linewidths=trace_linewidth,
                            label="_nolegend_",
                        )
            else:
                ax.scatter(
                    x_group,
                    y_group,
                    c=[trace_color],
                    s=trace_size,
                    marker=trace_marker,
                    alpha=trace_alpha,
                    edgecolors=trace_edgecolor,
                    linewidths=trace_linewidth,
                    zorder=6,
                    label=trace_label,
                )
                if isinstance(obj.boundaries, PeriodicBoundary):
                    for ghost_points in _minimal_periodic_highlight_groups(obj, cloud):
                        ax.scatter(
                            ghost_points[:, 0],
                            ghost_points[:, 1],
                            c=[trace_color],
                            s=trace_size,
                            marker=trace_marker,
                            alpha=periodic_image_opacity,
                            edgecolors=trace_edgecolor,
                            linewidths=trace_linewidth,
                            zorder=5.5,
                            label="_nolegend_",
                        )

    # Spins
    if spin_data is not None:
        if isinstance(spin_data, np.ndarray):
            spin_data = torch.from_numpy(spin_data)

        if spin_data.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Spin data shape {spin_data.shape} does not match sites {coords.shape[0]}"
            )

        spin_np = spin_data.numpy()
        u = spin_np[:, 0]
        v = spin_np[:, 1]

        if is_3d:
            w = spin_np[:, 2]
            # Quiver in 3D: x, y, z, u, v, w
            ax.quiver(
                x, y, z, u, v, w, length=0.5, normalize=True, color="red", label="Spins"
            )
        else:
            # Quiver in 2D
            ax.quiver(x, y, u, v, color="red", scale=20, width=0.005, label="Spins")

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if is_3d:
        cast(Any, ax).set_zlabel("Z")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


@PointCloud.register_plot_method("scatter", backend="matplotlib")
def plot_pointcloud_mpl(
    obj: PointCloud,
    ax: Optional[Any] = None,
    save_path: Optional[str] = None,
    use_lattice_coords: bool = False,
    **kwargs,
) -> plt.Figure:
    coords = _pointcloud_coords(obj)
    if coords.shape[1] not in (2, 3):
        raise ValueError(
            f"PointCloud scatter supports only 2D or 3D points, got dimension {coords.shape[1]}."
        )

    coords_np = coords.numpy()
    is_3d = coords.shape[1] == 3

    if ax is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title("3D Point Cloud")
        else:
            ax = fig.add_subplot(111)
            ax.set_title("2D Point Cloud")
            ax.set_aspect("equal")
    else:
        fig = ax.get_figure()

    trace_color = obj.color or kwargs.get("color", "#d1495b")
    trace_marker = pointcloud_marker_for_mpl(
        obj.marker or kwargs.get("marker"), default="o"
    )
    trace_alpha = obj.opacity
    raw_size = obj.size if obj.size is not None else kwargs.get("size")
    trace_size = pointcloud_size_for_mpl(raw_size, default_area=30 if is_3d else 60)
    trace_edgecolor = obj.border_color or kwargs.get("border_color")
    trace_linewidth = (
        obj.border_width if obj.border_width is not None else kwargs.get("border_width")
    )
    trace_label = obj.name or "PointCloud"
    if is_3d:
        cast(Any, ax).scatter(
            coords_np[:, 0],
            coords_np[:, 1],
            coords_np[:, 2],
            c=trace_color,
            s=trace_size,
            marker=trace_marker,
            alpha=trace_alpha,
            edgecolors=trace_edgecolor,
            linewidths=trace_linewidth,
            label=trace_label,
        )
        cast(Any, ax).set_zlabel("Z")
    else:
        ax.scatter(
            coords_np[:, 0],
            coords_np[:, 1],
            c=trace_color,
            s=trace_size,
            marker=trace_marker,
            alpha=trace_alpha,
            edgecolors=trace_edgecolor,
            linewidths=trace_linewidth,
            label=trace_label,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


@Tensor.register_plot_method("heatmap", backend="matplotlib")
def plot_heatmap_mpl(
    obj: Tensor,
    title: str = "Matrix Visualization",
    save_path: Optional[str] = None,
    fixed_indices: Optional[Tuple[int, ...]] = None,
    axes: Tuple[int, int] = (-2, -1),
    **kwargs,
) -> plt.Figure:
    """
    Plot a heatmap of a matrix using Matplotlib.

    If the matrix is complex, displays Real and Imaginary parts in side-by-side subplots.

    Parameters
    ----------
    obj : Tensor
        Tensor to visualize as a 2D heatmap.
    title : str, default "Matrix Visualization"
        Title of the figure.
    save_path : str, optional
        If provided, saves the figure to this path.
    fixed_indices : tuple of int, optional
        Indices used to fix non-heatmap dimensions. For an N-dimensional tensor,
        this must provide N-2 indices after selecting `axes`.
    axes : tuple of int, default (-2, -1)
        Pair of dimensions used as (row_axis, col_axis) in the heatmap.
    **kwargs
        Additional keyword arguments passed to `plt.subplots` (e.g., `figsize`).

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
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
        real_part = tensor.real.numpy()
        imag_part = tensor.imag.numpy()

        limit = max(np.abs(real_part).max(), np.abs(imag_part).max())

        fig, subplot_axes = plt.subplots(1, 2, figsize=kwargs.get("figsize", (12, 5)))

        im1 = subplot_axes[0].imshow(real_part, cmap="RdBu", vmin=-limit, vmax=limit)
        subplot_axes[0].set_title("Real Part")
        fig.colorbar(im1, ax=subplot_axes[0])

        im2 = subplot_axes[1].imshow(imag_part, cmap="RdBu", vmin=-limit, vmax=limit)
        subplot_axes[1].set_title("Imaginary Part")
        fig.colorbar(im2, ax=subplot_axes[1])

        fig.suptitle(title)
    else:
        data = tensor.numpy()
        limit = np.abs(data).max()

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 5)))
        im = ax.imshow(data, cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


@Tensor.register_plot_method("spectrum", backend="matplotlib")
def plot_spectrum_mpl(
    obj: Tensor,
    title: str = "Spectrum Visualization",
    save_path: Optional[str] = None,
    fixed_indices: Optional[Tuple[int, ...]] = None,
    axes: Tuple[int, int] = (-2, -1),
    **kwargs,
) -> plt.Figure:
    """
    Plot the eigenvalue spectrum of a matrix using Matplotlib.

    - If Hermitian/Symmetric: Plots sorted real eigenvalues.
    - If Non-Hermitian: Plots eigenvalues in the complex plane.

    Parameters
    ----------
    obj : Tensor
        Matrix/tensor to analyze as a 2D operator.
    title : str, default "Spectrum Visualization"
        Title of the figure.
    save_path : str, optional
        If provided, saves the figure to this path.
    fixed_indices : tuple of int, optional
        Indices used to fix non-matrix dimensions. For an N-dimensional tensor,
        this must provide N-2 indices after selecting `axes`.
    axes : tuple of int, default (-2, -1)
        Pair of dimensions used as (row_axis, col_axis) for spectrum analysis.
    **kwargs
        Additional keyword arguments passed to `plt.subplots`.

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
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

    # Check Hermiticity
    is_complex = tensor.is_complex()
    is_hermitian = False
    norm = torch.norm(tensor)
    if norm > 0:
        if is_complex:
            diff = torch.norm(tensor - tensor.conj().T)
        else:
            diff = torch.norm(tensor - tensor.T)
        if diff / norm < 1e-5:
            is_hermitian = True
    else:
        is_hermitian = True

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 5)))

    if is_hermitian:
        evals = torch.linalg.eigvalsh(tensor).numpy()
        ax.plot(evals, "b.-", markersize=10)
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue")
        ax.grid(True)
    else:
        evals = torch.linalg.eigvals(tensor).numpy()
        ax.scatter(evals.real, evals.imag, c="r", s=30)
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.grid(True)
        ax.set_aspect("equal")

    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


@Tensor.register_plot_method("column_scatter", backend="matplotlib")
def plot_tensor_column_scatter_mpl(
    obj: Tensor,
    title: str = "Tensor Scatter",
    save_path: Optional[str] = None,
    default_size: float = 16.0,
    ncols: int = 3,
    use_lattice_coords: bool = False,
    unwrap_periodic: bool = True,
    **kwargs,
) -> plt.Figure:
    if obj.rank() != 2:
        raise ValueError(
            "Tensor scatter requires a rank-2 tensor with dims "
            f"(HilbertSpace, StateSpace), got rank {obj.rank()}."
        )
    if default_size <= 0:
        raise ValueError(f"default_size must be positive, got {default_size}")
    if ncols <= 0:
        raise ValueError(f"ncols must be positive, got {ncols}")

    row_dim, col_dim = obj.dims
    if not isinstance(row_dim, HilbertSpace) or not isinstance(col_dim, StateSpace):
        raise ValueError(
            "Tensor scatter requires dims (HilbertSpace, StateSpace), got "
            f"({type(row_dim).__name__}, {type(col_dim).__name__})."
        )
    if not row_dim.is_homogeneous():
        raise ValueError(
            "Tensor scatter requires the first HilbertSpace to be homogeneous."
        )

    row_basis = cast(tuple[U1Basis, ...], row_dim.elements())
    if row_basis:
        try:
            row_basis[0].irrep_of(Offset)
        except ValueError as exc:
            raise ValueError(
                "Tensor scatter requires the first HilbertSpace to contain Offset irreps."
            ) from exc

    row_offsets = [basis.irrep_of(Offset) for basis in row_basis]
    if row_offsets:
        plot_coords = (
            unwrap_periodic_offsets(row_offsets, use_lattice_coords=False)
            if unwrap_periodic
            else None
        )
        coords = (
            plot_coords
            if plot_coords is not None
            else np.stack(
                [offset.to_vec(np.ndarray).reshape(-1) for offset in row_offsets]
            )
        )
        spatial_dim = coords.shape[1]
    else:
        coords = np.empty((0, 0), dtype=float)
        spatial_dim = 0
    if spatial_dim not in (1, 2, 3):
        raise ValueError(
            "Tensor scatter only supports Offset coordinates of dimension 1, 2, or 3; "
            f"got {spatial_dim}."
        )

    tensor = obj.data.detach().cpu()
    if not tensor.is_complex():
        tensor = tensor.to(torch.complex128)
    tensor_np = tensor.numpy()

    n_columns = tensor_np.shape[1]
    nrows, ncols_actual = _subplot_grid(n_columns, ncols=ncols)
    figsize = kwargs.pop("figsize", (5 * ncols_actual, 4 * nrows))
    column_labels = [_column_label(col_dim, j) for j in range(n_columns)]

    subplot_kwargs: dict[str, Any] = {"figsize": figsize, "squeeze": False}
    if spatial_dim == 3:
        subplot_kwargs["subplot_kw"] = {"projection": "3d"}

    fig, axes = plt.subplots(nrows, ncols_actual, **subplot_kwargs)
    fig.suptitle(title)

    max_mag = float(np.abs(tensor_np).max()) if tensor_np.size > 0 else 0.0
    for j in range(n_columns):
        ax = axes[j // ncols_actual, j % ncols_actual]
        column = tensor_np[:, j]
        magnitudes = np.abs(column)
        if max_mag > 0:
            sizes = default_size * magnitudes / max_mag
        else:
            sizes = np.full_like(magnitudes, fill_value=default_size, dtype=float)
        colors = _complex_phase_colors(column)

        if spatial_dim == 3:
            cast(Any, ax).scatter(
                coords[:, 0], coords[:, 1], coords[:, 2], s=sizes, c=colors
            )
            cast(Any, ax).set_zlabel("Z")
        else:
            y = coords[:, 1] if spatial_dim == 2 else np.zeros(coords.shape[0])
            ax.scatter(coords[:, 0], y, s=sizes, c=colors)
            if spatial_dim == 2:
                ax.set_aspect("equal")

        ax.set_title(column_labels[j])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.2)

    for j in range(n_columns, nrows * ncols_actual):
        axes[j // ncols_actual, j % ncols_actual].set_visible(False)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


@Tensor.register_plot_method("bandstructure", backend="matplotlib")
def plot_bandstructure_mpl(
    obj: Tensor,
    title: str = "Band Structure",
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
    data_aspect: bool = True,
    mode: str = "auto",
    hide_nullspace: bool = False,
    nullspace_tol: float = 1e-9,
    bz_path: Optional[BzPath] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot the band structure of a Hamiltonian tensor using Matplotlib.
    The tensor must be rank-3 with momentum samples on axis 0 and
    matrix axes on the last two dimensions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If provided, the plot is added to this axes and
        the corresponding figure is returned via `ax.get_figure()`.
    data_aspect : bool, default True
        If True, keep the real physical kx:ky axis ratio in 3D surface mode.
    hide_nullspace : bool, default False
        If True, mask surface points with |E| <= nullspace_tol so the
        band surface opens around the nullspace.
    nullspace_tol : float, default 1e-9
        Energy tolerance used when hide_nullspace is enabled.
    bz_path : BzPath, optional
        Brillouin-zone path returned by `[`interpolate_reciprocal_path`][qten.symbolics.ops.interpolate_reciprocal_path]`. When given,
        vertical dividers and high-symmetry-point labels are drawn on the
        path-mode x-axis.
    """
    # 1. Check Dimensions
    if obj.rank() != 3:
        raise ValueError(f"Tensor must be rank 3, got {obj.rank()}")
    if mode not in ("auto", "path", "surface"):
        raise ValueError(f"Invalid mode '{mode}'. Options: ('auto', 'path', 'surface')")
    if nullspace_tol < 0:
        raise ValueError(f"nullspace_tol must be non-negative, got {nullspace_tol}")

    k_space = obj.dims[0]

    # 2. Diagonalize
    hk_data = obj.data
    eigvals = torch.linalg.eigvalsh(hk_data)  # (K, N_bands)
    eigvals_np = eigvals.detach().cpu().numpy()
    n_bands = eigvals_np.shape[1]

    # 3. Build Cartesian reciprocal coordinates once for both modes.
    (
        k_cart,
        recip,
        is_surface_compatible_2d_bz,
        effective_dim,
        surface_order,
    ) = analyze_bandstructure_sampling(k_space)

    is_surface = mode == "surface" or (
        mode == "auto" and is_surface_compatible_2d_bz and effective_dim >= 2
    )
    if is_surface and not is_surface_compatible_2d_bz:
        raise ValueError(
            "Surface bandstructure plotting requires the momentum axis to be the "
            "full 2D Brillouin-zone mesh returned by brillouin_zone(recip), up to "
            "permutation."
        )
    if is_surface and effective_dim < 2:
        raise ValueError(
            "Surface bandstructure plotting requires two varying momentum "
            "directions. Use mode='path' for effectively 1D k-samples."
        )

    if is_surface and recip is not None and len(k_cart) > 0:
        # 2D Surface Plot
        # Requires 3D projection
        if ax is None:
            fig = plt.figure(figsize=kwargs.get("figsize", (10, 8)))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()
            if not hasattr(ax, "zaxis"):
                raise ValueError(
                    "A 3D axes is required for 2D grid bandstructure surface."
                )

        # Reshape eigenvalues
        if surface_order is None:
            raise ValueError(
                "Unable to map momentum samples onto the canonical BZ mesh."
            )
        surface_k_cart = k_cart[surface_order]
        surface_eigvals = eigvals_np[surface_order]

        evals_grid = surface_eigvals.reshape(recip.shape[0], recip.shape[1], n_bands)
        if hide_nullspace:
            evals_grid = evals_grid.copy()
            evals_grid[np.abs(evals_grid) <= nullspace_tol] = np.nan

        KX = surface_k_cart[:, 0].reshape(recip.shape[0], recip.shape[1])
        KY = surface_k_cart[:, 1].reshape(recip.shape[0], recip.shape[1])

        cmap = kwargs.get("cmap", "viridis")
        surface_alpha = kwargs.get("surface_alpha", 0.85)
        for b in range(n_bands):
            ax.plot_surface(
                KX,
                KY,
                evals_grid[:, :, b],
                cmap=cmap,
                alpha=surface_alpha,
                linewidth=0,
                antialiased=True,
            )

        ax.set_title(title)
        ax.set_xlabel("kx (1/Å)")
        ax.set_ylabel("ky (1/Å)")
        # Explicit cast to avoid type checking issues with dynamic ax
        cast(Any, ax).set_zlabel("Energy (eV)")
        if data_aspect:
            x_span = float(np.ptp(KX))
            y_span = float(np.ptp(KY))
            finite_evals = evals_grid[np.isfinite(evals_grid)]
            z_span = float(np.ptp(finite_evals)) if finite_evals.size > 0 else 1.0
            # Keep real kx:ky scaling (plotly's aspectmode='data' equivalent).
            cast(Any, ax).set_box_aspect(
                (
                    x_span if x_span > 0.0 else 1.0,
                    y_span if y_span > 0.0 else 1.0,
                    z_span if z_span > 0.0 else 1.0,
                )
            )

    else:
        # 1D Line Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))
        else:
            fig = ax.get_figure()

        if bz_path is not None:
            x_vals = np.array(bz_path.path_positions)
            if k_space == bz_path.k_space:
                plot_eigvals = eigvals_np[list(bz_path.path_order)]
            else:
                plot_eigvals = interpolate_path_on_grid(bz_path, k_space, eigvals_np)
        else:
            x_vals = band_path_positions(k_space, k_cart)
            plot_eigvals = eigvals_np

        line_width = kwargs.get("line_width", 1.5)
        for b in range(plot_eigvals.shape[1]):
            ax.plot(x_vals, plot_eigvals[:, b], linewidth=line_width, label=f"Band {b}")

        ax.set_title(title)
        ax.set_ylabel("Energy (eV)")
        if len(x_vals) > 1 and x_vals[-1] > x_vals[0]:
            ax.set_xlim(float(x_vals[0]), float(x_vals[-1]))
        else:
            ax.set_xlim(-0.5, 0.5)

        if bz_path is not None:
            wp_x = [float(x_vals[i]) for i in bz_path.waypoint_indices]
            for x in wp_x:
                ax.axvline(x, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xticks(wp_x)
            ax.set_xticklabels(list(bz_path.labels))
        else:
            ax.set_xlabel("k-path (1/Å)")

        ax.grid(True, alpha=kwargs.get("grid_alpha", 0.3))
        if kwargs.get("legend", False):
            ax.legend(loc=kwargs.get("legend_loc", "best"))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
