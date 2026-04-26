# Plot Methods

QTen plotting is exposed through [`Plottable.plot`][qten.plottings.Plottable.plot].
Backend implementations are registered by object type, method name, and backend
name, so users call a method on the object being visualized:

```python
fig = obj.plot("method_name", backend="plotly")
fig = obj.plot("method_name", backend="matplotlib")
```

The `qten-plots` extension currently registers Plotly and Matplotlib backends
through the `qten.plottings` entry-point group. The backend modules themselves
are implementation details; this page documents the public dispatcher calls.

## Overview

| Receiver | Method | Returns | Backends |
| --- | --- | --- | --- |
| [`Lattice`][qten.geometries.spatials.Lattice] | `structure` | Plotly or Matplotlib figure | `plotly`, `matplotlib` |
| [`PointCloud`][qten_plots.plottables.PointCloud] | `scatter` | Plotly or Matplotlib figure | `plotly`, `matplotlib` |
| [`Tensor`][qten.linalg.tensors.Tensor] | `heatmap` | Plotly or Matplotlib figure | `plotly`, `matplotlib` |
| [`Tensor`][qten.linalg.tensors.Tensor] | `spectrum` | Plotly or Matplotlib figure | `plotly`, `matplotlib` |
| [`Tensor`][qten.linalg.tensors.Tensor] | `column_scatter` | Plotly or Matplotlib figure | `plotly`, `matplotlib` |
| [`Tensor`][qten.linalg.tensors.Tensor] | `bandstructure` | Plotly or Matplotlib figure | `plotly`, `matplotlib` |

## Dispatcher Contract

All plot methods share the same dispatch shape:

```python
obj.plot(method, backend="plotly", *args, **kwargs)
```

| Input | Type | Meaning |
| --- | --- | --- |
| `method` | `str` | Registered plot method name for the receiver object. |
| `backend` | `str` | Backend implementation to use. The plotting extension supports `plotly` and `matplotlib`. |
| `*args` | `object` | Extra positional arguments forwarded to the selected backend implementation. |
| `**kwargs` | `object` | Extra keyword arguments forwarded to the selected backend implementation. |

Returns a backend-specific figure object: `plotly.graph_objects.Figure` for
Plotly or `matplotlib.figure.Figure` for Matplotlib.

Raises `ValueError` when no method/backend pair is registered for the receiver.
The error message includes available methods for that object when backends are
loaded successfully.

## `Lattice.plot("structure")`

Visualize the finite direct-space lattice as sites, nearest-neighbor bonds,
optional spin vectors, and optional highlighted point clouds.

```python
fig = lattice.plot("structure", backend="plotly", plot_type="edge-and-node")
fig = lattice.plot("structure", backend="matplotlib", plot_type="scatter")
```

Receiver:

| Receiver | Requirement |
| --- | --- |
| `lattice` | [`Lattice`][qten.geometries.spatials.Lattice] with finite boundary data and Cartesian coordinates available through `cartes(torch.Tensor)`. |

Inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `spin_data` | `np.ndarray` or `torch.Tensor` | `None` | Optional `(n_sites, 3)` spin vectors drawn at lattice sites. |
| `plot_type` | `str` | `edge-and-node` | `edge-and-node` draws nearest-neighbor bonds and sites; `scatter` draws only sites. |
| `color_by` | `str` | `basis` | `basis` colors by unit-cell site; `unit_cell` colors by translated cell. |
| `highlights` | `Sequence[PointCloud]` | `None` | Styled point-cloud overlays. |
| `use_lattice_coords` | `bool` | `False` | Show lattice-coordinate values in hover text where supported. |
| `periodic_image_opacity` | `float` | `0.5` | Opacity for periodic highlight images, in the inclusive range `[0, 1]`. |
| `**kwargs` | `object` | `{}` | Backend-specific figure options such as `figsize`, `cmap`, or line settings. |

Plotly inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `show` | `bool` | `True` | Immediately call `fig.show()` after adding traces. |
| `fig` | `plotly.graph_objects.Figure` | `None` | Existing figure to receive traces. |

Matplotlib inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `elev` | `float` | `30` | Elevation angle for 3D axes. |
| `azim` | `float` | `-60` | Azimuth angle for 3D axes. |
| `save_path` | `str` | `None` | Save the figure to this path. |
| `ax` | `matplotlib.axes.Axes` | `None` | Existing axes to draw on. |

Returns:

| Backend | Return type |
| --- | --- |
| `plotly` | `plotly.graph_objects.Figure` |
| `matplotlib` | `matplotlib.figure.Figure` |

Raises:

| Error | Condition |
| --- | --- |
| `ValueError` | `plot_type` is not `edge-and-node` or `scatter`. |
| `ValueError` | `color_by` is not `basis` or `unit_cell`. |
| `ValueError` | `periodic_image_opacity` is outside `[0, 1]`. |
| `ValueError` | `spin_data` does not have one row per lattice site. |
| `ValueError` | A highlight [`PointCloud`][qten_plots.plottables.PointCloud] has a spatial dimension different from the lattice. |

## `PointCloud.plot("scatter")`

Plot a reusable styled collection of spatial offsets. Use
[`PointCloud.of`][qten_plots.plottables.PointCloud.of] to build clouds from an
arbitrary iterable of offsets.

```python
cloud = PointCloud.of(offsets, name="Selected sites", color="red", marker="diamond")
fig = cloud.plot("scatter", backend="plotly")
fig = cloud.plot("scatter", backend="matplotlib")
```

Receiver attributes:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `offsets` | `FrozenSet[Offset]` | Spatial offsets to render as one point cloud. |
| `name` | `str | None` | Optional trace or legend label. |
| `color` | `str | None` | Optional backend color value. |
| `marker` | `str | None` | Optional marker alias such as `circle`, `square`, `diamond`, `cross`, or `x`. |
| `opacity` | `float | None` | Optional marker opacity in `[0, 1]`. |
| `size` | `float | None` | Optional positive marker size. |
| `border_color` | `str | None` | Optional marker border color. |
| `border_width` | `float | None` | Optional non-negative marker border width. |

Inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `use_lattice_coords` | `bool` | `False` | Show lattice-coordinate values in hover text where supported. |
| `**kwargs` | `object` | `{}` | Fallback style options such as `color`, `marker`, `size`, `border_color`, `border_width`, or `figsize`. |

Plotly inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `show` | `bool` | `True` | Immediately call `fig.show()` after adding the point-cloud trace. |
| `fig` | `plotly.graph_objects.Figure` | `None` | Existing figure to receive the trace. |

Matplotlib inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `ax` | `matplotlib.axes.Axes` | `None` | Existing axes to draw on. |
| `save_path` | `str` | `None` | Save the figure to this path. |

Returns:

| Backend | Return type |
| --- | --- |
| `plotly` | `plotly.graph_objects.Figure` |
| `matplotlib` | `matplotlib.figure.Figure` |

Raises:

| Error | Condition |
| --- | --- |
| `ValueError` | The point-cloud coordinates are not 2D or 3D. |

## `Tensor.plot("heatmap")`

Visualize a rank-2 tensor, or a rank-N tensor slice, as a real or complex matrix
heatmap. Complex matrices are split into real and imaginary panels.

```python
fig = tensor.plot("heatmap", backend="plotly")
fig = tensor.plot("heatmap", backend="matplotlib", axes=(-2, -1))
```

Receiver:

| Receiver | Requirement |
| --- | --- |
| `tensor` | [`Tensor`][qten.linalg.tensors.Tensor] with rank at least 2. |

Inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `title` | `str` | `Matrix Visualization` | Figure title. |
| `fixed_indices` | `Tuple[int, ...] | None` | Indices used to select all non-heatmap axes after `axes` are moved to the matrix position. Required when rank is greater than 2. |
| `axes` | `Tuple[int, int]` | `(-2, -1)` | Pair of tensor axes used as the heatmap row and column axes. |
| `**kwargs` | `object` | `{}` | Backend-specific layout options such as `figsize`. |

Plotly inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `show` | `bool` | `True` | Immediately call `fig.show()`. |

Matplotlib inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `save_path` | `str` | `None` | Save the figure to this path. |

Returns:

| Backend | Return type |
| --- | --- |
| `plotly` | `plotly.graph_objects.Figure` |
| `matplotlib` | `matplotlib.figure.Figure` |

Raises:

| Error | Condition |
| --- | --- |
| `ValueError` | The tensor rank is less than 2. |
| `ValueError` | `axes` does not contain exactly two distinct valid axes. |
| `ValueError` | `fixed_indices` is missing or has the wrong length for the selected matrix slice. |
| `IndexError` | A provided fixed index is out of bounds. |

## `Tensor.plot("spectrum")`

Plot eigenvalues of a square matrix or matrix slice. Hermitian or symmetric
matrices are shown as sorted real eigenvalues; general matrices are shown in the
complex plane.

```python
fig = tensor.plot("spectrum", backend="plotly")
fig = tensor.plot("spectrum", backend="matplotlib", fixed_indices=(0,))
```

Receiver:

| Receiver | Requirement |
| --- | --- |
| `tensor` | [`Tensor`][qten.linalg.tensors.Tensor] with rank at least 2 and a square selected matrix slice. |

Inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `title` | `str` | `Spectrum Visualization` | Figure title. |
| `fixed_indices` | `Tuple[int, ...] | None` | Indices used to select all non-matrix axes after `axes` are moved to the matrix position. |
| `axes` | `Tuple[int, int]` | `(-2, -1)` | Pair of tensor axes used as the matrix row and column axes. |
| `**kwargs` | `object` | `{}` | Backend-specific layout options such as `figsize`. |

Plotly inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `show` | `bool` | `True` | Immediately call `fig.show()`. |

Matplotlib inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `save_path` | `str` | `None` | Save the figure to this path. |

Returns:

| Backend | Return type |
| --- | --- |
| `plotly` | `plotly.graph_objects.Figure` |
| `matplotlib` | `matplotlib.figure.Figure` |

Raises:

| Error | Condition |
| --- | --- |
| `ValueError` | The tensor rank is less than 2. |
| `ValueError` | `axes` does not contain exactly two distinct valid axes. |
| `ValueError` | `fixed_indices` is missing or has the wrong length for the selected matrix slice. |
| `ValueError` | The selected matrix slice is not square. |
| `IndexError` | A provided fixed index is out of bounds. |

## `Tensor.plot("column_scatter")`

Plot each column of a rank-2 tensor as a separate spatial scatter panel. Marker
size encodes magnitude and marker color encodes complex phase.

```python
fig = tensor.plot("column_scatter", backend="plotly")
fig = tensor.plot("column_scatter", backend="matplotlib", ncols=4)
```

Receiver:

| Receiver | Requirement |
| --- | --- |
| `tensor` | Rank-2 [`Tensor`][qten.linalg.tensors.Tensor] with dims `(HilbertSpace, StateSpace)`. The first Hilbert-space dimension must be homogeneous and contain [`Offset`][qten.geometries.spatials.Offset] irreps. |

Inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `title` | `str` | `Tensor Scatter` | Figure title. |
| `default_size` | `float` | `16.0` | Marker size scale used when mapping magnitudes to scatter sizes. |
| `ncols` | `int` | `3` | Maximum number of subplot columns. |
| `use_lattice_coords` | `bool` | `False` | Use lattice-coordinate values in hover text where supported. |
| `unwrap_periodic` | `bool` | `True` | Unwrap periodic offsets for a spatially continuous display. |
| `**kwargs` | `object` | `{}` | Backend-specific layout options such as `figsize`. |

Plotly inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `show` | `bool` | `True` | Immediately call `fig.show()`. |

Matplotlib inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `save_path` | `str` | `None` | Save the figure to this path. |

Returns:

| Backend | Return type |
| --- | --- |
| `plotly` | `plotly.graph_objects.Figure` |
| `matplotlib` | `matplotlib.figure.Figure` |

Raises:

| Error | Condition |
| --- | --- |
| `ValueError` | The tensor is not rank 2. |
| `ValueError` | The tensor dims are not `(HilbertSpace, StateSpace)`. |
| `ValueError` | The first Hilbert space is not homogeneous. |
| `ValueError` | The first Hilbert space does not contain `Offset` irreps. |
| `ValueError` | The spatial coordinate dimension is not 1, 2, or 3. |
| `ValueError` | `default_size` or `ncols` is not positive. |

## `Tensor.plot("bandstructure")`

Plot eigenvalue bands for a rank-3 Hamiltonian tensor with momentum samples on
the first axis and matrix axes on the last two axes.

```python
fig = hamiltonian.plot("bandstructure", backend="plotly", mode="auto")
fig = hamiltonian.plot("bandstructure", backend="matplotlib", mode="path")
```

Receiver:

| Receiver | Requirement |
| --- | --- |
| `hamiltonian` | Rank-3 [`Tensor`][qten.linalg.tensors.Tensor] whose first dim is a momentum space and whose trailing axes are square Hamiltonian matrices. |

Inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `title` | `str` | `Band Structure` | Figure title. |
| `mode` | `str` | `auto` | `auto`, `path`, or `surface`. |
| `hide_nullspace` | `bool` | `False` | In surface mode, hide points whose absolute energy is below `nullspace_tol`. |
| `nullspace_tol` | `float` | `1e-9` | Non-negative energy tolerance used when hiding the nullspace. |
| `bz_path` | `BzPath | None` | Brillouin-zone path used for path-mode labels and interpolation from a regular grid. |
| `**kwargs` | `object` | `{}` | Backend-specific styling options such as `cmap`, `surface_alpha`, `line_width`, or legend controls. |

Plotly inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `show` | `bool` | `True` | Immediately call `fig.show()`. |
| `fig` | `plotly.graph_objects.Figure` | `None` | Existing figure to receive traces. |

Matplotlib inputs:

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `save_path` | `str` | `None` | Save the figure to this path. |
| `ax` | `matplotlib.axes.Axes` | `None` | Existing axes to draw on. Surface mode requires 3D axes. |
| `data_aspect` | `bool` | `True` | Preserve physical `kx:ky` aspect ratio in surface mode. |

Returns:

| Backend | Return type |
| --- | --- |
| `plotly` | `plotly.graph_objects.Figure` |
| `matplotlib` | `matplotlib.figure.Figure` |

Raises:

| Error | Condition |
| --- | --- |
| `ValueError` | The tensor is not rank 3. |
| `ValueError` | `mode` is not `auto`, `path`, or `surface`. |
| `ValueError` | `nullspace_tol` is negative. |
| `ValueError` | Surface mode is requested for momentum samples that are not a full 2D Brillouin-zone mesh. |
| `ValueError` | Surface mode is requested for effectively one-dimensional momentum samples. |
| `ValueError` | The momentum samples cannot be mapped onto the canonical Brillouin-zone mesh. |
| `ValueError` | Matplotlib surface mode receives non-3D axes. |

## Backend Notes

The functions registered in `qten_plots._plotly_impl` and
`qten_plots._mpl_impl` are deliberately documented through `obj.plot(...)`
because users should not import `plot_structure`, `plot_heatmap`, or similar
backend functions directly. Registering functions privately keeps the backend
surface flexible while giving users one stable plotting entry point.
