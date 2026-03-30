# qten-plots

`qten-plots` is the plotting extension package for `qten`.

It provides concrete plotting backends and registers them through the
`qten.plottings` entry-point group so `qten` objects can expose a uniform
`.plot(...)` API without forcing plotting dependencies into the core package.

## What It Adds

This extension currently provides plotting implementations for:

- `Lattice.plot("structure")`
- `Tensor.plot("heatmap")`
- `Tensor.plot("spectrum")`
- `Tensor.plot("column_scatter")`
- `Tensor.plot("bandstructure")`
- `PointCloud.plot("scatter")`

Backends currently included:

- `plotly`
- `matplotlib`

## Installation

Install the core package first, then install the plotting extension:

```bash
pip install qten
pip install qten-plots
```

If you use `uv` from the repository workspace:

```bash
uv sync --group dev
```

## How It Works

`qten` defines the `Plottable` dispatch mechanism in `qten.plottings`.
When `.plot(...)` is called, `qten` loads all installed entry points in the
`qten.plottings` group and registers the plotting implementations provided by
extension packages like `qten-plots`.

This keeps the main `qten` package focused on tensor and geometry logic while
allowing plotting support to evolve independently.

## Example

```python
import sympy as sy

from qten.geometries import Lattice, PeriodicBoundary

lattice = Lattice(
    basis=sy.ImmutableDenseMatrix([[1, 0], [0, 1]]),
    boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(4, 4)),
    unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
)

fig = lattice.plot("structure", backend="plotly", show=False)
fig.show()
```

To switch backend:

```python
fig = lattice.plot("structure", backend="matplotlib", show=False)
```

## Notes

- `qten-plots` depends on `qten`.
- The extension is loaded dynamically through Python package entry points.
- If `qten-plots` is not installed, `.plot(...)` calls will fail with a backend
  registration error.
