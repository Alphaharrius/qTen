# QTen

A torch-based tensor library for quantum related computations.

## Installation

QTen requires Python 3.11 or newer.

Install the package from the project root with `pip`:

```bash
pip install '.[cpu]'
```

If you want a CUDA-enabled PyTorch build, choose one of the CUDA extras declared in `pyproject.toml`:

```bash
pip install '.[cu126]'
pip install '.[cu128]'
pip install '.[cu129]'
pip install '.[cu130]'
```

If you use `uv`, install the package rather than syncing a developer environment:

```bash
uv pip install '.[cpu]'
```

For development:

```bash
uv sync --extra cpu --group dev
```

## Quick Start

The package root exports tensor helpers such as `Tensor`, `zeros`, `ones`, `eye`, and `fourier_transform`. Geometry, symbolic, and band-structure tools are exposed from submodules.

## Build And Plot A Lattice

This example builds a square lattice with two sites in the unit cell and plots it.

```python
import sympy as sy

from qten.geometries import Lattice, PeriodicBoundary

lattice = Lattice(
    basis=sy.ImmutableDenseMatrix([[1, 0], [0, 1]]),
    boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(4, 4)),
    unit_cell={
        "A": sy.ImmutableDenseMatrix([0, 0]),
        "B": sy.ImmutableDenseMatrix([sy.Rational(1, 2), sy.Rational(1, 2)]),
    },
)

print(lattice.shape)         # (4, 4)
print(lattice.coords().shape)  # torch.Size([32, 2])

fig = lattice.plot("structure", backend="plotly", show=False)
fig.show()
```

For a simple one-site lattice, `shape=` is enough:

```python
import sympy as sy

from qten.geometries import Lattice

lattice = Lattice(
    basis=sy.ImmutableDenseMatrix([[1, 0], [0, 1]]),
    shape=(3, 3),
)
```

## Tensor Examples

### 1. Tensor With Named Hilbert-Space Axes

```python
from dataclasses import dataclass

import torch

import qten
from qten.symbolics import HilbertSpace, U1Basis


@dataclass(frozen=True)
class Orb:
    name: str


left = HilbertSpace.new(
    [
        U1Basis.new(Orb("s0")),
        U1Basis.new(Orb("s1")),
    ]
)
right = HilbertSpace.new(
    [
        U1Basis.new(Orb("p0")),
        U1Basis.new(Orb("p1")),
        U1Basis.new(Orb("p2")),
    ]
)

t = qten.Tensor(
    data=torch.arange(6, dtype=torch.float64).reshape(2, 3),
    dims=(left, right),
)

print(t.rank())       # 2
print(t.dims[0].dim)  # 2
print(t.dims[1].dim)  # 3
```

### 2. Identity, Zeros, And Tensor Algebra

```python
from dataclasses import dataclass

import torch

import qten
from qten.symbolics import HilbertSpace, U1Basis


@dataclass(frozen=True)
class Orb:
    name: str


space = HilbertSpace.new([U1Basis.new(Orb("a")), U1Basis.new(Orb("b"))])

hopping = qten.Tensor(
    data=torch.tensor([[0.0, -1.0], [-1.0, 0.0]], dtype=torch.float64),
    dims=(space, space),
)

identity = qten.eye((space, space))
mask = qten.ones((space, space))
empty = qten.zeros((space, space))

shifted = hopping + 0.5 * identity
assert shifted.dims == (space, space)
assert mask.data.shape == empty.data.shape == (2, 2)
```

### 3. Fourier Transform To Momentum Space

This is a common workflow for building a momentum-resolved Hamiltonian from a real-space operator.

```python
from dataclasses import dataclass

import sympy as sy
import torch

import qten
from qten.geometries import Lattice, PeriodicBoundary, Offset
from qten.symbolics import HilbertSpace, U1Basis, brillouin_zone


@dataclass(frozen=True)
class Orb:
    name: str


lattice = Lattice(
    basis=sy.ImmutableDenseMatrix([[1, 0], [0, 1]]),
    boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(4, 4)),
    unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
)

r0 = Offset(rep=sy.ImmutableDenseMatrix([0, 0]), space=lattice.affine)
bloch_space = HilbertSpace.new([U1Basis.new(r0, Orb("s"))])

region_offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
region_space = HilbertSpace.new(
    U1Basis.new(Offset(rep=sy.ImmutableDenseMatrix([dx, dy]), space=lattice.affine), Orb("s"))
    for dx, dy in region_offsets
)

h_real = qten.Tensor(
    data=torch.tensor(
        [
            [0, -1, -1, -1, -1],
            [-1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
        ],
        dtype=torch.complex128,
    ),
    dims=(region_space, region_space),
)

k_space = brillouin_zone(lattice.dual)
F = qten.fourier_transform(k_space, bloch_space, region_space)
h_k = F @ h_real @ F.h(1, 2)

print(h_k.dims[0])  # MomentumSpace(...)
```

## AffineTransform Examples

You can construct affine symmetries directly or use the point-group query helper.

### 1. Build An Affine Transform Directly

```python
import sympy as sy

from qten.geometries import AffineSpace, Offset
from qten.pointgroups import AffineTransform

x, y = sy.symbols("x y")
space = AffineSpace(basis=sy.ImmutableDenseMatrix.eye(2))
shift = Offset(rep=sy.ImmutableDenseMatrix([1, 0]), space=space)

t = AffineTransform(
    irrep=sy.ImmutableDenseMatrix([[0, -1], [1, 0]]),
    axes=(x, y),
    offset=shift,
    basis_function_order=1,
)

print(t.affine_rep)
```

### 2. Use The Point-Group Parser

```python
from qten.pointgroups import pointgroup

c4 = pointgroup("c4-xy:xy-o1")
mirror = pointgroup("m-xy:x-o1")

print(c4.irrep)
print(mirror.irrep)
```

### 3. Act On Symbolic Basis Functions

```python
import sympy as sy
from sympy import ImmutableDenseMatrix

from qten.geometries import AffineSpace, Offset
from qten.pointgroups import AbelianBasis, AffineTransform

x = sy.symbols("x")
space = AffineSpace(basis=ImmutableDenseMatrix.eye(1))
origin = Offset(rep=ImmutableDenseMatrix([0]), space=space)

parity = AffineTransform(
    irrep=ImmutableDenseMatrix([[-1]]),
    axes=(x,),
    offset=origin,
    basis_function_order=1,
)

f = AbelianBasis(
    expr=x,
    axes=(x,),
    order=1,
    rep=ImmutableDenseMatrix([1]),
)

phase, transformed = parity(f).coef, parity(f).base
print(phase)        # -1
print(transformed)  # AbelianBasis(x)
```

### 4. Build `AbelianBasis` Directly From An `AffineTransform`

`AffineTransform.basis` computes eigen-basis functions of the polynomial representation and returns them as a mapping from eigenvalue to `AbelianBasis`.

```python
import sympy as sy
from sympy import ImmutableDenseMatrix

from qten.geometries import AffineSpace, Offset
from qten.pointgroups import AffineTransform

x, y = sy.symbols("x y")
space = AffineSpace(basis=ImmutableDenseMatrix.eye(2))
origin = Offset(rep=ImmutableDenseMatrix([0, 0]), space=space)

mirror = AffineTransform(
    irrep=ImmutableDenseMatrix([[1, 0], [0, -1]]),
    axes=(x, y),
    offset=origin,
    basis_function_order=1,
)

eigen_basis = mirror.basis
even = eigen_basis[1]
odd = eigen_basis[-1]

print(even)      # AbelianBasis(x)
print(odd)       # AbelianBasis(y)
print(even.expr) # x
print(odd.expr)  # y
```

## `bandfold` And `bandaffine`

These helpers work on rank-3 tensors with dims `(MomentumSpace, HilbertSpace, HilbertSpace)`.

### 1. Build A Simple Band Tensor

```python
from dataclasses import dataclass

import sympy as sy
import torch

import qten
from qten.geometries import Lattice, PeriodicBoundary, Offset
from qten.symbolics import HilbertSpace, U1Basis, brillouin_zone


@dataclass(frozen=True)
class Orb:
    name: str


lattice = Lattice(
    basis=sy.ImmutableDenseMatrix.eye(2),
    boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
    unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
)
k_space = brillouin_zone(lattice.dual)

r_x = Offset(rep=sy.ImmutableDenseMatrix([sy.Rational(1, 2), 0]), space=lattice.affine)
r_y = Offset(rep=sy.ImmutableDenseMatrix([0, sy.Rational(1, 2)]), space=lattice.affine)
band_space = HilbertSpace.new([U1Basis.new(r_x, Orb("p")), U1Basis.new(r_y, Orb("p"))])

data = torch.zeros((k_space.dim, 2, 2), dtype=torch.complex128)
for n in range(k_space.dim):
    data[n, 0, 0] = 0.2 + 0.1 * n
    data[n, 1, 1] = -0.3 + 0.05 * n
    data[n, 0, 1] = 1.0
    data[n, 1, 0] = 1.0

h_k = qten.Tensor(data=data, dims=(k_space, band_space, band_space))
```

### 2. Apply A Symmetry With `bandaffine`

```python
from dataclasses import dataclass

import sympy as sy
import torch

import qten
from qten.bands import bandaffine
from qten.geometries import Lattice, Offset, PeriodicBoundary
from qten.pointgroups import AffineTransform
from qten.symbolics import HilbertSpace, U1Basis, brillouin_zone


@dataclass(frozen=True)
class Orb:
    name: str


lattice = Lattice(
    basis=sy.ImmutableDenseMatrix.eye(2),
    boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
    unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
)
k_space = brillouin_zone(lattice.dual)

r_x = Offset(rep=sy.ImmutableDenseMatrix([sy.Rational(1, 2), 0]), space=lattice.affine)
r_y = Offset(rep=sy.ImmutableDenseMatrix([0, sy.Rational(1, 2)]), space=lattice.affine)
band_space = HilbertSpace.new([U1Basis.new(r_x, Orb("p")), U1Basis.new(r_y, Orb("p"))])

data = torch.zeros((k_space.dim, 2, 2), dtype=torch.complex128)
for n in range(k_space.dim):
    data[n, 0, 0] = 0.2 + 0.1 * n
    data[n, 1, 1] = -0.3 + 0.05 * n
    data[n, 0, 1] = 1.0
    data[n, 1, 0] = 1.0

h_k = qten.Tensor(data=data, dims=(k_space, band_space, band_space))

x, y = sy.symbols("x y")
c4 = AffineTransform(
    irrep=sy.ImmutableDenseMatrix([[0, -1], [1, 0]]),
    axes=(x, y),
    offset=Offset(rep=sy.ImmutableDenseMatrix([0, 0]), space=lattice.affine),
    basis_function_order=1,
)

h_k_rot = bandaffine(c4, h_k, opt="both")
print(h_k_rot.data.shape)
```

`opt="left"` and `opt="right"` are useful when you only want to transform one side of the operator.

### 3. Fold Bands Into A Larger Real-Space Cell

```python
from dataclasses import dataclass

import sympy as sy
import torch

import qten
from qten.bands import bandfold
from qten.geometries import BasisTransform, Lattice, Offset, PeriodicBoundary
from qten.symbolics import HilbertSpace, U1Basis, brillouin_zone


@dataclass(frozen=True)
class Orb:
    name: str


lattice = Lattice(
    basis=sy.ImmutableDenseMatrix([[1]]),
    boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(4)),
    unit_cell={"r": sy.ImmutableDenseMatrix([0])},
)
k_space = brillouin_zone(lattice.dual)

r0 = Offset(rep=sy.ImmutableDenseMatrix([0]), space=lattice)
band_space = HilbertSpace.new([U1Basis.new(r0, Orb("s"))])

data = torch.arange(k_space.dim, dtype=torch.float64).reshape(k_space.dim, 1, 1)

h_k = qten.Tensor(data=data, dims=(k_space, band_space, band_space))

M = sy.ImmutableDenseMatrix([[2]])
h_folded = bandfold(BasisTransform(M), h_k, opt="both")

print(h_folded.dims[0])  # new MomentumSpace for the folded Brillouin zone
print(h_folded.data.shape)
```

### 4. Plot The Band Structure

```python
from dataclasses import dataclass

import sympy as sy
import torch

import qten
from qten.geometries import Lattice, Offset, PeriodicBoundary
from qten.symbolics import HilbertSpace, U1Basis, brillouin_zone


@dataclass(frozen=True)
class Orb:
    name: str


lattice = Lattice(
    basis=sy.ImmutableDenseMatrix.eye(2),
    boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(2, 2)),
    unit_cell={"r": sy.ImmutableDenseMatrix([0, 0])},
)
k_space = brillouin_zone(lattice.dual)

r_x = Offset(rep=sy.ImmutableDenseMatrix([sy.Rational(1, 2), 0]), space=lattice.affine)
r_y = Offset(rep=sy.ImmutableDenseMatrix([0, sy.Rational(1, 2)]), space=lattice.affine)
band_space = HilbertSpace.new([U1Basis.new(r_x, Orb("p")), U1Basis.new(r_y, Orb("p"))])

data = torch.zeros((k_space.dim, 2, 2), dtype=torch.complex128)
for n in range(k_space.dim):
    data[n, 0, 0] = 0.2 + 0.1 * n
    data[n, 1, 1] = -0.3 + 0.05 * n
    data[n, 0, 1] = 1.0
    data[n, 1, 0] = 1.0

h_k = qten.Tensor(data=data, dims=(k_space, band_space, band_space))

fig = h_k.plot(
    "bandstructure",
    backend="plotly",
    title="Band structure",
    show=False,
)
fig.show()
```

## IO

QTen exposes a small versioned pickle-based storage helper at the package root as `qten.io`.

```python
import qten

qten.io.iodir(".data")
qten.io.env("demo")

payload = {
    "label": "trial-run",
    "values": [1, 2, 3],
}

version = qten.io.save(payload, "experiment")
print(version)  # 1

rows = qten.io.list_saved("experiment")
print(rows[-1]["version"])   # 1
print(rows[-1]["size_mib"])  # file size in MiB

latest = qten.io.load("experiment", -1)
print(latest)
```

Use environments to separate different runs or projects:

```python
import qten

qten.io.iodir(".data")

qten.io.env("train")
qten.io.save({"loss": 0.12}, "metrics")

qten.io.env("eval")
qten.io.save({"accuracy": 0.98}, "metrics")

qten.io.env("train")
train_metrics = qten.io.load("metrics", -1)
```

## Notes

- Plotting supports both `plotly` and `matplotlib` backends.
- `Tensor.plot("heatmap")`, `Tensor.plot("spectrum")`, and `Tensor.plot("bandstructure")` are available for common workflows.
- The most important tensor convention in QTen is that axis metadata lives in `dims`, so tensor operations preserve physical meaning instead of only raw shape.
