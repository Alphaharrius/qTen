# QTen

A torch-based tensor library for quantum-related computations.

## Installation

QTen requires Python 3.11 or newer.

Install PyTorch first. For CPU or CUDA-specific builds, use the official
PyTorch installation command for your platform.

Then install QTen:

```bash
pip install qten
```

If you want plotting support:

```bash
pip install qten-plots
```

If you use `uv`:

```bash
uv add torch
uv add qten
uv add qten-plots
```

For development from this repository:

```bash
uv sync --extra cpu --group dev
```

If you want a CUDA-specific PyTorch build in this repository, choose one of the
defined extras:

```bash
uv sync --extra cu126 --group dev
uv sync --extra cu128 --group dev
uv sync --extra cu129 --group dev
uv sync --extra cu130 --group dev
```

## Quick Start

The package root exports the main tensor helpers: `Tensor`, `eye`, `zeros`,
`ones`, and `set_precision`.

For simple examples, the easiest axis type is `IndexSpace`.

## Tensors

Create a tensor with named dimensions:

```python
import torch

import qten
from qten.symbolics import IndexSpace

rows = IndexSpace.linear(2)
cols = IndexSpace.linear(3)

x = qten.Tensor(
    data=torch.arange(6, dtype=torch.float64).reshape(2, 3),
    dims=(rows, cols),
)

print(x.rank())      # 2
print(x.data.shape)  # torch.Size([2, 3])
print(x.dims)        # (IndexSpace(size=2), IndexSpace(size=3))
```

Create standard tensors on the same space:

```python
import torch

import qten
from qten.symbolics import IndexSpace

space = IndexSpace.linear(2)

a = qten.Tensor(
    data=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    dims=(space, space),
)

identity = qten.eye((space, space))
filled = qten.ones((space, space))
empty = qten.zeros((space, space))
product = a @ identity

print(product.data)
print(filled.data.shape)  # torch.Size([2, 2])
print(empty.data.shape)   # torch.Size([2, 2])
```

If you want QTen to use single precision instead of double precision:

```python
import qten

qten.set_precision("32")
```

## Lattices

Build a square lattice:

```python
import sympy as sy
import torch

from qten.geometries import Lattice

lattice = Lattice(
    basis=sy.ImmutableDenseMatrix([[1, 0], [0, 1]]),
    shape=(3, 3),
)

print(lattice.shape)                 # (3, 3)
print(len(lattice.cartes()))         # 9
print(lattice.cartes(torch.Tensor))  # Cartesian coordinates
```

### Boundary Conditions

Finite `Lattice` objects use periodic boundary conditions. When you pass
`shape=(Lx, Ly, ...)`, QTen treats lattice indices modulo those extents, so a site
at `(Lx, y)` is identified with `(0, y)`.

That means:

- `shape=(3, 3)` creates a `3 x 3` torus, not an open patch.
- Distances are measured using the nearest periodic image.
- `shape` is shorthand for a diagonal `PeriodicBoundary`.

If you want to specify the periodic identifications explicitly, pass
`boundaries=PeriodicBoundary(...)`:

```python
import sympy as sy

from qten.geometries import Lattice, PeriodicBoundary

lattice = Lattice(
    basis=sy.ImmutableDenseMatrix([[1, 0], [0, 1]]),
    boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix([[3, 0], [0, 3]])),
)

print(lattice.shape)  # (3, 3)
```

For a unit cell with more than one site:

```python
import sympy as sy

from qten.geometries import Lattice

lattice = Lattice(
    basis=sy.ImmutableDenseMatrix([[1, 0], [0, 1]]),
    shape=(2, 2),
    unit_cell={
        "A": sy.ImmutableDenseMatrix([0, 0]),
        "B": sy.ImmutableDenseMatrix([sy.Rational(1, 2), sy.Rational(1, 2)]),
    },
)

print(lattice.shape)         # (2, 2)
print(len(lattice.cartes())) # 8
```

## Plotting

Plotting requires `qten-plots`.

```python
import torch

import qten
from qten.symbolics import IndexSpace

rows = IndexSpace.linear(2)
cols = IndexSpace.linear(3)

x = qten.Tensor(
    data=torch.arange(6, dtype=torch.float64).reshape(2, 3),
    dims=(rows, cols),
)

fig = x.plot("heatmap", show=False)
fig.show()
```

You can also plot a lattice:

```python
import sympy as sy

from qten.geometries import Lattice

lattice = Lattice(
    basis=sy.ImmutableDenseMatrix([[1, 0], [0, 1]]),
    shape=(3, 3),
)

fig = lattice.plot("structure", show=False)
fig.show()
```

## IO

QTen includes a small versioned pickle-based storage helper at `qten.io`.

```python
import qten

qten.io.iodir(".data")
qten.io.env("demo")

qten.io.save({"loss": 0.12}, "metrics")
qten.io.save({"loss": 0.08}, "metrics")

print(qten.io.load("metrics", -1))
print(qten.io.list_saved("metrics"))
```

## Worked Example: Dirac Semi-Metal On The Honeycomb Lattice

The example does six things:

1. Builds the real-space Dirac semi-metal Hamiltonian on a honeycomb lattice.
2. Fourier transforms it into Bloch form.
3. Folds the bands into a larger real-space cell.
4. Transforms the blocked Hamiltonian by `C6` and compares the result.
5. Builds the ground-state correlation matrix and restricts it to a local region.
6. Extracts a local `C6`-symmetric basis from the restricted eigenvectors.

### 1. Imports And Lattice Setup

Start by defining the honeycomb lattice, the `C6` symmetry, and the two primitive translations.

```python
import sympy as sy
import qten
import qten.ops as Q

from qten.bands import bandfillings, bandfold, bandtransform
from qten.geometries import BasisTransform, Lattice, Offset
from qten.phys import FFObservable
from qten.pointgroups import AbelianOpr, pointgroup
from qten.symbolics import FuncOpr, HilbertSpace, U1Basis, brillouin_zone
from qten_plots.plottables import PointCloud


triangular = sy.ImmutableMatrix(
    [
        [sy.sqrt(3) / 2, 0],
        [-sy.Rational(1, 2), 1],
    ]
)

honeycomb = Lattice(
    basis=triangular,
    unit_cell={
        "a": sy.ImmutableMatrix([sy.Rational(1, 3), sy.Rational(2, 3)]),
        "b": sy.ImmutableMatrix([sy.Rational(2, 3), sy.Rational(1, 3)]),
    },
    shape=(96, 96),
)

c6 = pointgroup("c6-xy:xy")
T_c6 = AbelianOpr(c6)

a1, a2 = honeycomb.basis_vectors()
R_a1 = FuncOpr(Offset, lambda r: r + a1)
R_a2 = FuncOpr(Offset, lambda r: r + a2)
```

What this does:

- `triangular` is the triangular Bravais lattice underlying the honeycomb lattice.
- `unit_cell` adds the two sublattice sites `a` and `b`.
- `c6` is the sixfold point-group generator.
- `T_c6 = AbelianOpr(c6)` is the affine version of that rotation. At this stage it only contains the linear `C6` action, centered at the canonical origin.
- `R_a1` and `R_a2` are translation operators that shift an orbital by one primitive lattice vector.

### 2. Define The Two Bloch Basis Orbitals

Use one orbital on each sublattice site, both transforming in the same `C6` basis channel.

```python
psi_a = U1Basis.new(honeycomb.at("a"), c6.basis_table[1])
psi_b = U1Basis.new(honeycomb.at("b"), c6.basis_table[1])
```

Here:

- `honeycomb.at("a")` and `honeycomb.at("b")` pick the two sites in the reference unit cell.
- `c6.basis_table` is a lookup table from `C6` eigenvalue to an `AbelianBasis`.
- `c6.basis_table[1]` picks the eigen-basis with eigenvalue `1`, which is the trivial `C6` character. In this case it is the constant basis function `e`.
- Using that basis means the internal orbital itself is `C6` invariant; the nontrivial symmetry action in this example comes from how the rotation moves the lattice position, not from an extra internal orbital phase.

### 3. Build The Dirac Semi-Metal Hamiltonian In Real Space

Now add the nearest-neighbor hoppings that define the honeycomb Dirac Hamiltonian.

```python
t = sy.Number(1)

dirac = FFObservable()
dirac.add_bond(-t, psi_a, psi_b)         # A(R) <-> B(R)
dirac.add_bond(-t, psi_a, R_a2 @ psi_b)  # A(R) <-> B(R + a2)
dirac.add_bond(-t, psi_b, R_a1 @ psi_a)  # B(R) <-> A(R + a1)

harmonic = dirac.to_tensor()
```

Interpretation:

- `FFObservable()` collects quadratic fermion terms.
- Each `add_bond` adds one hopping channel.
- `harmonic` is the real-space tensor form of the tight-binding Hamiltonian.

The three bonds are enough because the remaining nearest-neighbor hoppings are generated automatically by translation and Hermitian completion in the observable-to-tensor construction. In other words, you specify one representative set of bonds in the unit cell, not every bond in the full finite lattice.

This is the standard nearest-neighbor honeycomb model, so its Bloch spectrum contains Dirac cones.

### 4. Build The Bloch Hamiltonian

Create the Bloch basis, the Brillouin-zone momentum space, and the Fourier transform.

```python
bloch_space = HilbertSpace.new([psi_a, psi_b])
k_space = brillouin_zone(honeycomb.dual)

F = Q.fourier_transform(k_space, bloch_space, harmonic.dims[0])
bloch = F @ harmonic @ F.h(-2, -1)
```

What happens here:

- `bloch_space` is the two-orbital basis inside one unit cell.
- `k_space` is the sampled momentum space of the finite lattice.
- `F` maps the real-space basis into the Bloch basis.
- `bloch` is the momentum-space Hamiltonian with dims `(MomentumSpace, HilbertSpace, HilbertSpace)`.

This is the first place where the model becomes a band Hamiltonian in the usual condensed-matter sense: for each crystal momentum `k`, `bloch[k]` is a finite matrix acting on the orbital content of one unit cell.

### 5. Block The Hamiltonian Into A Larger Cell

Here the real-space cell is doubled in both primitive directions.

```python
blocking = BasisTransform(sy.ImmutableMatrix.eye(2) * 2)
blocked_bloch = bandfold(blocking, bloch, opt="both")
```

This does a band folding operation:

- the real-space unit cell becomes larger,
- the Brillouin zone becomes smaller,
- the number of Bloch bands increases accordingly.

Using `opt="both"` means the basis change is applied consistently on the bra and ket sides of the operator, so the result is still a Hamiltonian in the blocked basis rather than a mixed-basis object.

### 6. Transform The Blocked Hamiltonian By `C6`

Apply the sixfold symmetry operator to the blocked Bloch Hamiltonian.

```python
c6_bloch = bandtransform(T_c6, blocked_bloch)
```

To compare the transformed Hamiltonian with the original one:

```python
_ = (blocked_bloch - c6_bloch).plot("bandstructure")
```

If the construction is symmetry-compatible, this difference should be small or vanish up to the expected numerical and representation conventions.

This comparison is worth doing after blocking because folding changes the band labeling and basis organization. Checking `bandtransform(T_c6, blocked_bloch)` against `blocked_bloch` is a direct sanity check that the blocked model still carries the intended `C6` symmetry.

### 7. Plot The Blocked Band Structure

Plot the blocked Hamiltonian itself:

```python
_ = blocked_bloch.plot("bandstructure")
```

This gives the folded band structure after enlarging the unit cell.

### 8. Find The Ground-State Occupied Subspace

Fill half the bands to form the free-fermion ground state.

```python
gs = bandfillings(blocked_bloch, 0.5)
P_gs = gs @ gs.h(-2, -1)
C_gs = 1.0 * qten.eye(P_gs.dims) - P_gs
```

Meaning:

- `gs` contains the occupied Bloch eigenvectors.
- `P_gs` is the projector onto the occupied subspace.
- `C_gs` is the single-particle correlation matrix for the ground state.

For a free-fermion ground state, this is the main object you need for entanglement and local symmetry analysis.

The filling fraction `0.5` means half filling of the total one-particle spectrum. `bandfillings` diagonalizes the Hamiltonian across all sampled momenta, finds the global filling threshold, and keeps the states below that threshold. For this blocked Dirac model, that gives the half-filled ground state. The relation

$$
C_\mathrm{gs} = I - P_\mathrm{gs}
$$

is the correlation matrix convention used in this workflow: it stores the complementary one-particle projector associated with the chosen filling.

### 9. Choose A Local Region In Real Space

Pick a cluster of sites near a chosen center.

```python
region = Q.nearest_sites(honeycomb, 2 * a1 + 2 * a2, 3)
```

The third argument is `n_nearest`, the number of distinct distance shells to include around the chosen center.

- `1` means only the nearest-distance shell,
- `2` means the first two distinct shells,
- `3` means the first three distinct shells.

So this call does not mean “take exactly 3 sites” or “use radius 3”. It means “collect every lattice site whose distance from `2 * a1 + 2 * a2` lies in one of the first three distinct distance shells”.

You can inspect that region on the lattice:

```python
_ = honeycomb.plot("structure", highlights=[PointCloud(region)])
```

This is useful to confirm that the restricted subsystem is centered where you expect.

The center `2 * a1 + 2 * a2` matters later when constructing the local symmetry action. A rotation is not determined only by “rotate by 60 degrees”; it also needs a center. The local symmetry operator must be centered on the same point as the local patch if you want the patch and its restricted modes to transform into themselves.

### 10. Build The Restricted Correlation Matrix

Construct the restriction operator and then compress the global correlation matrix into the chosen region.

```python
R = Q.region_restrict(C_gs, region)
C_R = (R.h(-2, -1) @ C_gs @ R).mean(dim=0)
```

Here:

- `R` maps the full lattice Hilbert space down to the selected region.
- `C_R` is the restricted correlation matrix seen by that region.
- `.mean(dim=0)` combines the contributions from all crystal momenta to reconstruct the real-space correlation matrix of the chosen subsystem. Physically, the local reduced state is built from the full occupied Fermi sea, not from any single momentum sector, so the restricted correlator is obtained only after summing or averaging over all `k`.

The mean over momentum is what turns the translationally organized band description into the correlation matrix of one finite real-space patch. Without that step you would still have a momentum-resolved family of partial contributions rather than the physical local correlator of the region.

You can inspect `C_R` directly:

```python
C_R
```

And plot it:

```python
_ = C_R.plot("heatmap")
_ = C_R.plot("spectrum")
```

The heatmap shows the local correlation structure, and the spectrum is the entanglement-spectrum-style eigenvalue distribution of the restricted free-fermion state.

### 11. Diagonalize The Restricted Correlation Matrix

Get eigenvalues and eigenvectors of the restricted matrix.

```python
v, u = qten.linalg.eigh(C_R)
```

At this point:

- `v` contains the eigenvalues of `C_R`,
- `u` contains the corresponding local eigenmodes as columns.

These local modes are the starting point for symmetry-adapted basis construction.

### 12. Build The Local `C6` Action On The Restricted Hilbert Space

Center the symmetry at the same point used for the local region and build its matrix representation on the local basis.

```python
g_c6 = Q.hilbert_opr_repr(T_c6.fixpoint_at(2 * a1 + 2 * a2), u.dims[0])
```

This gives the local unitary representation of the sixfold rotation acting on the restricted Hilbert space.

The `.fixpoint_at(2 * a1 + 2 * a2)` part is essential. `T_c6` by itself is only the affine rotation with its default center. `fixpoint_at(r)` changes the affine translation part so that the point `r` is invariant under the operation. In formula form, if the affine action is

```text
x -> R x + t
```

then fixing the point `r` means choosing `t` so that

```text
R r + t = r
```

or equivalently

```text
t = (I - R) r
```

That is exactly what `fixpoint_at(...)` does internally.

Why this matters here:

- the local region was chosen around `2 * a1 + 2 * a2`,
- the restricted Hilbert space `u.dims[0]` is the Hilbert space of that local patch,
- the local `C6` operator should rotate around the same center as the patch.

If you skipped `fixpoint_at(...)`, you would generally be representing a rotation around the wrong point, so the local subspace would not transform the way you intend.

### 13. Get A Local `C6`-Symmetric Basis Directly

Now symmetry-adapt the first few local modes.

```python
u_sym = Q.abelian_column_symmetrize(
    T_c6.fixpoint_at(2 * a1 + 2 * a2),
    u[:, :3],
)
```

This is the simplest high-level way to get a `C6`-adapted basis from the local unitary structure.

Conceptually, this takes a set of columns spanning a small local subspace and rotates them into columns with definite abelian symmetry character under `C6`. That is usually the cleanest entry point if your goal is “give me local modes with good symmetry labels”.

You can visualize the resulting basis vectors:

```python
_ = u_sym.plot("column_scatter", default_size=15)
```

And confirm the transformed column space:

```python
T_c6 @ u_sym.dims[1]
```

### 14. Get The Local `C6`-Symmetric Basis Manually From The Unitary

If you want to see the linear algebra explicitly, you can also do the symmetry adaptation by hand inside the selected subspace.

First choose a working subspace:

```python
w = u[:, :3]
```

Project the local `C6` operator into that subspace:

```python
g_c6_w = w.h(-2, -1) @ g_c6 @ w
```

Now diagonalize the symmetry operator inside the subspace:

```python
v_c6, u_c6 = qten.linalg.eig(g_c6_w)
print(v_c6.data)
```

Finally rotate the original basis `w` by the eigenvectors of the projected symmetry operator:

```python
w_sym = w @ u_c6
```

This `w_sym` is a local `C6` eigenbasis extracted from the unitary itself.

The reason this works is standard subspace linear algebra:

- `w` is a basis for the subspace you care about,
- `g_c6_w = w^† g_c6 w` is the symmetry operator written in that subspace basis,
- diagonalizing `g_c6_w` finds the combinations of columns of `w` that carry definite `C6` eigenvalues,
- multiplying back by `w` lifts those symmetry eigenvectors to the full local Hilbert space.

So `w_sym` and `u_sym` are solving the same problem in two different ways:

- `u_sym` uses the high-level helper,
- `w_sym` constructs the symmetry-adapted basis explicitly from the projected unitary.

To compare before and after:

```python
_ = w.plot("column_scatter")
_ = w_sym.plot("column_scatter")
```

This manual route is useful when you want to inspect the symmetry representation explicitly instead of using the higher-level helper.

## Notes

- QTen tensors always carry axis metadata in `dims`.
- `IndexSpace` is the simplest place to start.
- Geometry, symbolic, point-group, and band tools are available from
  submodules when you need them.
