r"""
Physics-facing labels and operator assembly.

This package sits between [`qten.symbolics`][qten.symbolics] and
[`qten.linalg`][qten.linalg]. Use it for two things:

- free-fermion bonds that become tensors
- spin-1/2 labels that decide whether a point group must lift to \(SU(2)\)

Spin and point groups
---------------------
[`Spin`][qten.phys.spin.Spin] is a typed irrep inside
[`U1Basis`][qten.symbolics.hilbert_space.U1Basis]. Putting `Spin.up` /
`Spin.down` on a site is the definition of a spinful Hilbert space. Point-group
code later asks [`contains_spin`][qten.phys.spin.contains_spin]; you do not
call `.with_spin` on the group.

Electron spin always lives in \(\mathbb{C}^2\). The spatial model can be 1D,
2D, or 3D. The map from a point operation to spin is the SU(2) lift of that
operation's stored 3D rotation `rotation3`, not a padded copy of the small
spatial matrix.

```python
from qten.phys import Spin, su2_of_point_group
from qten.pointgroups import pointgroup
from qten.symbolics import U1Basis

psi = U1Basis.new(site, Spin.up)   # this basis state is spinful
g = pointgroup("C4v", plane="xy")  # geometry is fixed here
u = su2_of_point_group(g.elements()[1])  # 2x2 SU(2) factor
```

Inspect a lift when you need the matrix. Projection and representation
assembly live in [`qten.pointgroups`][qten.pointgroups]:
[`hilbert_repr`][qten.pointgroups.ops.hilbert_repr] builds
\(D(g)=D_{\mathrm{orb}}(g)\otimes u(g)\) on a spinful space.

Rare exception, still at construction: `pointgroup("C4v", spin="trivial")`
sets \(u(g)=I\) (flavor spin / no SOC). Do not redefine an already built group.

Spin APIs
---------
- [`Spin`][qten.phys.spin.Spin] — \(m_s=\pm 1/2\) labels `Spin.up` / `Spin.down`
- [`as_spin`][qten.phys.spin.as_spin] — `Spin` or leftover ``"up"`` / ``"down"`` strings
- [`contains_spin`][qten.phys.spin.contains_spin] — Hilbert space already has a spin-1/2 label
- [`su2_of_point_group`][qten.phys.spin.su2_of_point_group] — principal lift of `rotation3`
- [`su2_numeric`][qten.phys.spin.su2_numeric] — the same factor as complex floats
- [`expand_spin`][qten.phys.spin.expand_spin] — \(u(g)|s\rangle\) in the \(\{\uparrow,\downarrow\}\) basis
- [`proper_rotation_matrix`][qten.phys.spin.proper_rotation_matrix] — \(O(3)\to SO(3)\) before the lift
- [`su2_from_so3`][qten.phys.spin.su2_from_so3] — lift a proper \(3\times 3\) matrix
- [`SU2_SECTION_CONVENTION`][qten.phys.spin.SU2_SECTION_CONVENTION] — `qten-su2-principal-v1`

Free-fermion assembly
---------------------
- [`Bond`][qten.phys.Bond] stores a weighted directed transition between two
  [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] states.
- [`FFObservable`][qten.phys.FFObservable] accumulates bond terms and converts
  them into a rank-2 [`Tensor`][qten.linalg.tensors.Tensor].
"""

from ._bonds import Bond as Bond
from ._ff_observables import FFObservable as FFObservable
from .spin import (
    SU2_SECTION_CONVENTION as SU2_SECTION_CONVENTION,
    Spin as Spin,
    as_spin as as_spin,
    contains_spin as contains_spin,
    expand_spin as expand_spin,
    proper_rotation_matrix as proper_rotation_matrix,
    su2_from_so3 as su2_from_so3,
    su2_numeric as su2_numeric,
    su2_of_point_group as su2_of_point_group,
)
