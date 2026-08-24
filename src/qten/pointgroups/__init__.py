r"""
Point-group symmetry helpers.

Two rules decide spin and geometry:

1. Spin follows the Hilbert space. If a basis already carries
   [`Spin`][qten.phys.spin.Spin], projection uses the \(SU(2)\) lift. If it does
   not, the group is ordinary. There is no `.with_spin`.
2. Geometry is written in the constructor. `plane=` / `axis=` / `spin=` belong
   on that one `pointgroup(...)` call. Move the origin later with `fixpoint=`
   on the single-group helpers, or `fixpoint_at` on a `PointGroupOpr`. Joint
   projection has no `fixpoint=`; center each operator first.

[`pointgroup`][qten.pointgroups.pointgroup] chooses the object
from the query, not from whether the group is abelian.

How to construct
----------------
Named crystallographic symbols return a
[`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] with ordinary
Bilbao characters and spinor characters from QTen's lift. Affine queries such
as `c4-xy:xy` return one
[`PointGroupElement`][qten.pointgroups.elements.PointGroupElement].

```python
from qten.pointgroups import pointgroup

td = pointgroup("Td")                   # tetrahedral; rotation3 is the spatial matrix
c4v = pointgroup("C4v", plane="xy")     # 2x2 spatial, 3D C4v kept for spin
c3v = pointgroup("C3v", axis=(1, 1, 1)) # still 3D; C3 about [111]
mirror = pointgroup("Cs", plane="x")    # 1D spatial [[-1]], rotation3 = σ_yz
c4 = pointgroup("C4", plane="xy")       # fourfold in xy
flavor = pointgroup("C4v", spin="trivial")  # u(g)=I; define-time only
```

`C4v-xy` is the same as `plane="xy"`. A 2D or 1D custom matrix group with no
`rotation3` cannot lift spin: rewrite the constructor with `plane=` / `axis=`,
do not pad the small matrix later.

How to use with spin
--------------------
Put `Spin` on the basis, then symmetrize. The group is not told twice.

```python
from qten.phys import Spin
from qten.pointgroups import (
    PointGroupOpr,
    hilbert_repr,
    point_group_column_symmetrize,
    point_group_operator_symmetrize,
)
from qten.symbolics import U1Basis

# diamond, C_R, seed, center, and space come from the surrounding model
A_up = U1Basis.new(diamond.at("A"), Spin.up)
td = pointgroup("-43m")
C_sym = point_group_operator_symmetrize(td, C_R, fixpoint=center)
w = point_group_column_symmetrize(td, seed, fixpoint=center)
D = hilbert_repr(PointGroupOpr(td.elements()[0]).fixpoint_at(center), space)
```

Mathematics
-----------
On a spinless space, \(D(g)=D_{\mathrm{orb}}(g)\). On a spinful space
\[
D(g)=D_{\mathrm{orb}}(g)\otimes u(g),
\]
where \(u(g)\in SU(2)\) is the principal lift of \(R_+(g)=(\det R(g))\,R(g)\).
The section is a 2-cocycle,
\(u(g)u(h)=\omega(g,h)\,u(gh)\) with \(\omega(g,h)\in\{\pm 1\}\).

Finite groups use the character projector
\[
P^\mu=\frac{d_\mu}{|G|}\sum_{g\in G}\chi^\mu(g)^*D(g).
\]
Ordinary (linear) \(\chi\) if the space has no `Spin`; projective spinor
\(\chi\) if it does, unless the group was built with `spin="trivial"`.
Class-wise spinor rows vanish on non-\(\omega\)-regular classes; the
projector uses the element-wise hat-table section, not those averages.

A cyclic [`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr] of
spatial order \(n\) is the abelian case of the same formula,
\[
P_\zeta=\frac{1}{N}\sum_{k=0}^{N-1}\zeta^{-k}D(g)^k,\qquad\zeta^N=1,
\]
with \(N=n\) spinless and \(N=2n\) spinful (\(u(2\pi)=-I\)). Operator
twirling needs no \(\chi\):
\(A_G=|G|^{-1}\sum_g D(g)AD(g)^\dagger\). The sign of each lift cancels
between \(D(g)\) and \(D(g)^\dagger\).

Joint spinful projection of several operators needs `group=` the same
already-defined \(G\).

Dispatch
--------
- Affine queries (`c4-xy:xy`, `m-xyz:yz`) → one element, phase-sector
  projection after wrapping in `PointGroupOpr`.
- Named symbols (`4mm`, `C4v`, `-43m`) → finite group, packaged ordinary
  table plus QTen spinor table.
- `plane="xy"` or suffix `-xy` shrinks spatial matrices only. A group that
  is not faithful on that plane (for example `4/m`) raises.
- `axis=(1,1,1)` reorients a 3D group; ordinary class labels follow
  conjugation. `plane=(1,1,1)` is a 2D cut of that reoriented group, still
  with the 3D `rotation3`.

Core exports
------------
[`pointgroup`][qten.pointgroups.pointgroup] parses a compact query
or named symbol.
[`PointGroupElement`][qten.pointgroups.elements.PointGroupElement] is one
linear operation (`irrep` on the model, `rotation3` in \(O(3)\)).
[`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr] adds a translation
and `fixpoint_at`.
[`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] is the closure
of generators, with ordinary / spinor class tables.
[`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] labels polynomial
sectors.
[`FiniteIrrepSector`][qten.pointgroups.sectors.FiniteIrrepSector] /
[`SpinorIrrepSector`][qten.pointgroups.sectors.SpinorIrrepSector] /
[`SpinfulPhaseSector`][qten.pointgroups.sectors.SpinfulPhaseSector] /
[`JointSpinfulPhaseSector`][qten.pointgroups.sectors.JointSpinfulPhaseSector]
label projected columns.
[`SymmetryDegeneracy`][qten.pointgroups.sectors.SymmetryDegeneracy] tags repeated
copies of the same sector.
[`hilbert_repr`][qten.pointgroups.ops.hilbert_repr] assembles \(D(g)\).
[`spinful_hilbert_opr_repr`][qten.pointgroups.ops.spinful_hilbert_opr_repr] /
[`spinful_transform_basis`][qten.pointgroups.ops.spinful_transform_basis]
are the spinful fast path used by that assembler.
[`point_group_column_symmetrize`][qten.pointgroups.ops.point_group_column_symmetrize]
projects columns.
[`point_group_operator_symmetrize`][qten.pointgroups.ops.point_group_operator_symmetrize]
twirls an operator (no \(\chi\) needed).
[`joint_point_group_column_symmetrize`][qten.pointgroups.ops.joint_point_group_column_symmetrize]
and [`joint_point_group_basis`][qten.pointgroups.ops.joint_point_group_basis]
handle a commuting family.
"""

from ._pointgroups import pointgroup as pointgroup

from .elements import (
    PointGroupElement as PointGroupElement,
    PointGroupOpr as PointGroupOpr,
)
from .basis import PointGroupBasis as PointGroupBasis
from .finite import FinitePointGroup as FinitePointGroup
from .sectors import (
    FiniteIrrepSector as FiniteIrrepSector,
    JointSpinfulPhaseSector as JointSpinfulPhaseSector,
    SpinorIrrepSector as SpinorIrrepSector,
    SpinfulPhaseSector as SpinfulPhaseSector,
    SymmetryDegeneracy as SymmetryDegeneracy,
)
from .ops import (
    get_direct_transform as get_direct_transform,
    hilbert_repr as hilbert_repr,
    joint_point_group_basis as joint_point_group_basis,
    joint_point_group_column_symmetrize as joint_point_group_column_symmetrize,
    point_group_column_symmetrize as point_group_column_symmetrize,
    point_group_operator_symmetrize as point_group_operator_symmetrize,
    spinful_hilbert_opr_repr as spinful_hilbert_opr_repr,
    spinful_transform_basis as spinful_transform_basis,
)
