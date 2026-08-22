"""
Point-group symmetry helpers.

This package provides compact constructors and symbolic representations for
finite point operations, including single abelian generators, full crystallographic
point groups, and their induced actions on polynomial bases.

Dispatch
--------
[`pointgroup`][qten.pointgroups._pointgroups.pointgroup] chooses the object from
the query string, not from whether the group is abelian:

- Affine queries such as `c4-xy:xy` or `m-xyz:yz` return a single
  [`PointGroupElement`][qten.pointgroups.elements.PointGroupElement] with no
  character table. Projection then uses
  [`point_group_column_symmetrize`][qten.pointgroups.ops.point_group_column_symmetrize]
  on a [`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr] (phase sectors).
- Named Hermann-Mauguin or Schoenflies symbols such as `4`, `C4v`, or `-43m`
  return a [`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] with
  packaged ordinary characters, and spinor characters when the axes are `xyz`.
  Passing that group to `point_group_column_symmetrize` uses character projectors.
- A named suffix such as `C4v-xy` projects the 3D generators onto a coordinate
  plane. Ordinary characters stay attached; the spinor table is dropped.

Core exports
------------
[`pointgroup`][qten.pointgroups._pointgroups.pointgroup] parses a compact query
string into a symmetry object.
[`PointGroupElement`][qten.pointgroups.elements.PointGroupElement] is a single
exact linear generator acting on coordinate functions.
[`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr] is its affine
extension with translation.
[`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] is a finite group
closed under one or more packaged generators and optional character-table data.
[`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] labels polynomial
basis functions in a representation sector.
[`FiniteIrrepSector`][qten.pointgroups.ops.FiniteIrrepSector] labels ordinary
finite-group irrep sectors.
[`SpinfulPhaseSector`][qten.pointgroups.ops.SpinfulPhaseSector] labels
double-valued abelian phases for spin-1/2 representations.
[`JointSpinfulPhaseSector`][qten.pointgroups.ops.JointSpinfulPhaseSector]
labels simultaneous spinorial phases of a commuting family.
[`SpinorIrrepSector`][qten.pointgroups.ops.SpinorIrrepSector] labels finite
double-valued sectors obtained from packaged spgrep character data.
[`SymmetryDegeneracy`][qten.pointgroups.ops.SymmetryDegeneracy] is a typed copy
index used when a projected sector occurs more than once.
[`point_group_column_symmetrize`][qten.pointgroups.ops.point_group_column_symmetrize]
projects columns onto abelian phase sectors or finite-group irreps.
[`joint_point_group_column_symmetrize`][qten.pointgroups.ops.joint_point_group_column_symmetrize]
projects onto simultaneous sectors of a commuting family.
[`point_group_operator_symmetrize`][qten.pointgroups.ops.point_group_operator_symmetrize]
averages spinless or spinful operators without requiring irrep character data.

Joint-basis helper
------------------
[`joint_point_group_basis`][qten.pointgroups.ops.joint_point_group_basis]
constructs a common eigen-basis for compatible commuting abelian operators.
"""

from ._pointgroups import pointgroup as pointgroup

from .elements import (
    PointGroupElement as PointGroupElement,
    PointGroupOpr as PointGroupOpr,
)
from .basis import PointGroupBasis as PointGroupBasis
from .finite import FinitePointGroup as FinitePointGroup
from .ops import (
    FiniteIrrepSector as FiniteIrrepSector,
    JointSpinfulPhaseSector as JointSpinfulPhaseSector,
    SpinorIrrepSector as SpinorIrrepSector,
    SpinfulPhaseSector as SpinfulPhaseSector,
    SymmetryDegeneracy as SymmetryDegeneracy,
    get_direct_transform as get_direct_transform,
    hilbert_repr as hilbert_repr,
    joint_point_group_basis as joint_point_group_basis,
    joint_point_group_column_symmetrize as joint_point_group_column_symmetrize,
    point_group_column_symmetrize as point_group_column_symmetrize,
    point_group_operator_symmetrize as point_group_operator_symmetrize,
    spinful_hilbert_opr_repr as spinful_hilbert_opr_repr,
    spinful_transform_basis as spinful_transform_basis,
)
