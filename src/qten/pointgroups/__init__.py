"""
Point-group symmetry helpers.

This package provides compact constructors and symbolic representations for
finite point operations, including single abelian generators, full crystallographic
point groups, and their induced actions on polynomial bases.

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
[`SpinfulPhaseSector`][qten.pointgroups.ops.SpinfulPhaseSector] labels
double-valued abelian phases for spin-1/2 representations.
[`SymmetryDegeneracy`][qten.pointgroups.ops.SymmetryDegeneracy] is a typed copy
index used when a projected sector occurs more than once.
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
    SpinfulPhaseSector as SpinfulPhaseSector,
    SymmetryDegeneracy as SymmetryDegeneracy,
    get_direct_transform as get_direct_transform,
    joint_point_group_basis as joint_point_group_basis,
    point_group_operator_symmetrize as point_group_operator_symmetrize,
    spinful_hilbert_opr_repr as spinful_hilbert_opr_repr,
    spinful_transform_basis as spinful_transform_basis,
)
