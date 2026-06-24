"""
Point-group and abelian symmetry helpers.

This package provides compact constructors and symbolic representations for
finite abelian point operations, especially Cartesian rotations, mirrors, and
their induced actions on polynomial bases.

Core exports
------------
[`pointgroup`][qten.pointgroups] parses a compact query string into a symmetry
object. [`PointGroupElement`][qten.pointgroups.elements.PointGroupElement] represents a
linear abelian symmetry acting on coordinate functions.
[`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr] is the affine extension of
an abelian group with translation. [`PointGroupBasis`][qten.pointgroups.elements.PointGroupBasis]
is the eigen-basis function object produced from symmetry representations.

Joint-basis helper
------------------
[`joint_point_group_basis`][qten.pointgroups.ops.joint_point_group_basis] constructs a
common eigen-basis for compatible commuting operators.
"""

from ._pointgroups import pointgroup as pointgroup

from .elements import (
    PointGroupElement as PointGroupElement,
    PointGroupOpr as PointGroupOpr,
)
from .basis import PointGroupBasis as PointGroupBasis
from .finite import FinitePointGroup as FinitePointGroup
from .ops import (
    get_direct_transform as get_direct_transform,
    joint_point_group_basis as joint_point_group_basis,
)
