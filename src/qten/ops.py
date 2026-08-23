"""
Convenience namespace for QTen operation helpers.

This module re-exports commonly used functional operations from geometry,
symbolic, band-structure, and point-group modules. It is intended as a compact
import surface for workflows that combine several parts of QTen without needing
to remember each helper's implementation module.

Geometry operations
-------------------
- [`nearest_sites()`][qten.geometries.ops.nearest_sites]
- [`region_tile()`][qten.geometries.ops.region_tile]
- [`region_centering()`][qten.geometries.ops.region_centering]
- [`center_of_region()`][qten.geometries.ops.center_of_region]
- [`interstitial_centers()`][qten.geometries.ops.interstitial_centers]
- [`get_strip_region_2d()`][qten.geometries.ops.get_strip_region_2d]

Fourier and band operations
---------------------------
- [`fourier_kernel()`][qten.geometries.fourier.fourier_kernel]
- [`fourier_transform()`][qten.geometries.fourier.fourier_transform]
- [`region_restrict()`][qten.geometries.fourier.region_restrict]
- [`interpolate_path()`][qten.bands.interpolate_path]
- [`svd_projection()`][qten.bands.svd_projection]

Symbolic operations
-------------------
- [`translate_opr()`][qten.symbolics.ops.translate_opr]
- [`rebase_opr()`][qten.symbolics.ops.rebase_opr]
- [`fractional_opr()`][qten.symbolics.ops.fractional_opr]
- [`region_hilbert()`][qten.symbolics.ops.region_hilbert]
- [`hilbert_opr_repr()`][qten.symbolics.ops.hilbert_opr_repr]
  One-to-one symbolic representation; it cannot express SU(2) spin mixing.

Point-group operations
----------------------
- [`hilbert_repr()`][qten.pointgroups.ops.hilbert_repr]
  Hilbert-space `D(g)`, including the SU(2) factor on a spinful space.
- [`joint_point_group_basis()`][qten.pointgroups.ops.joint_point_group_basis]
- [`point_group_column_symmetrize()`][qten.pointgroups.ops.point_group_column_symmetrize]
- [`point_group_operator_symmetrize()`][qten.pointgroups.ops.point_group_operator_symmetrize]
- [`joint_point_group_column_symmetrize()`][qten.pointgroups.ops.joint_point_group_column_symmetrize]
- [`get_direct_transform()`][qten.pointgroups.ops.get_direct_transform]

Notes
-----
The full behavioral contracts are documented on the original functions linked
above. This module does not wrap or alter the imported functions.
"""

from .geometries.fourier import (
    fourier_kernel as fourier_kernel,
    fourier_transform as fourier_transform,
    region_restrict as region_restrict,
)
from .geometries import (
    center_of_region as center_of_region,
    get_strip_region_2d as get_strip_region_2d,
    interstitial_centers as interstitial_centers,
    nearest_sites as nearest_sites,
    region_centering as region_centering,
    region_tile as region_tile,
)
from .symbolics import (
    fractional_opr as fractional_opr,
    region_hilbert as region_hilbert,
    hilbert_opr_repr as hilbert_opr_repr,
    rebase_opr as rebase_opr,
    translate_opr as translate_opr,
)
from .bands import (
    interpolate_path as interpolate_path,
    svd_projection as svd_projection,
)
from .pointgroups.ops import (
    get_direct_transform as get_direct_transform,
    hilbert_repr as hilbert_repr,
    joint_point_group_basis as joint_point_group_basis,
    joint_point_group_column_symmetrize as joint_point_group_column_symmetrize,
    point_group_column_symmetrize as point_group_column_symmetrize,
    point_group_operator_symmetrize as point_group_operator_symmetrize,
)
