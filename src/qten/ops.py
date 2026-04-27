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

Symbolic operations
-------------------
- [`translate_opr()`][qten.symbolics.ops.translate_opr]
- [`rebase_opr()`][qten.symbolics.ops.rebase_opr]
- [`fractional_opr()`][qten.symbolics.ops.fractional_opr]
- [`region_hilbert()`][qten.symbolics.ops.region_hilbert]
- [`hilbert_opr_repr()`][qten.symbolics.ops.hilbert_opr_repr]
- [`interpolate_reciprocal_path()`][qten.symbolics.ops.interpolate_reciprocal_path]

Point-group operations
----------------------
- [`joint_abelian_basis()`][qten.pointgroups.ops.joint_abelian_basis]
- [`abelian_column_symmetrize()`][qten.pointgroups.ops.abelian_column_symmetrize]
- [`joint_abelian_column_symmetrize()`][qten.pointgroups.ops.joint_abelian_column_symmetrize]

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
    interpolate_reciprocal_path as interpolate_reciprocal_path,
    rebase_opr as rebase_opr,
    translate_opr as translate_opr,
)
from .bands import interpolate_path as interpolate_path
from .pointgroups.ops import (
    abelian_column_symmetrize as abelian_column_symmetrize,
    joint_abelian_basis as joint_abelian_basis,
    joint_abelian_column_symmetrize as joint_abelian_column_symmetrize,
)
