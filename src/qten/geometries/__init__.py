"""
Geometry and spatial-structure primitives used throughout QTen.

This package collects the concrete coordinate-space objects needed to describe
real-space and reciprocal-space structure, boundary identification, basis
changes, and region construction.

Core spaces and coordinates
---------------------------
- [`AffineSpace`][qten.geometries.spatials.AffineSpace]
  Generic affine coordinate space.
- [`Lattice`][qten.geometries.spatials.Lattice]
  Real-space lattice with basis, boundaries, and optional unit cell.
- [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]
  Reciprocal-space dual lattice.
- [`Offset`][qten.geometries.spatials.Offset]
  Real-space coordinate/irrep attached to an affine space.
- [`Momentum`][qten.geometries.spatials.Momentum]
  Reciprocal-space momentum coordinate.

Boundary handling
-----------------
- [`BoundaryCondition`][qten.geometries.boundary.BoundaryCondition]
  Abstract boundary interface.
- [`PeriodicBoundary`][qten.geometries.boundary.PeriodicBoundary]
  Smith-normal-form based periodic wrapping and minimum-image distances.

Basis transforms
----------------
- [`BasisTransform`][qten.geometries.basis_transform.BasisTransform]
  Forward-view basis-change operator.
- [`InverseBasisTransform`][qten.geometries.basis_transform.InverseBasisTransform]
  Inverse-view companion transform.

Region and neighborhood helpers
-------------------------------
- [`center_of_region`][qten.geometries.ops.center_of_region]
- [`get_strip_region_2d`][qten.geometries.ops.get_strip_region_2d]
- [`interstitial_centers`][qten.geometries.ops.interstitial_centers]
- [`nearest_sites`][qten.geometries.ops.nearest_sites]
- [`region_centering`][qten.geometries.ops.region_centering]
- [`region_tile`][qten.geometries.ops.region_tile]

Related modules
---------------
Additional geometry-related APIs are available in:
- [`qten.geometries.fourier`][qten.geometries.fourier]
- [`qten.geometries.boundary`][qten.geometries.boundary]
- [`qten.geometries.basis_transform`][qten.geometries.basis_transform]
"""

from .spatials import (
    AffineSpace as AffineSpace,
    Lattice as Lattice,
    ReciprocalLattice as ReciprocalLattice,
    Offset as Offset,
    Momentum as Momentum,
)

from .boundary import (
    BoundaryCondition as BoundaryCondition,
    PeriodicBoundary as PeriodicBoundary,
)

from .basis_transform import BasisTransform as BasisTransform
from .basis_transform import InverseBasisTransform as InverseBasisTransform
from .ops import (
    center_of_region as center_of_region,
    get_strip_region_2d as get_strip_region_2d,
    interstitial_centers as interstitial_centers,
    nearest_sites as nearest_sites,
    region_centering as region_centering,
    region_tile as region_tile,
)
