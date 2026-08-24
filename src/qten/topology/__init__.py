r"""Quantum geometry and topology of momentum-resolved band Hamiltonians.

This package computes geometric and topological properties of an isolated
occupied-band subspace carried by a rank-3
[`Tensor`][qten.linalg.tensors.Tensor] with dims
``(MomentumSpace, HilbertSpace, HilbertSpace)``. Chern and quantum-geometry
routines diagonalize independently at every supplied momentum. The
two- and three-dimensional \(\mathbb{Z}_2\) routine Fourier-interpolates the
mesh so TRIM and Wilson strings can be sampled off the original grid.

Core API
--------
- [`quantum_geometric_tensor`][qten.topology.quantum_geometric_tensor]
  Gauge-invariant quantum geometric tensor obtained from finite differences
  of the occupied projector.
- [`fubini_study_metric`][qten.topology.fubini_study_metric]
  Symmetric metric given by the real part of the quantum geometric tensor.
- [`berry_curvature`][qten.topology.berry_curvature]
  Local Berry curvature given by its imaginary antisymmetric part.
- [`chern_number`][qten.topology.chern_number]
  First Chern number computed either with discrete FHS link variables or by
  integrating the finite-difference Berry curvature.
- [`z2_indices`][qten.topology.z2_indices]
  Two- and three-dimensional \(\mathbb{Z}_2\) indices from Fu--Kane inversion
  parities or hybrid-Wannier Wilson loops.

Submodules
----------
- [`qten.topology.chern`][qten.topology.chern]
  Quantum geometric tensor, Fubini--Study metric, Berry curvature, and Chern
  number.
- [`qten.topology.z2`][qten.topology.z2]
  Two- and three-dimensional \(\mathbb{Z}_2\) invariants.
"""

from .chern import (
    FHSResult as FHSResult,
    QGTResult as QGTResult,
    berry_curvature as berry_curvature,
    chern_number as chern_number,
    fubini_study_metric as fubini_study_metric,
    quantum_geometric_tensor as quantum_geometric_tensor,
)
from .z2 import (
    Z2CombinedResult as Z2CombinedResult,
    Z2ParityResult as Z2ParityResult,
    Z2ParityTrimDiagnostics as Z2ParityTrimDiagnostics,
    Z2WilsonPlaneResult as Z2WilsonPlaneResult,
    Z2WilsonResult as Z2WilsonResult,
    z2_indices as z2_indices,
)

from . import chern as chern
from . import z2 as z2

__all__ = [
    "FHSResult",
    "QGTResult",
    "Z2CombinedResult",
    "Z2ParityResult",
    "Z2ParityTrimDiagnostics",
    "Z2WilsonPlaneResult",
    "Z2WilsonResult",
    "berry_curvature",
    "chern",
    "chern_number",
    "fubini_study_metric",
    "quantum_geometric_tensor",
    "z2",
    "z2_indices",
]
