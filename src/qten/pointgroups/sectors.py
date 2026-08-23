"""
Labels for projected point-group symmetry sectors.

These tags mark Hilbert-space columns after
[`point_group_column_symmetrize`][qten.pointgroups.ops.point_group_column_symmetrize]
and the joint helpers. They are not group objects and do not live in
[`ops`][qten.pointgroups.ops].
"""

from dataclasses import dataclass

import sympy as sy


@dataclass(frozen=True)
class FiniteIrrepSector:
    """Label for a finite-group non-abelian symmetry sector."""

    group: str
    irrep: str
    dim: int


@dataclass(frozen=True)
class SpinorIrrepSector:
    """Label for a projective spinor irrep sector of a finite point group."""

    group: str
    irrep: str
    dim: int
    source: str = "qten-su2-principal-v1"


@dataclass(frozen=True)
class SpinfulPhaseSector:
    r"""Label for an abelian spinorial sector.

    The phase \(\zeta\) satisfies \(\zeta^{2n}=1\), where \(n\) is the
    spatial order of the generator. The extra factor of two is the safe
    period of \(u(g)\), since \(u(2\pi)=-I\).
    """

    phase: sy.Expr
    spatial_order: int


@dataclass(frozen=True)
class JointSpinfulPhaseSector:
    r"""Label for simultaneous spinorial phases of a commuting family.

    Each \(\zeta_i\) satisfies \(\zeta_i^{2n_i}=1\) for the corresponding
    spatial order \(n_i\).
    """

    phases: tuple[sy.Expr, ...]
    spatial_orders: tuple[int, ...]


@dataclass(frozen=True)
class SymmetryDegeneracy:
    """Typed copy index for repeated symmetry-sector labels."""

    index: int
