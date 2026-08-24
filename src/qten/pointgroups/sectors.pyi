from dataclasses import dataclass
import sympy as sy

@dataclass(frozen=True)
class FiniteIrrepSector:
    group: str
    irrep: str
    dim: int

@dataclass(frozen=True)
class SpinorIrrepSector:
    group: str
    irrep: str
    dim: int
    source: str = "qten-su2-principal-v1"

@dataclass(frozen=True)
class SpinfulPhaseSector:
    phase: sy.Expr
    spatial_order: int

@dataclass(frozen=True)
class JointSpinfulPhaseSector:
    phases: tuple[sy.Expr, ...]
    spatial_orders: tuple[int, ...]

@dataclass(frozen=True)
class SymmetryDegeneracy:
    index: int
