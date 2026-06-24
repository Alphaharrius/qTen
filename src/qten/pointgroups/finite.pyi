import sympy as sy
from dataclasses import dataclass
from typing import Any, Iterable
from .basis import PointGroupBasis as PointGroupBasis
from .elements import PointGroupElement as PointGroupElement

@dataclass(frozen=True)
class FinitePointGroup:
    generators: tuple[PointGroupElement, ...]
    axes: tuple[sy.Symbol, ...]
    symbol: str | None
    irreps: dict[str, Any] | None

    @classmethod
    def from_matrices(
        cls,
        matrices: Iterable[sy.ImmutableDenseMatrix],
        axes: tuple[sy.Symbol, ...],
        *,
        symbol: str | None = None,
        irreps: dict[str, Any] | None = None,
    ) -> FinitePointGroup: ...
    def elements(self, max_order: int = 512) -> tuple[PointGroupElement, ...]: ...
    def order(self) -> int: ...
    def is_abelian(self) -> bool: ...
    def conjugacy_classes(self) -> tuple[tuple[int, ...], ...]: ...
    def trivial_projector(self, order: int) -> sy.ImmutableDenseMatrix: ...
    def irrep_projector(self, order: int, irrep: str) -> sy.ImmutableDenseMatrix: ...
    def irrep_basis(self, order: int, irrep: str) -> tuple[PointGroupBasis, ...]: ...
    def invariant_basis(self, order: int) -> tuple[PointGroupBasis, ...]: ...
