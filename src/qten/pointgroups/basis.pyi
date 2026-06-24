import sympy as sy
from dataclasses import dataclass
from ..geometries.spatials import Spatial as Spatial


@dataclass(frozen=True)
class PointGroupBasis(Spatial):
    expr: sy.Expr
    axes: tuple[sy.Symbol, ...]
    order: int
    rep: sy.ImmutableDenseMatrix
    group: str
    irrep: sy.Expr | str | tuple[sy.Expr, ...]
    irrep_dim: int
    copy_index: int
    component_index: int

    @classmethod
    def from_rep(
        cls,
        rep: sy.ImmutableDenseMatrix,
        euclidean_basis: sy.ImmutableDenseMatrix,
        axes: tuple[sy.Symbol, ...],
        order: int,
        *,
        group: str = "generic",
        irrep: sy.Expr | str | tuple[sy.Expr, ...] = sy.Integer(1),
        irrep_dim: int = 1,
        copy_index: int = 0,
        component_index: int = 0,
    ) -> PointGroupBasis: ...

    @property
    def dim(self) -> int: ...

    def __lt__(self, other: PointGroupBasis) -> bool: ...
    def __gt__(self, other: PointGroupBasis) -> bool: ...
