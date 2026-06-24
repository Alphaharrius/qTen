"""Point-group polynomial basis labels."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sy

from ..abstracts import Operable
from ..geometries.spatials import Spatial


@dataclass(frozen=True)
class PointGroupBasis(Spatial):
    """Polynomial basis label belonging to a point-group representation sector."""

    expr: sy.Expr
    axes: tuple[sy.Symbol, ...]
    order: int
    rep: sy.ImmutableDenseMatrix
    group: str = "generic"
    irrep: sy.Expr | str | tuple[sy.Expr, ...] = sy.Integer(1)
    irrep_dim: int = 1
    copy_index: int = 0
    component_index: int = 0

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
    ) -> "PointGroupBasis":
        principle_term = next(x for x in rep if x != 0)
        normalized = sy.ImmutableDenseMatrix(sy.simplify(rep / sy.Abs(principle_term)))
        expr = sy.simplify(normalized.dot(euclidean_basis))
        return cls(
            expr=expr,
            axes=axes,
            order=order,
            rep=normalized,
            group=group,
            irrep=irrep,
            irrep_dim=irrep_dim,
            copy_index=copy_index,
            component_index=component_index,
        )

    @property
    def dim(self) -> int:
        """Number of coordinate axes for this basis label."""

        return len(self.axes)

    def __str__(self) -> str:
        if sy.simplify(self.expr - 1) == 0:
            expr = "e"
        else:
            expr = str(self.expr)
        if (
            self.group == "generic"
            and isinstance(self.irrep, sy.Basic)
            and sy.simplify(self.irrep - 1) == 0
        ):
            return expr
        return f"{self.group}:{self.irrep}:{expr}"

    def __repr__(self) -> str:
        return self.__str__()


@Operable.__lt__.register
def _(a: PointGroupBasis, b: PointGroupBasis) -> bool:
    return str(a.expr) < str(b.expr)


@Operable.__gt__.register
def _(a: PointGroupBasis, b: PointGroupBasis) -> bool:
    return str(a.expr) > str(b.expr)
