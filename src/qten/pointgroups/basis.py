"""
Point-group polynomial basis labels.

[`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] stores a homogeneous
polynomial together with the representation metadata that identifies its
symmetry sector. Abelian eigen-sectors usually carry a phase eigenvalue, while
finite-group sectors carry a character-table irrep label.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sy

from ..abstracts import Operable
from ..geometries.spatials import Spatial


@dataclass(frozen=True)
class PointGroupBasis(Spatial):
    """
    Polynomial basis label belonging to a point-group representation sector.

    [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] pairs an exact
    homogeneous polynomial `expr` with its coefficient vector `rep` in a fixed
    Euclidean monomial basis. Sector metadata distinguishes abelian phase labels
    from finite-group irrep labels.

    Attributes
    ----------
    expr : sy.Expr
        Exact polynomial expression reconstructed from `rep`.
    axes : tuple[sy.Symbol, ...]
        Ordered coordinate symbols used by the polynomial.
    order : int
        Homogeneous degree of the polynomial.
    rep : sy.ImmutableDenseMatrix
        Coefficient vector in the Euclidean monomial basis of degree `order`,
        normalized so the first nonzero coefficient has unit magnitude.
    group : str
        Group tag used in string labels. Defaults to `"generic"` for abelian
        eigen-bases and to a Hermann-Mauguin symbol for finite-group sectors.
    irrep : sy.Expr | str | tuple[sy.Expr, ...]
        Sector label. Abelian eigen-bases store a phase eigenvalue; finite-group
        sectors store a character-table irrep name such as `"A1"` or `"E"`.
    irrep_dim : int
        Dimension of the irrep sector. Equals `1` for abelian phase labels.
    copy_index : int
        Copy index when the same irrep appears with multiplicity.
    component_index : int
        Component index within an irrep multiplet.

    Notes
    -----
    String rendering collapses to the bare polynomial when `group == "generic"`
    and the sector is the trivial abelian phase `1`. Otherwise the label is
    formatted as `group:irrep:expr`.
    """

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
        """
        Build a normalized basis label from a Euclidean coefficient vector.

        Parameters
        ----------
        rep : sy.ImmutableDenseMatrix
            Coefficient vector in the Euclidean monomial basis.
        euclidean_basis : sy.ImmutableDenseMatrix
            Row matrix of monomials matching `rep`.
        axes : tuple[sy.Symbol, ...]
            Ordered coordinate symbols.
        order : int
            Homogeneous polynomial degree.
        group : str, default `"generic"`
            Group tag stored on the returned label.
        irrep : sy.Expr | str | tuple[sy.Expr, ...], optional
            Sector label stored on the returned basis.
        irrep_dim : int, default 1
            Dimension of the irrep sector.
        copy_index : int, default 0
            Copy index for repeated irreps.
        component_index : int, default 0
            Component index within an irrep multiplet.

        Returns
        -------
        PointGroupBasis
            Basis label whose `rep` is magnitude-normalized by the first nonzero
            coefficient while preserving overall sign.
        """

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
