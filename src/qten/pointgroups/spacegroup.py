r"""Seitz factor groups and little groups of affine operations.

A [`SpaceGroup`][qten.pointgroups.spacegroup.SpaceGroup] is a finite list of
[`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr] representatives
\(\{R\mid\tau\}\) for the factor group \(G/T\). Lattice translations are not
stored; they appear as wrap phases in \(D(g,k)\).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence, cast

import sympy as sy

from .elements import PointGroupOpr
from .finite import _matrix_key
from ..geometries import AffineSpace, Momentum, Offset


def _same_momentum(left: Momentum, right: Momentum) -> bool:
    if left.space != right.space:
        return False
    diff = sy.Matrix(left.fractional().rep - right.fractional().rep)
    return all(sy.simplify(entry) == 0 for entry in diff)


def _seitz_key(opr: PointGroupOpr) -> tuple[tuple[float, ...], tuple[object, ...]]:
    r"""Identify \(\{R\mid\tau\}\) modulo primitive-cell translations."""
    tau = opr.offset.fractional()
    tau_key = []
    for i in range(tau.dim):
        value = sy.nsimplify(sy.simplify(tau.rep[i, 0]))
        tau_key.append(0 if value == 0 else value)
    return _matrix_key(opr.g.irrep), tuple(tau_key)


def _with_fractional_translation(opr: PointGroupOpr) -> PointGroupOpr:
    return opr.with_translation(opr.offset.fractional())


def _reduce_seitz(opr: PointGroupOpr, modulo: AffineSpace | None) -> PointGroupOpr:
    r"""Return \(\{R\mid\tau\}\) with \(\tau\) reduced by `modulo`, if given.

    The default (`modulo is None`) keeps the intra-cell fractional part on
    `opr.offset.space`. That space is the computational cell: a supercell
    Offset treats a primitive translation as a nonzero \(\tau\). Pass the
    primitive lattice (or any `AffineSpace` with that basis) so products that
    differ by a primitive vector stay in \(G/T\).
    """
    if modulo is None:
        return _with_fractional_translation(opr)
    home = opr.offset.space
    tau = opr.offset.rebase(modulo).fractional()
    if tau.space != home:
        tau = tau.rebase(home)
    return opr.with_translation(tau)


def _identity_seitz(opr: PointGroupOpr) -> PointGroupOpr:
    ident = opr.g.inv() @ opr.g
    zero = Offset(
        sy.ImmutableDenseMatrix([0] * opr.offset.dim),
        opr.offset.space,
    )
    return PointGroupOpr._from_parts(ident, zero)


@dataclass(frozen=True)
class SpaceGroup:
    r"""Finite Seitz factor group: one \(\{R\mid\tau\}\) per point operation.

    Elements are stored with intra-cell translations. Two representatives that
    differ by a primitive lattice vector are the same factor-group element.

    Parameters
    ----------
    elements : tuple[PointGroupOpr, ...]
        Closed list of Seitz operators, typically from
        [`from_generators`][qten.pointgroups.spacegroup.SpaceGroup.from_generators].
    """

    elements: tuple[PointGroupOpr, ...]

    def __post_init__(self) -> None:
        if not self.elements:
            raise ValueError("SpaceGroup needs at least one Seitz representative.")

    @property
    def order(self) -> int:
        """Number of factor-group representatives."""
        return len(self.elements)

    @classmethod
    def from_generators(
        cls,
        generators: Sequence[PointGroupOpr],
        *,
        max_order: int = 512,
        modulo: AffineSpace | None = None,
    ) -> "SpaceGroup":
        r"""Close Seitz generators modulo translations.

        Parameters
        ----------
        generators : Sequence[PointGroupOpr]
            Affine generators \(\{R\mid\tau\}\).
        max_order : int, default 512
            Abort if the factor group grows past this many elements.
        modulo : AffineSpace, optional
            Lattice (or affine space) whose fractional part is \(T\). Default
            is `offset.space` of each generator. On a supercell Offset that
            is \(T_{\mathrm{supercell}}\); pass the primitive lattice for
            \(G/T\).

        Returns
        -------
        SpaceGroup
            Identity first, then remaining representatives in discovery order.

        Raises
        ------
        ValueError
            If `generators` is empty or closure exceeds `max_order`.
        """
        if not generators:
            raise ValueError("SpaceGroup.from_generators needs at least one generator.")

        seen: dict[tuple[tuple[float, ...], tuple[object, ...]], PointGroupOpr] = {}
        queue: deque[PointGroupOpr] = deque()

        def add(opr: PointGroupOpr) -> None:
            reduced = _reduce_seitz(opr, modulo)
            key = _seitz_key(reduced)
            if key in seen:
                return
            if len(seen) >= max_order:
                raise ValueError(
                    "Space-group closure exceeded max_order; "
                    "translations may be wrapping on a supercell. "
                    "Pass modulo= the primitive lattice."
                )
            seen[key] = reduced
            queue.append(reduced)

        gens = tuple(generators)
        add(_identity_seitz(gens[0]))
        for gen in gens:
            add(gen)
        while queue:
            current = queue.popleft()
            for gen in gens:
                add(current @ gen)
                add(gen @ current)
        return cls(elements=tuple(seen.values()))

    def little_group(self, k: Momentum) -> tuple[PointGroupOpr, ...]:
        r"""Return \(\{g: R(g)k\equiv k\}\) in this factor group, in stored order."""
        target = k.fractional()
        members = tuple(
            element
            for element in self.elements
            if _same_momentum(cast(Momentum, element @ k), target)
        )
        if not members:
            raise ValueError(f"No little-group elements stabilize {k}.")
        return members
