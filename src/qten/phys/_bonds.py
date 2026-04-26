"""
Symbolic bond terms for physics observables.

A bond is the smallest operator-building block used by
[`FFObservable`][qten.phys.FFObservable]: a scalar coefficient attached to an
ordered pair of basis states. Keeping the coefficient separate from the
endpoint states lets symbolic manipulations preserve the distinction between
"which transition" and "with what weight" until tensor construction.

In the current free-fermionic convention, each bond contributes one directed
matrix element from a source [`U1Basis`][qten.symbolics.hilbert_space.U1Basis]
state to a destination state. Hermitian completion is handled by the observable
that consumes the bond, not by the bond object itself.
"""

from dataclasses import dataclass
from typing import Tuple

from ..symbolics import Multiple
from ..symbolics import U1Basis


@dataclass(frozen=True)
class Bond(Multiple[Tuple[U1Basis, U1Basis]]):
    """
    Weighted directed transition between two [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] states.

    `Bond` specializes [`Multiple`][qten.symbolics.base.Multiple] for the common physics case where the base
    object is an ordered pair `(src, dst)` of basis states. The scalar
    coefficient is kept separate from the endpoint states so symbolic
    manipulations can preserve the structural meaning of the bond until a later
    tensor-construction step.

    In an [`FFObservable`][qten.phys.FFObservable], the ordered pair is
    interpreted as a matrix element from `src` to `dst`. The observable is
    responsible for adding the Hermitian conjugate contribution when converting
    the collected bonds to a tensor.

    Attributes
    ----------
    coef : sy.Number
        Symbolic coefficient multiplying the bond contribution.
    base : Tuple[U1Basis, U1Basis]
        Ordered pair `(src, dst)` describing the directed basis-state
        transition represented by the bond.

    See Also
    --------
    qten.phys.FFObservable
        Observable builder that consumes `Bond` instances.
    """
