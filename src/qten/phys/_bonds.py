from dataclasses import dataclass
from typing import Tuple

from ..symbolics import Multiple
from ..symbolics import U1Basis


@dataclass(frozen=True)
class Bond(Multiple[Tuple[U1Basis, U1Basis]]):
    """
    Weighted directed transition between two [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] states.

    `Bond` specializes [`Multiple`][qten.symbolics.Multiple] for the common physics case where the base
    object is an ordered pair `(src, dst)` of basis states. The scalar
    coefficient is kept separate from the endpoint states so symbolic
    manipulations can preserve the structural meaning of the bond until a later
    tensor-construction step.

    Attributes
    ----------
    coef : `sy.Number`
        Symbolic coefficient multiplying the bond contribution.
    base : `Tuple[U1Basis, U1Basis]`
        Ordered pair `(src, dst)` describing the directed basis-state
        transition represented by the bond.
    """
