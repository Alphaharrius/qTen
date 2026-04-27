"""Lightweight symbolic container types used across [`qten`][qten].

Globals
-------
BaseType : TypeVar
    Type variable representing the symbolic object wrapped by
    [`Multiple`][qten.symbolics.Multiple]. The parameter keeps the container
    generic so the same class can be used for SymPy expressions,
    project-specific symbolic types, or other scalar-like objects without
    runtime conversion overhead.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

import sympy as sy

BaseType = TypeVar("BaseType")


@dataclass(frozen=True)
class Multiple(Generic[BaseType]):
    """
    Represent a scalar coefficient multiplied by an arbitrary base object.

    The class is intentionally minimal: it stores the coefficient and base
    separately so higher-level symbolic routines can postpone expansion or
    simplification until needed. This avoids unnecessary SymPy work in hot
    paths where the structured form is more efficient than immediately building
    a combined symbolic expression.

    Attributes
    ----------
    coef : sy.Number
        Numeric SymPy coefficient applied to `base`.
    base : BaseType
        The symbolic or scalar object being multiplied by `coef`.
    """

    coef: sy.Number
    base: BaseType
