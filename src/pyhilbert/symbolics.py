"""Lightweight symbolic container types used across :mod:`pyhilbert`.

Globals
-------
BaseType:
    Type variable representing the symbolic object wrapped by
    :class:`Multiple`. The parameter keeps the container generic so the same
    class can be used for SymPy expressions, project-specific symbolic types,
    or other scalar-like objects without runtime conversion overhead.
"""

from dataclasses import dataclass
from typing import TypeVar, Generic

import sympy as sy

BaseType = TypeVar("BaseType")


@dataclass
class Multiple(Generic[BaseType]):
    """Represent a scalar coefficient multiplied by an arbitrary base object.

    The class is intentionally minimal: it stores the coefficient and base
    separately so higher-level symbolic routines can postpone expansion or
    simplification until needed. This avoids unnecessary SymPy work in hot
    paths where the structured form is more efficient than immediately building
    a combined symbolic expression.

    Attributes
    ----------
    coef:
        Numeric SymPy coefficient applied to ``base``.
    base:
        The symbolic or scalar object being multiplied by ``coef``.
    """

    coef: sy.Number
    base: BaseType
