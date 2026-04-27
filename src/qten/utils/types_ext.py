"""
Type-introspection helpers.

The functions in this module provide small wrappers around Python class
introspection for places where QTen needs a stable display name or the full set
of registered subclasses for dispatch and diagnostics.
"""

from abc import ABCMeta
from typing import (
    Tuple,
    Type,
    cast,
)


def subtypes(cls: Type) -> Tuple[ABCMeta, ...]:
    """
    Return all transitive subclasses of a class.

    The result includes direct and indirect subclasses reachable through
    `cls.__subclasses__()`. Ordering follows set traversal and should not be
    treated as stable.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Tuple[ABCMeta, ...]
        A tuple containing all direct and indirect subclasses of `cls`.

    Examples
    --------
    ```python
    subtypes(BaseClass)
    ```
    """
    out = set()
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        if sub not in out:
            out.add(sub)
            stack.extend(sub.__subclasses__())
    return cast(Tuple[ABCMeta, ...], tuple(out))


def full_typename(cls: Type) -> str:
    """
    Get the full module and class name of a type.

    Parameters
    ----------
    cls : Type
        The class to get the name of.

    Returns
    -------
    str
        The full name of the class, including its module.

    Examples
    --------
    ```python
    full_typename(Device)
    ```
    """
    return f"{cls.__module__}.{cls.__qualname__}"
