from typing import Any, Callable

import sympy as sy


def check_invertibility(attr_name: str) -> Callable[[Any], None]:
    """Build a validator that enforces matrix invertibility on an instance attribute.

    The returned callable looks up ``attr_name`` on the provided instance and
    verifies that the value is a ``sympy.ImmutableDenseMatrix`` with a non-zero
    determinant. A ``TypeError`` is raised when the attribute is not stored as
    an immutable dense SymPy matrix, and a ``ValueError`` is raised when the
    matrix is singular.

    Parameters
    ----------
    `attr_name`:
        Name of the instance attribute expected to contain the symbolic matrix.

    Returns
    -------
    `Callable[[Any], None]`
        A validator callable suitable for use in attribute validation hooks.
    """

    def validator(instance: Any) -> None:
        matrix = getattr(instance, attr_name)
        if not isinstance(matrix, sy.ImmutableDenseMatrix):
            raise TypeError(f"{attr_name} must be an ImmutableDenseMatrix")
        if matrix.det() == 0:
            raise ValueError(f"{attr_name} must have non-zero determinant")

    return validator


def check_proper_transformation(attr_name: str) -> Callable[[Any], None]:
    """Build a validator that enforces a positive determinant on a matrix attribute.

    The returned callable retrieves ``attr_name`` from the provided instance and
    verifies that the value is a ``sympy.ImmutableDenseMatrix`` whose determinant
    is strictly positive. A ``TypeError`` is raised if the attribute is not stored
    as an immutable dense SymPy matrix, and a ``ValueError`` is raised if the
    determinant is non-positive.

    This validator is useful when the matrix represents an orientation-preserving
    linear transformation.

    Parameters
    ----------
    `attr_name`:
        Name of the instance attribute expected to contain the symbolic matrix.

    Returns
    -------
    `Callable[[Any], None]`
        A validator callable suitable for use in attribute validation hooks.
    """

    def validator(instance: Any) -> None:
        matrix = getattr(instance, attr_name)
        if not isinstance(matrix, sy.ImmutableDenseMatrix):
            raise TypeError(f"{attr_name} must be an ImmutableDenseMatrix")
        if matrix.det() <= 0:
            raise ValueError(f"{attr_name} must have positive determinant")

    return validator


def check_numerical(attr_name: str) -> Callable[[Any], None]:
    """Build a validator that enforces numerical matrix entries on an attribute.

    The returned callable looks up ``attr_name`` on the provided instance and
    verifies that the value is a ``sympy.ImmutableDenseMatrix`` whose entries
    all report ``is_number`` as true. A ``TypeError`` is raised when the
    attribute is not an immutable dense SymPy matrix, and a ``ValueError`` is
    raised when any element is symbolic or otherwise non-numeric.

    Parameters
    ----------
    `attr_name`:
        Name of the instance attribute expected to contain the symbolic matrix.

    Returns
    -------
    `Callable[[Any], None]`
        A validator callable suitable for use in attribute validation hooks.
    """

    def validator(instance: Any) -> None:
        matrix = getattr(instance, attr_name)
        if not isinstance(matrix, sy.ImmutableDenseMatrix):
            raise TypeError(f"{attr_name} must be an ImmutableDenseMatrix")
        if any(not entry.is_number for entry in matrix):
            raise ValueError(f"{attr_name} must contain only numerical entries")

    return validator
