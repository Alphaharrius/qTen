"""
Symbolic matrix validator factories.

This module provides reusable validators for dataclasses that store SymPy
matrices. Each public function returns a validator compatible with
[`need_validation()`][qten.validations.need_validation]. The returned callable
looks up a named attribute on the validated instance and raises when the stored
matrix does not satisfy the requested symbolic constraint.
"""

from typing import Any, Callable

import sympy as sy


def check_invertibility(attr_name: str) -> Callable[[Any], None]:
    """
    Build a validator that enforces matrix invertibility on an instance attribute.

    The returned callable looks up ``attr_name`` on the provided instance and
    verifies that the value is a `sympy.ImmutableDenseMatrix` with a non-zero
    determinant.

    Returned validator raises
    -------------------------
    TypeError
        If the named attribute is not an immutable dense SymPy matrix.
    ValueError
        If the named matrix is singular.

    Parameters
    ----------
    attr_name : str
        Name of the instance attribute expected to contain the symbolic matrix.

    Returns
    -------
    Callable[[Any], None]
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
    """
    Build a validator that enforces a positive determinant on a matrix attribute.

    The returned callable retrieves ``attr_name`` from the provided instance and
    verifies that the value is a `sympy.ImmutableDenseMatrix` whose determinant
    is strictly positive.

    This validator is useful when the matrix represents an orientation-preserving
    linear transformation.

    Returned validator raises
    -------------------------
    TypeError
        If the named attribute is not an immutable dense SymPy matrix.
    ValueError
        If the named matrix has a non-positive determinant.

    Parameters
    ----------
    attr_name : str
        Name of the instance attribute expected to contain the symbolic matrix.

    Returns
    -------
    Callable[[Any], None]
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
    """
    Build a validator that enforces numerical matrix entries on an attribute.

    The returned callable looks up ``attr_name`` on the provided instance and
    verifies that the value is a `sympy.ImmutableDenseMatrix` whose entries all
    report `is_number` as true.

    Returned validator raises
    -------------------------
    TypeError
        If the named attribute is not an immutable dense SymPy matrix.
    ValueError
        If any matrix entry is symbolic or otherwise non-numeric.

    Parameters
    ----------
    attr_name : str
        Name of the instance attribute expected to contain the symbolic matrix.

    Returns
    -------
    Callable[[Any], None]
        A validator callable suitable for use in attribute validation hooks.
    """

    def validator(instance: Any) -> None:
        matrix = getattr(instance, attr_name)
        if not isinstance(matrix, sy.ImmutableDenseMatrix):
            raise TypeError(f"{attr_name} must be an ImmutableDenseMatrix")
        if any(not entry.is_number for entry in matrix):
            raise ValueError(f"{attr_name} must contain only numerical entries")

    return validator
