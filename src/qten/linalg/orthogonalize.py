"""
Tensor-aware matrix orthogonalization routines for QTen.

This module provides orthogonalization algorithms that operate on
[`Tensor`][qten.linalg.tensors.Tensor] objects while preserving symbolic
dimension metadata.

Public orthogonalizations
-------------------------
- [`lowdin_orthonormalize`][qten.linalg.orthogonalize.lowdin_orthonormalize]
  Symmetric orthonormalization of matrix columns using the positive-definite
  inverse square root of their Gram matrix.

Conventions
-----------
All orthogonalizations act on the last two tensor dimensions as matrix axes.
The penultimate axis labels the ambient row space, the final axis labels the
columns to orthonormalize, and any leading dimensions are treated as batch
axes. The input dimensions and their ordering are preserved in the result.

The routines require every matrix in the batch to have full column rank at
the requested numerical tolerance. Consequently, the number of columns
cannot exceed the dimension of the row space. A rank-deficient input raises
an error rather than silently dropping columns or changing symbolic spaces.

Unlike the routines in [`qten.linalg.decompose`][qten.linalg.decompose], these
functions return a transformed tensor rather than decomposition factors.
"""

from .decompose import eigh
from .tensors import Tensor, einsum


def lowdin_orthonormalize(tensor: Tensor, rank_tolerance: float = 1e-10) -> Tensor:
    r"""Symmetrically orthonormalize the columns of a tensor.

    Applies the Lowdin transformation
    \(A \mapsto A(A^\dagger A)^{-1/2}\) independently to every matrix in the
    leading batch dimensions. The input dimensions are preserved.

    Parameters
    ----------
    tensor : Tensor
        Input whose last two dimensions are the row and column matrix axes.
    rank_tolerance : float, default=1e-10
        Minimum allowed eigenvalue of the column Gram matrix. A non-positive
        value or one at or below this threshold indicates linearly dependent
        columns.

    Returns
    -------
    Tensor
        A tensor with the same dimensions and orthonormal columns.

    Raises
    ------
    ValueError
        If the input has fewer than two dimensions or ``rank_tolerance`` is
        negative.
    RuntimeError
        If any matrix in the batch has linearly dependent columns at the
        requested tolerance.
    """
    if tensor.rank() < 2:
        raise ValueError(
            "Input tensor must have at least two dimensions for Lowdin "
            "orthonormalization."
        )
    if rank_tolerance < 0:
        raise ValueError("rank_tolerance must be non-negative.")

    gram = tensor.h(-2, -1) @ tensor
    eigenvalues, eigenvectors = eigh(gram)
    minimum_eigenvalue = eigenvalues.data.amin()
    if minimum_eigenvalue <= rank_tolerance:
        raise RuntimeError(
            "Lowdin orthonormalization encountered linearly dependent columns: "
            f"minimum Gram eigenvalue={minimum_eigenvalue.item():.6e}."
        )

    inverse_sqrt_eigenvalues = Tensor(
        data=eigenvalues.data.rsqrt().to(dtype=eigenvectors.data.dtype),
        dims=eigenvalues.dims,
    )
    scaled_eigenvectors = einsum(
        "...ij,...j->...ij", eigenvectors, inverse_sqrt_eigenvalues
    )
    inverse_sqrt = scaled_eigenvectors @ eigenvectors.h(-2, -1)
    return tensor @ inverse_sqrt
