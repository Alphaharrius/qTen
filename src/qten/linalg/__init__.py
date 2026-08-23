"""
Linear-algebra routines built on top of QTen tensors.

This package contains decomposition algorithms and tensor-aware numerical
helpers that operate on [`Tensor`][qten.linalg.tensors.Tensor] objects while
preserving symbolic dimension metadata.

Decompositions
--------------
- [`eig`][qten.linalg.decompose.eig]
  General eigendecomposition for square tensor-valued matrices.
- [`eigh`][qten.linalg.decompose.eigh]
  Hermitian eigendecomposition.
- [`eigvals`][qten.linalg.decompose.eigvals]
  General eigenvalues only.
- [`eigvalsh`][qten.linalg.decompose.eigvalsh]
  Hermitian eigenvalues only.
- [`qr`][qten.linalg.decompose.qr]
  QR factorization.
- [`lowdin_orthonormalize`][qten.linalg.orthogonalize.lowdin_orthonormalize]
  Symmetric orthonormalization of matrix columns.
- [`svd`][qten.linalg.decompose.svd]
  Singular value decomposition.

Convenience re-exports
----------------------
- [`einsum`][qten.linalg.tensors.einsum]
  Tensor contraction helper re-exported from [`qten.linalg.tensors`][qten.linalg.tensors].
- [`norm`][qten.linalg.tensors.norm]
  Tensor norm helper re-exported from [`qten.linalg.tensors`][qten.linalg.tensors].

Related modules
---------------
- [`qten.linalg.tensors`][qten.linalg.tensors]
- [`qten.linalg.decompose`][qten.linalg.decompose]
"""

from .decompose import (
    eig as eig,
    eigh as eigh,
    eigvalsh as eigvalsh,
    eigvals as eigvals,
    qr as qr,
    svd as svd,
)
from .orthogonalize import lowdin_orthonormalize as lowdin_orthonormalize
from .tensors import einsum as einsum, norm as norm
