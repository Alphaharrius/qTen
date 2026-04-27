"""
Tensor-aware matrix decomposition routines for QTen.

This module wraps PyTorch's dense linear-algebra decompositions so they operate
on [`Tensor`][qten.linalg.tensors.Tensor] objects while preserving symbolic
dimension metadata.

Public decompositions
---------------------
- [`eigh`][qten.linalg.decompose.eigh]
  Hermitian eigendecomposition returning [`EigH`][qten.linalg.decompose.EigH].
- [`eigvalsh`][qten.linalg.decompose.eigvalsh]
  Hermitian eigenvalues only.
- [`eig`][qten.linalg.decompose.eig]
  General eigendecomposition returning [`EigH`][qten.linalg.decompose.EigH].
- [`eigvals`][qten.linalg.decompose.eigvals]
  General eigenvalues only.
- [`qr`][qten.linalg.decompose.qr]
  QR factorization returning [`QR`][qten.linalg.decompose.QR].
- [`svd`][qten.linalg.decompose.svd]
  Singular value decomposition returning [`SVD`][qten.linalg.decompose.SVD].

Conventions
-----------
All decompositions act on the last two tensor dimensions as matrix axes and
preserve any leading dimensions as batch axes. The returned
[`Tensor`][qten.linalg.tensors.Tensor] objects replace the matrix axes with one
or more [`IndexSpace`][qten.symbolics.state_space.IndexSpace] factors
describing the decomposition bond dimensions.

For eigendecompositions, the last two dimensions must describe the same
Hilbert space up to ray ordering so the matrix is square as a symbolic
operator. For [`eigh`][qten.linalg.decompose.eigh] and
[`eigvalsh`][qten.linalg.decompose.eigvalsh], that operator is additionally
assumed to be Hermitian.
"""

from typing import NamedTuple, overload

import torch

from ..symbolics import IndexSpace, same_rays
from .tensors import Tensor


class EigH(NamedTuple):
    r"""
    Eigen-decomposition result container.

    This is the shared return type of both
    [`eigh`][qten.linalg.decompose.eigh] and
    [`eig`][qten.linalg.decompose.eig].

    Reconstruction
    --------------
    The returned tensors encode the matrix factorization on the last two axes.

    - For [`eigh`][qten.linalg.decompose.eigh], reconstruct the original
      Hermitian matrix by forming a diagonal matrix from
      `result.eigenvalues`, then evaluating the code expression
      `V @ W @ V.h(...)`, where `V = result.eigenvectors` and `W` is that
      diagonal matrix.
      In conventional notation, \(A = V \Lambda V^\dagger\).
    - For [`eig`][qten.linalg.decompose.eig], the returned tensors satisfy the
      eigenvalue equation \(A V = V\Lambda\). If the matrix is diagonalizable,
      then it can be reconstructed as \(V\Lambda V^{-1}\), where \(\Lambda\)
      is the diagonal matrix of eigenvalues. In code, this corresponds to
      products like `A @ V`, `V @ W`, and `V @ W @ V.inv()`.
      In conventional notation, \(A V = V \Lambda\) and
      \(A = V \Lambda V^{-1}\).

    Attributes
    ----------
    eigenvalues : Tensor
        Eigenvalues tensor. Its dims keep the leading batch dimensions and
        replace the matrix axes with a single
        [`IndexSpace`][qten.symbolics.state_space.IndexSpace] labeling the
        spectrum.
    eigenvectors : Tensor
        Eigenvectors tensor. Its dims keep the leading batch dimensions,
        followed by the matrix row space and the spectral
        [`IndexSpace`][qten.symbolics.state_space.IndexSpace].
    """

    eigenvalues: Tensor
    eigenvectors: Tensor


def _assert_eig_dims(tensor: Tensor) -> None:
    if tensor.rank() < 2:
        raise ValueError(
            "Input tensor must have at least two dimensions for matrix decomposition."
        )

    dim0, dim1 = tensor.dims[-2], tensor.dims[-1]
    if not same_rays(dim0, dim1):
        raise ValueError(
            "The last two dimensions of the tensor must span the same Hilbert space."
        )


def eigh(tensor: Tensor) -> EigH:
    r"""
    Perform Hermitian eigendecomposition on the last two tensor dimensions.

    This function applies [`torch.linalg.eigh`](https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html)
    to the matrix axes of a [`Tensor`][qten.linalg.tensors.Tensor]. The final
    two dimensions must span the same Hilbert space up to ray ordering so they
    can be interpreted as a Hermitian operator. Any leading dimensions are
    treated as batch dimensions and are preserved in both outputs.

    Parameters
    ----------
    tensor : Tensor
        Input tensor whose last two dimensions form Hermitian matrices.

    Returns
    -------
    EigH
        [`EigH`][qten.linalg.decompose.EigH] containing:
        - `eigenvalues`, whose dtype is the real dtype associated with the
          input and whose dims replace the matrix axes with one
          [`IndexSpace`][qten.symbolics.state_space.IndexSpace].
        - `eigenvectors`, whose dtype matches the input dtype and whose dims
          are the leading batch dims followed by `(row_dim, spectrum)`.

    Examples
    --------
    ```python
    result = eigh(tensor)
    eigenvalues = result.eigenvalues
    eigenvectors = result.eigenvectors
    ```

    Notes
    -----
    `torch.linalg.eigh` is differentiable for Hermitian inputs, but the gradients
    can be ill-defined or unstable when eigenvalues are degenerate or nearly
    degenerate. If you use this in autograd, consider stabilizing the spectrum
    (e.g., with a small perturbation) or avoiding backpropagation through
    eigenvectors when bands are expected to merge.

    The original matrix is recovered by forming a diagonal matrix from
    `eigenvalues` and evaluating \(V\Lambda V^\dagger\). In code, this is
    `eigenvectors @ W @ eigenvectors.h(-2, -1)`.
    """
    _assert_eig_dims(tensor)

    dim0 = tensor.dims[-2]
    target = tensor.align(-1, dim0)  # Align column space to match the row space
    eigenvalues, eigenvectors = torch.linalg.eigh(target.data)

    spectrum = IndexSpace.linear(eigenvalues.shape[-1])

    eigvals = Tensor(
        data=eigenvalues,
        dims=target.dims[:-2] + (spectrum,),
    )
    eigvecs = Tensor(
        data=eigenvectors,
        dims=target.dims[:-2] + (dim0, spectrum),
    )

    return EigH(eigvals, eigvecs)


def eigvalsh(tensor: Tensor) -> Tensor:
    """
    Compute Hermitian eigenvalues on the last two tensor dimensions.

    This is the eigenvalues-only companion to
    [`eigh`][qten.linalg.decompose.eigh]. The last two dimensions must span
    the same Hilbert space up to ray ordering and represent a Hermitian
    operator. Leading dimensions are treated as batch dimensions.

    Parameters
    ----------
    tensor : Tensor
        Input tensor whose last two dimensions form Hermitian matrices.

    Returns
    -------
    Tensor
        Eigenvalues as a [`Tensor`][qten.linalg.tensors.Tensor] whose dtype
        matches the real dtype associated with the input and whose dims keep
        the leading batch dimensions while replacing the matrix axes with a
        single [`IndexSpace`][qten.symbolics.state_space.IndexSpace].

    Examples
    --------
    ```python
    values = eigvalsh(tensor)
    ```
    """
    _assert_eig_dims(tensor)

    dim0 = tensor.dims[-2]
    target = tensor.align(-1, dim0)  # Align column space to match the row space
    eigenvalues = torch.linalg.eigvalsh(target.data)

    spectrum = IndexSpace.linear(eigenvalues.shape[-1])

    vals = Tensor(
        data=eigenvalues,
        dims=target.dims[:-2] + (spectrum,),
    )

    return vals


def _lexsort_eigenvalues(eigenvalues: torch.Tensor) -> torch.Tensor:
    imag_order = torch.argsort(eigenvalues.imag, dim=-1, stable=True)
    real_sorted = torch.gather(eigenvalues.real, -1, imag_order)
    real_order = torch.argsort(real_sorted, dim=-1, stable=True)
    return torch.gather(imag_order, -1, real_order)


@overload
def _sort_eigenpairs(
    eigenvalues: torch.Tensor, eigenvectors: None = None
) -> tuple[torch.Tensor, None]: ...


@overload
def _sort_eigenpairs(
    eigenvalues: torch.Tensor, eigenvectors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...


def _sort_eigenpairs(
    eigenvalues: torch.Tensor, eigenvectors: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    order = _lexsort_eigenvalues(eigenvalues)
    eigenvalues = torch.gather(eigenvalues, -1, order)
    if eigenvectors is None:
        return eigenvalues, None
    index = order.unsqueeze(-2).expand(*eigenvectors.shape[:-1], eigenvectors.shape[-1])
    eigenvectors = torch.gather(eigenvectors, -1, index)
    return eigenvalues, eigenvectors


def eig(tensor: Tensor) -> EigH:
    r"""
    Perform eigendecomposition on general square matrix axes.

    This function applies [`torch.linalg.eig`](https://pytorch.org/docs/stable/generated/torch.linalg.eig.html)
    to the final two dimensions of a
    [`Tensor`][qten.linalg.tensors.Tensor]. The last two dimensions must span
    the same Hilbert space up to ray ordering so they can be interpreted as a
    square operator. Any leading dimensions are treated as batch dimensions and
    are preserved in both outputs.

    Parameters
    ----------
    tensor : Tensor
        Input tensor whose last two dimensions form square matrices.

    Returns
    -------
    EigH
        [`EigH`][qten.linalg.decompose.EigH] containing:
        - `eigenvalues`, whose dtype is the complex dtype associated with the
          input and whose dims replace the matrix axes with one
          [`IndexSpace`][qten.symbolics.state_space.IndexSpace].
        - `eigenvectors`, whose dtype matches that complex dtype and whose dims
          are the leading batch dims followed by `(row_dim, spectrum)`.

    Examples
    --------
    ```python
    result = eig(tensor)
    values = result.eigenvalues
    vectors = result.eigenvectors
    ```

    Notes
    -----
    `torch.linalg.eig` does not guarantee any ordering of the eigenvalues. This
    function sorts eigenvalues lexicographically by `(real, imag)` and applies
    the same reordering to eigenvectors.

    The returned tensors satisfy \(A V = V\Lambda\), where \(\Lambda\) is the
    diagonal matrix of eigenvalues. If the input matrix is diagonalizable, this
    gives the reconstruction \(A = V\Lambda V^{-1}\). In code, \(V\) is
    `eigenvectors` and \(\Lambda\) is the diagonal matrix built from
    `eigenvalues`.
    """
    _assert_eig_dims(tensor)

    dim0 = tensor.dims[-2]
    target = tensor.align(-1, dim0)  # Align column space to match the row space
    eigenvalues, eigenvectors = torch.linalg.eig(target.data)
    eigenvalues, eigenvectors = _sort_eigenpairs(eigenvalues, eigenvectors)

    spectrum = IndexSpace.linear(eigenvalues.shape[-1])

    eigvals = Tensor(
        data=eigenvalues,
        dims=target.dims[:-2] + (spectrum,),
    )
    eigvecs = Tensor(
        data=eigenvectors,
        dims=target.dims[:-2] + (dim0, spectrum),
    )

    return EigH(eigvals, eigvecs)


def eigvals(tensor: Tensor) -> Tensor:
    """
    Compute eigenvalues of general square matrix axes.

    This is the eigenvalues-only companion to
    [`eig`][qten.linalg.decompose.eig]. The last two dimensions must span the
    same Hilbert space up to ray ordering and represent a square operator.
    Leading dimensions are treated as batch dimensions.

    Parameters
    ----------
    tensor : Tensor
        Input tensor whose last two dimensions form square matrices.

    Returns
    -------
    Tensor
        Eigenvalues as a [`Tensor`][qten.linalg.tensors.Tensor] whose dtype
        matches the complex dtype associated with the input and whose dims keep
        the leading batch dimensions while replacing the matrix axes with a
        single [`IndexSpace`][qten.symbolics.state_space.IndexSpace].

    Notes
    -----
    `torch.linalg.eigvals` does not guarantee any ordering of the eigenvalues.
    This function sorts eigenvalues lexicographically by `(real, imag)`.

    Examples
    --------
    ```python
    values = eigvals(tensor)
    ```
    """
    _assert_eig_dims(tensor)

    dim0 = tensor.dims[-2]
    target = tensor.align(-1, dim0)  # Align column space to match the row space
    eigenvalues = torch.linalg.eigvals(target.data)
    eigenvalues, _ = _sort_eigenpairs(eigenvalues)

    spectrum = IndexSpace.linear(eigenvalues.shape[-1])

    vals = Tensor(
        data=eigenvalues,
        dims=target.dims[:-2] + (spectrum,),
    )

    return vals


class QR(NamedTuple):
    r"""
    QR decomposition result container.

    Reconstruction
    --------------
    Reconstruct the original matrix as \(Q R\) on the last two axes. In code,
    this is `Q @ R`.

    In conventional notation, \(A = Q R\) and \(Q^\dagger Q = I\).

    Attributes
    ----------
    Q : Tensor
        Orthogonal/unitary factor with dims equal to the leading batch
        dimensions followed by `(row_dim, factor)`.
    R : Tensor
        Upper-triangular factor with dims equal to the leading batch
        dimensions followed by `(factor, col_dim)`.
    """

    Q: Tensor
    R: Tensor


def qr(tensor: Tensor) -> QR:
    r"""
    Perform reduced QR decomposition on the last two tensor dimensions.

    This function applies [`torch.linalg.qr`](https://pytorch.org/docs/stable/generated/torch.linalg.qr.html)
    with `mode="reduced"` to the matrix axes of the input tensor. The last two
    dimensions may be rectangular. Any leading dimensions are treated as batch
    dimensions and are preserved in both outputs.

    Parameters
    ----------
    tensor : Tensor
        Input tensor whose last two dimensions form matrices.

    Returns
    -------
    QR
        [`QR`][qten.linalg.decompose.QR] containing:
        - `Q`, a [`Tensor`][qten.linalg.tensors.Tensor] with orthonormal
          columns and dims `(..., row_dim, factor)`.
        - `R`, an upper-triangular
          [`Tensor`][qten.linalg.tensors.Tensor] with dims
          `(..., factor, col_dim)`.

    Examples
    --------
    ```python
    result = qr(tensor)
    q = result.Q
    r = result.R
    ```

    Notes
    -----
    The shared `factor` axis is represented by an
    [`IndexSpace`][qten.symbolics.state_space.IndexSpace] whose size equals the
    reduced QR bond dimension. The original matrix is recovered as \(Q R\), via
    `Q @ R` in code.
    """
    if tensor.rank() < 2:
        raise ValueError(
            "Input tensor must have at least two dimensions for matrix decomposition."
        )

    row_dim = tensor.dims[-2]
    col_dim = tensor.dims[-1]

    q_data, r_data = torch.linalg.qr(tensor.data, mode="reduced")
    spectral_dim = IndexSpace.linear(q_data.shape[-1])

    q = Tensor(
        data=q_data,
        dims=tensor.dims[:-2] + (row_dim, spectral_dim),
    )
    r = Tensor(
        data=r_data,
        dims=tensor.dims[:-2] + (spectral_dim, col_dim),
    )

    return QR(q, r)


class SVD(NamedTuple):
    r"""
    Singular-value decomposition result container.

    Reconstruction
    --------------
    Reconstruct the original matrix as \(U\Sigma V^\dagger\). In code, this is
    `U @ Sigma @ Vh`, where `Sigma` is either `result.S` itself when
    `values_as_matrix=True`, or the diagonal matrix formed from the singular
    values when `result.S` is returned as a vector.

    In conventional notation, \(A = U \Sigma V^\dagger\).

    Attributes
    ----------
    U : Tensor
        Left singular vectors with dims determined by `full_matrices`.
    S : Tensor
        Singular values, either as a vector or diagonal matrix depending on
        `values_as_matrix`.
    Vh : Tensor
        Right singular vectors in conjugate-transposed form.
    """

    U: Tensor
    S: Tensor
    Vh: Tensor


def svd(
    tensor: Tensor,
    values_as_matrix: bool = False,
    full_matrices: bool = False,
) -> SVD:
    r"""
    Perform singular value decomposition on the last two tensor dimensions.

    This function applies [`torch.linalg.svd`](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html)
    to the matrix axes of the input tensor and returns symbolic dimensions that
    distinguish reduced and full factorizations. The last two dimensions may be
    rectangular. Any leading dimensions are treated as batch dimensions and are
    preserved in all outputs.

    Parameters
    ----------
    tensor : Tensor
        Input tensor whose last two dimensions form matrices.
    values_as_matrix : bool, default `False`
        If `True`, return singular values as an explicit diagonal matrix
        tensor. If `False`, return them as a vector on a single spectral axis.
    full_matrices : bool, default `False`
        If `True`, compute full-sized `U` and `Vh`. If `False`, compute the
        reduced SVD.

    Returns
    -------
    SVD
        [`SVD`][qten.linalg.decompose.SVD] containing:
        - `U`, with dims `(..., row_dim, factor)` for reduced SVD or
          `(..., row_dim, left_factor)` for full SVD.
        - `S`, with dims `(..., factor)` by default, `(..., factor, factor)`
          when `values_as_matrix=True` in reduced mode, or
          `(..., left_factor, right_factor)` in full matrix form.
        - `Vh`, with dims `(..., factor, col_dim)` for reduced SVD or
          `(..., right_factor, col_dim)` for full SVD.

    Examples
    --------
    ```python
    result = svd(tensor)
    u = result.U
    s = result.S
    vh = result.Vh
    ```

    Notes
    -----
    In reduced mode, `factor` is the shared singular-value
    [`IndexSpace`][qten.symbolics.state_space.IndexSpace]. In full mode,
    `left_factor` and `right_factor` are sized to the full row and column
    spaces of the input matrix axes. The original matrix is recovered as
    \(U\Sigma V^\dagger\), using `U @ Sigma @ Vh` in code. Here `Sigma` is
    either the returned `S` tensor (`values_as_matrix=True`) or the diagonal
    matrix formed from the returned singular-value vector.
    """
    if tensor.rank() < 2:
        raise ValueError(
            "Input tensor must have at least two dimensions for matrix decomposition."
        )

    row_dim = tensor.dims[-2]
    col_dim = tensor.dims[-1]

    u_data, s_data, vh_data = torch.linalg.svd(tensor.data, full_matrices=full_matrices)

    factor = IndexSpace.linear(s_data.shape[-1])

    if full_matrices:
        left_factor = IndexSpace.linear(row_dim.dim)
        right_factor = IndexSpace.linear(col_dim.dim)
        u = Tensor(
            data=u_data,
            dims=tensor.dims[:-2] + (row_dim, left_factor),
        )
    else:
        u = Tensor(
            data=u_data,
            dims=tensor.dims[:-2] + (row_dim, factor),
        )
    if values_as_matrix:
        if full_matrices:
            k = s_data.shape[-1]
            s_mat = torch.zeros(
                *s_data.shape[:-1],
                left_factor.dim,
                right_factor.dim,
                dtype=s_data.dtype,
                device=s_data.device,
            )
            diag = torch.diag_embed(s_data)
            s_mat[..., :k, :k] = diag
            s = Tensor(
                data=s_mat,
                dims=tensor.dims[:-2] + (left_factor, right_factor),
            )
        else:
            s_mat = torch.diag_embed(s_data)
            s = Tensor(
                data=s_mat,
                dims=tensor.dims[:-2] + (factor, factor),
            )
    else:
        s = Tensor(
            data=s_data,
            dims=tensor.dims[:-2] + (factor,),
        )
    if full_matrices:
        vh = Tensor(
            data=vh_data,
            dims=tensor.dims[:-2] + (right_factor, col_dim),
        )
    else:
        vh = Tensor(
            data=vh_data,
            dims=tensor.dims[:-2] + (factor, col_dim),
        )

    return SVD(u, s, vh)
