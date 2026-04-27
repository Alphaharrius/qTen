r"""
Fourier-transform helpers connecting finite real-space and momentum-space geometry.

This module provides the Fourier phase-factor machinery used to move between
discrete momentum points and finite real-space offsets in QTen. Its core role
is to build the finite Fourier kernel associated with a bounded lattice and to
package that kernel into
[`Tensor`][qten.linalg.tensors.Tensor] objects whose legs are labeled by the
repository's symbolic
[`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] and
[`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] objects.

Two related APIs are defined here:

- [`fourier_kernel`][qten.geometries.fourier.fourier_kernel] computes the raw
  phase matrix \(\exp(-\mathrm{i}\, k \cdot r)\) for momentum points and
  real-space offsets.
- [`fourier_transform`][qten.geometries.fourier.fourier_transform] lifts that
  kernel into a labeled `(K, B, R)` tensor that maps a region basis into a
  Bloch basis.
- [`region_restrict`][qten.geometries.fourier.region_restrict] rebuilds an
  existing Fourier-transform tensor on a different real-space region while
  preserving the momentum and Bloch-space structure.

The implementation follows the repository's reciprocal-lattice convention:
[`Momentum.to_vec()`][qten.geometries.spatials.Momentum.to_vec] already uses
Cartesian reciprocal coordinates containing the \(2\pi\) factor induced by
[`Lattice.dual`][qten.geometries.spatials.Lattice.dual]. As a result, the
Fourier phase is evaluated as
\(\exp(-\mathrm{i}\, k_{\mathrm{cart}}\cdot r_{\mathrm{cart}})\), which is
equivalent to \(\exp(-2\pi\mathrm{i}\,\kappa\cdot n)\) in fractional
direct/reciprocal coordinates. In code, the exponent is assembled from the
Cartesian arrays as `-1j * torch.matmul(ten_K, ten_R)`.

In matrix form, the sampled kernel has entries
\(K_{\alpha\beta} = \exp(-\mathrm{i}\, k_\alpha \cdot r_\beta)
= \exp(-2\pi\mathrm{i}\, \kappa_\alpha \cdot n_\beta)\).

Repository usage
----------------
This module sits at the junction of geometry and tensor assembly:

- Finite-region Hilbert spaces contribute the real-space offsets whose phases
  are sampled against a discrete
  [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace].
- Bloch-space labeling is recovered through
  [`mapping_matrix`][qten.linalg.tensors.mapping_matrix], allowing the raw
  Fourier kernel to be embedded into a tensor with explicit symbolic legs.
- Region-changing workflows can rebuild the rightmost real-space leg without
  modifying the momentum grid or Bloch labeling by calling
  [`region_restrict`][qten.geometries.fourier.region_restrict].

Notes
-----
The functions in this module assume a finite sampled momentum set and a finite
real-space region supplied by the surrounding geometry/symbolics layers. They
therefore implement the discrete Fourier transform conventions used by the
repository's bounded-lattice workflows rather than a continuum transform API.
"""

from ..utils.devices import Device
from typing import Dict, Tuple, overload, Optional
from typing import cast

from multimethod import multimethod

import numpy as np
import torch
from ..precision import get_precision_config

from . import Momentum, Offset
from ..symbolics import HilbertSpace, MomentumSpace, U1Basis, region_hilbert
from ..linalg.tensors import Tensor
from ..linalg.tensors import mapping_matrix
from ..utils.collections_ext import matchby


def fourier_kernel(
    K: Tuple[Momentum, ...], R: Tuple[Offset, ...], *, device: Optional[Device] = None
) -> torch.Tensor:
    r"""
    Compute the raw discrete Fourier kernel for momentum points and offsets.

    `Momentum.to_vec()` uses the reciprocal basis, which already carries the
    \(2\pi\) convention via `Lattice.dual`, so the phase is computed as
    \(\exp(-\mathrm{i}\, k_{\mathrm{cart}}\cdot r_{\mathrm{cart}})\).
    Equivalently, in fractional coordinates this is
    \(\exp(-2\pi\mathrm{i}\,\kappa\cdot n)\).
    In code this is the `torch.exp` of `-1j * torch.matmul(ten_K, ten_R)`.

    The returned matrix is
    \(K_{\alpha\beta} = \exp(-\mathrm{i}\, k_\alpha \cdot r_\beta)\).

    Parameters
    ----------
    K : Tuple[Momentum, ...]
        Momentum points for the raw-kernel form
        [`fourier_kernel(K, R)`][qten.geometries.fourier.fourier_kernel].
    R : Tuple[Offset, ...]
        Real-space offsets for the raw-kernel form
        [`fourier_kernel(K, R)`][qten.geometries.fourier.fourier_kernel].

    Returns
    -------
    torch.Tensor
        Complex tensor of shape `(len(K), len(R))` with elements
        \(\exp(-\mathrm{i}\, k_{\mathrm{cart}}\cdot r_{\mathrm{cart}})\)
        (equivalently \(\exp(-2\pi\mathrm{i}\,\kappa\cdot n)\) in
        fractional coordinates).
    """
    precision = get_precision_config()
    torch_device = device.torch_device() if device is not None else None

    # Batch-extract K Cartesian coordinates via numpy matrix multiply
    # instead of per-element sympy to_vec calls.
    k_space = K[0].space
    k_dim = k_space.dim
    k_basis_np = np.array(k_space.basis.evalf(), dtype=precision.np_float)
    k_frac = np.array(
        [[float(k.rep[j, 0]) for j in range(k_dim)] for k in K],
        dtype=precision.np_float,
    )
    ten_K = torch.from_numpy(k_frac @ k_basis_np.T).to(  # (K, d)
        device=torch_device
    )

    ten_R = torch.from_numpy(  # (d, R)
        np.stack(
            [
                np.array(
                    (r - r.fractional()).to_vec(np.ndarray), dtype=precision.np_float
                ).reshape(-1)
                for r in R
            ],
            axis=1,
        )
    ).to(device=torch_device)
    # `ten_K` is already in Cartesian reciprocal coordinates (includes 2π),
    # so multiplying by `2π` here would double count the phase.
    exponent = -1j * torch.matmul(ten_K, ten_R)  # (K, R)
    return torch.exp(exponent)  # (K, R)


def fourier_transform(
    k_space: MomentumSpace,
    bloch_space: HilbertSpace,
    region_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> Tensor:
    """
    Build the labeled Fourier transform tensor for symbolic Hilbert spaces.

    This function is the high-level symbolic wrapper around
    [`fourier_kernel`][qten.geometries.fourier.fourier_kernel]. It enumerates
    momentum points from `k_space`, collects real-space offsets from
    `region_space`, evaluates the raw Fourier kernel, and maps region modes
    into `bloch_space` with
    [`mapping_matrix`][qten.linalg.tensors.mapping_matrix].

    Parameters
    ----------
    k_space : MomentumSpace
        Momentum space defining the k points.
    bloch_space : HilbertSpace
        Bloch space to map region modes into.
    region_space : HilbertSpace
        Real-space region defining offsets.

    Returns
    -------
    Tensor
        Tensor with data shape `(K, B, R)` and dims
        (k_space, bloch_space, region_space).

    See Also
    --------
    [`fourier_kernel`][qten.geometries.fourier.fourier_kernel]
        Low-level Fourier phase matrix used internally by this function.
    """
    K: Tuple[Momentum, ...] = k_space.elements()
    R: Tuple[Offset, ...] = tuple(
        cast(U1Basis, el).irrep_of(Offset) for el in region_space.elements()
    )
    f = fourier_kernel(K, R, device=device)  # (K, R)

    region_to_bloch: Dict[U1Basis, U1Basis] = matchby(
        region_space,
        bloch_space,
        lambda psi: cast(U1Basis, psi).irrep_of(Offset).fractional(),
    )

    map = mapping_matrix(
        region_space, bloch_space, region_to_bloch, device=device
    ).transpose(0, 1)  # (B, R)
    # (K, 1, R) * (1, B, R)
    f = f.to(map.data.device).unsqueeze(1) * map.data.unsqueeze(0)
    return Tensor(data=f, dims=(k_space, bloch_space, region_space))  # (K, B, R)


@overload
def region_restrict(
    tensor: Tensor, R: HilbertSpace, *, device: Optional[Device] = None
) -> Tensor:
    """
    Rebuild a Fourier transform tensor on a different real-space region.

    Supported forms
    ---------------
    [`region_restrict(tensor, R)`][qten.geometries.fourier.region_restrict]
        Rebuild the Fourier transform tensor on the target real-space
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] `R`, reusing the momentum and Bloch spaces from
        `tensor`.

    [`region_restrict(tensor, region)`][qten.geometries.fourier.region_restrict]
        Accept a tuple of [`Offset`][qten.geometries.spatials.Offset] values, construct the corresponding real-
        space [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] via [`region_hilbert`][qten.symbolics.ops.region_hilbert], then rebuild the transform
        on that region.

    Parameters
    ----------
    tensor : Tensor
        Fourier transform tensor whose dims are expected to be
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    R : HilbertSpace
        Target real-space region for the form
        [`region_restrict(tensor, R)`][qten.geometries.fourier.region_restrict].
    region : tuple[Offset, ...]
        Offsets defining the target real-space region for the form
        [`region_restrict(tensor, region)`][qten.geometries.fourier.region_restrict]. These are converted to a
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] via [`region_hilbert`][qten.symbolics.ops.region_hilbert] before rebuilding the transform.

    Returns
    -------
    Tensor
        Tensor with dims `(tensor.dims[0], tensor.dims[1], R)`, where `R` is
        either the provided Hilbert space or the Hilbert space generated from
        region.
    """
    ...


@overload
def region_restrict(
    tensor: Tensor, region: Tuple[Offset, ...], *, device: Optional[Device] = None
) -> Tensor:
    """
    Rebuild a Fourier transform tensor on a region specified by offsets.

    This overload accepts a tuple of [`Offset`][qten.geometries.spatials.Offset]
    values, converts it into the matching real-space
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace], and then
    dispatches to the main [`region_restrict`][qten.geometries.fourier.region_restrict]
    implementation.
    """
    ...


@multimethod
def region_restrict(
    tensor: Tensor, R: HilbertSpace, *, device: Optional[Device] = None
) -> Tensor:
    """
    Rebuild a Fourier transform tensor on a different real-space region.

    Supported forms
    ---------------
    [`region_restrict(tensor, R)`][qten.geometries.fourier.region_restrict]
        Rebuild the Fourier transform tensor on the target real-space
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] `R`,
        reusing the momentum and Bloch spaces from `tensor`.

    [`region_restrict(tensor, region)`][qten.geometries.fourier.region_restrict]
        Accept a tuple of
        [`Offset`][qten.geometries.spatials.Offset] values, construct the
        corresponding real-space
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] via
        [`region_hilbert`][qten.symbolics.ops.region_hilbert], then rebuild
        the transform on that region. Here `region` means the target finite
        real-space region written explicitly as an ordered tuple of offsets,
        for example `(r0, r1, r2)`, rather than as a preconstructed
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].

    Parameters
    ----------
    tensor : Tensor
        Fourier transform tensor whose first two dims are expected to be
        `(MomentumSpace, HilbertSpace)`.
    R : HilbertSpace
        Target real-space region used as the new rightmost dimension for the
        form [`region_restrict(tensor, R)`][qten.geometries.fourier.region_restrict].
    device : Optional[Device], optional
        Device on which to construct the rebuilt transform.

    Returns
    -------
    Tensor
        Fourier transform tensor with dims `(K, B, R)` where `K` and `B` are
        taken from `tensor`.

    Raises
    ------
    TypeError
        If the first two dims of `tensor` are not `MomentumSpace` and
        `HilbertSpace`, respectively.

    Notes
    -----
    The generated API docs for this module show overload signatures, but the
    prose is rendered from this public implementation docstring. The overload
    accepting a tuple of [`Offset`][qten.geometries.spatials.Offset] first
    converts that tuple into a
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] and then
    dispatches here. In that overload, `region` is the explicit tuple of
    target real-space offsets.
    """
    K, B, _ = tensor.dims
    if not isinstance(K, MomentumSpace):
        raise TypeError(
            f"Expected first dim to be MomentumSpace, got {type(K).__name__}"
        )
    if not isinstance(B, HilbertSpace):
        raise TypeError(
            f"Expected second dim to be HilbertSpace, got {type(B).__name__}"
        )
    return fourier_transform(K, B, R, device=device)


@region_restrict.register
def _region_restrict_from_offsets(
    tensor: Tensor, region: Tuple[Offset, ...], *, device: Optional[Device] = None
) -> Tensor:
    B = tensor.dims[1]
    if not isinstance(B, HilbertSpace):
        raise TypeError(
            f"Expected second dim to be HilbertSpace, got {type(B).__name__}"
        )
    R = region_hilbert(B, region)
    return region_restrict(tensor, R, device=device)
