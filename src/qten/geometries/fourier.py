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


# TODO: Consider allow the creation of the tensor on a specified device.
@multimethod
def fourier_transform(
    K: Tuple[Momentum, ...], R: Tuple[Offset, ...], *, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Compute Fourier phase factors between momentum and real-space offsets.

    This returns the Fourier kernel evaluated for all pairs of momentum points
    in `K` and offsets in `R`.

    `Momentum.to_vec()` uses the reciprocal basis (which already carries the
    `2π` convention via `Lattice.dual`), so the phase is computed as
    `exp(-i k_cart·r_cart)`. Equivalently, in fractional coordinates this is
    `exp(-2π i κ·n)`.

    Parameters
    ----------
    `K` : `Tuple[Momentum, ...]`
        Momentum points.
    `R` : `Tuple[Offset, ...]`
        Real-space offsets.

    Returns
    -------
    `torch.Tensor`
        Complex tensor of shape `(len(K), len(R))` with elements
        `exp(-i k_cart·r_cart)` (equivalently `exp(-2π i κ·n)` in fractional
        coordinates).
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


@fourier_transform.register
def _(
    k_space: MomentumSpace,
    bloch_space: HilbertSpace,
    region_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> Tensor:
    """
    Build the Fourier transform tensor between `k_space` and `region_space`.

    This computes phase factors for `k_space` against the offsets collected
    from `region_space`, then maps region modes into `bloch_space` using
    the coordinate named by `r_name`.

    Parameters
    ----------
    `k_space` : `MomentumSpace`
        Momentum space defining the k points.
    `bloch_space` : `HilbertSpace`
        Bloch space to map region modes into.
    `region_space` : `HilbertSpace`
        Real-space region defining offsets.

    Returns
    -------
    `Tensor`
        Tensor with data shape `(K, B, R)` and dims
        `(k_space, bloch_space, region_space)`.
    """
    K: Tuple[Momentum, ...] = k_space.elements()
    R: Tuple[Offset, ...] = tuple(
        cast(U1Basis, el).irrep_of(Offset) for el in region_space.elements()
    )
    f = fourier_transform(K, R, device=device)  # (K, R)

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
    `region_restrict(tensor, R)`
        Rebuild the Fourier transform tensor on the target real-space
        `HilbertSpace` `R`, reusing the momentum and Bloch spaces from
        `tensor`.

    `region_restrict(tensor, region)`
        Accept a tuple of `Offset` values, construct the corresponding real-
        space `HilbertSpace` via `region_hilbert`, then rebuild the transform
        on that region.

    Parameters
    ----------
    `tensor` : `Tensor`
        Fourier transform tensor whose dims are expected to be
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    `R` : `HilbertSpace`
        Target real-space region for the form
        `region_restrict(tensor, R)`.
    `region` : `tuple[Offset, ...]`
        Offsets defining the target real-space region for the form
        `region_restrict(tensor, region)`. These are converted to a
        `HilbertSpace` via `region_hilbert` before rebuilding the transform.

    Returns
    -------
    `Tensor`
        Tensor with dims `(tensor.dims[0], tensor.dims[1], R)`, where `R` is
        either the provided Hilbert space or the Hilbert space generated from
        `region`.
    """
    ...


@overload
def region_restrict(
    tensor: Tensor, region: Tuple[Offset, ...], *, device: Optional[Device] = None
) -> Tensor: ...


@multimethod
def region_restrict(
    tensor: Tensor, R: HilbertSpace, *, device: Optional[Device] = None
) -> Tensor:
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
def _(
    tensor: Tensor, region: Tuple[Offset, ...], *, device: Optional[Device] = None
) -> Tensor:
    B = tensor.dims[1]
    if not isinstance(B, HilbertSpace):
        raise TypeError(
            f"Expected second dim to be HilbertSpace, got {type(B).__name__}"
        )
    R = region_hilbert(B, region)
    return region_restrict(tensor, R, device=device)
