from typing import Dict, Tuple
from typing import cast

from multipledispatch import dispatch

import numpy as np
import torch
from ..precision import get_precision_config

from .spatials import Momentum, Offset
from ..state_space import MomentumSpace
from ..hilbert_space import HilbertSpace, U1Basis
from ..tensors import Tensor
from ..tensors import mapping_matrix
from ..utils.collections_ext import matchby


@dispatch(tuple, tuple)
def fourier_transform(K: Tuple[Momentum, ...], R: Tuple[Offset, ...]) -> torch.Tensor:
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
    ten_K = torch.from_numpy(  # (K, d)
        np.stack(
            [
                np.array(k.to_vec(np.ndarray), dtype=precision.np_float).reshape(-1)
                for k in K
            ],
            axis=0,
        )
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
    )
    # `ten_K` is already in Cartesian reciprocal coordinates (includes 2π),
    # so multiplying by `2π` here would double count the phase.
    exponent = -1j * torch.matmul(ten_K, ten_R)  # (K, R)
    return torch.exp(exponent)  # (K, R)


@dispatch(MomentumSpace, HilbertSpace, HilbertSpace)  # type: ignore[no-redef]
def fourier_transform(
    k_space: MomentumSpace,
    bloch_space: HilbertSpace,
    region_space: HilbertSpace,
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
    K: Tuple[Momentum] = k_space.elements()
    R: Tuple[Offset] = tuple(
        cast(U1Basis, el).irrep_of(Offset) for el in region_space.elements()
    )
    f = fourier_transform(K, R)  # (K, R)

    region_to_bloch: Dict[U1Basis, U1Basis] = matchby(
        region_space,
        bloch_space,
        lambda psi: cast(U1Basis, psi).irrep_of(Offset).fractional(),
    )

    map = mapping_matrix(region_space, bloch_space, region_to_bloch).transpose(
        0, 1
    )  # (B, R)
    # (K, 1, R) * (1, B, R)
    f = f.unsqueeze(1) * map.data.unsqueeze(0)
    return Tensor(data=f, dims=(k_space, bloch_space, region_space))  # (K, B, R)
