from typing import Any, overload
import torch

from ..linalg.tensors import Tensor as Tensor
from ..symbolics.hilbert_space import HilbertSpace as HilbertSpace
from ..symbolics.state_space import (
    MomentumSpace as MomentumSpace,
    StateSpace as StateSpace,
)
from .spatials import Momentum as Momentum, Offset as Offset

@overload
def fourier_transform(
    K: tuple[Momentum, ...], R: tuple[Offset, ...]
) -> torch.Tensor: ...
@overload
def fourier_transform(
    k_space: MomentumSpace,
    bloch_space: HilbertSpace,
    region_space: HilbertSpace,
) -> Tensor: ...
@overload
def fourier_transform(
    k_space: MomentumSpace,
    bloch_space: StateSpace[Any],
    region_space: HilbertSpace,
) -> Tensor: ...
