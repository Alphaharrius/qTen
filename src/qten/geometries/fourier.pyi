from typing import Any, overload, Optional
import torch

from ..utils.devices import Device as Device
from ..linalg.tensors import Tensor as Tensor
from ..symbolics.hilbert_space import HilbertSpace as HilbertSpace
from ..symbolics.state_space import (
    MomentumSpace as MomentumSpace,
    StateSpace as StateSpace,
)
from .spatials import Momentum as Momentum, Offset as Offset

@overload
def fourier_transform(
    K: tuple[Momentum, ...], R: tuple[Offset, ...], *, device: Optional[Device] = None
) -> torch.Tensor: ...
@overload
def fourier_transform(
    k_space: MomentumSpace,
    bloch_space: HilbertSpace,
    region_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> Tensor: ...
@overload
def fourier_transform(
    k_space: MomentumSpace,
    bloch_space: StateSpace[Any],
    region_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> Tensor: ...
@overload
def region_restrict(
    tensor: Tensor, R: HilbertSpace, *, device: Optional[Device] = None
) -> Tensor: ...
@overload
def region_restrict(
    tensor: Tensor, region: tuple[Offset, ...], *, device: Optional[Device] = None
) -> Tensor: ...
